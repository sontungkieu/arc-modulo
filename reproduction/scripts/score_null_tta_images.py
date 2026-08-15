#!/usr/bin/env python3
"""Sequentially evaluate saved Null-TTA images with the three deferred scorers."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Scorer produced an empty or non-finite metric vector")
    sample_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean(values),
        "std": sample_std,
        "n": len(values),
        "se": sample_std / math.sqrt(len(values)),
    }


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> int:
    args = parse_args()
    metrics_path = args.metrics.resolve()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not payload.get("target_only_generation"):
        raise RuntimeError("Sequential post-scoring requires a target-only generation payload")
    if payload.get("target_reward") != "pickscore" or not payload.get("target_scorer_is_real"):
        raise RuntimeError("Expected a real PickScore-target generation payload")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for official Table 1 scorers")

    upstream = args.upstream_root.resolve()
    sys.path.insert(0, str(upstream))
    import das.rewards as rewards

    result_csv = metrics_path.parent / payload["result_csv"]
    image_dir = result_csv.parent
    device = torch.device("cuda")
    factories: list[tuple[str, str, Callable[..., Any]]] = [
        ("aesthetic", "aesthetic_score", rewards.aesthetic_score),
        ("hpsv2", "hps_score", rewards.hps_score),
        ("imagereward", "ImageReward", rewards.ImageReward),
    ]
    phase_records: list[dict[str, Any]] = []
    postscore_started = time.perf_counter()

    for metric_name, factory_name, factory in factories:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        phase_started = time.perf_counter()
        scorer = factory(device=device)
        if hasattr(scorer, "eval"):
            scorer.eval()
        for local_index, row in enumerate(payload["per_prompt"]):
            base_path = image_dir / f"base_prompt_{local_index:03d}_target-pickscore.png"
            opt_path = image_dir / f"opt_prompt_{local_index:03d}_target-pickscore.png"
            if not base_path.is_file() or not opt_path.is_file():
                raise FileNotFoundError(f"Missing saved image pair: {base_path}, {opt_path}")
            prompt = row["prompt"]
            with torch.no_grad():
                base_score = float(scorer(load_image_tensor(base_path, device), [prompt])[0].item())
                opt_score = float(scorer(load_image_tensor(opt_path, device), [prompt])[0].item())
            if not math.isfinite(base_score) or not math.isfinite(opt_score):
                raise RuntimeError(f"Non-finite {metric_name} score at local prompt {local_index}")
            row["metrics"][metric_name] = {
                "baseline": base_score,
                "optimized": opt_score,
                "improvement": opt_score - base_score,
            }

        metric_rows = [row["metrics"][metric_name] for row in payload["per_prompt"]]
        payload["metrics"][metric_name] = {
            field: stats([float(row[field]) for row in metric_rows])
            for field in ("baseline", "optimized", "improvement")
        }
        payload["scorers"][metric_name] = {
            "type": "official_sequential_scorer",
            "module": "das.rewards",
            "factory": factory_name,
            "is_mock": False,
            "is_disabled": False,
        }
        phase_records.append(
            {
                "metric": metric_name,
                "factory": f"das.rewards.{factory_name}",
                "wall_seconds": time.perf_counter() - phase_started,
                "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            }
        )
        del scorer
        gc.collect()
        torch.cuda.empty_cache()

    # PickScore was evaluated in the official generation program.  The other
    # three metrics were evaluated from its losslessly saved PNG image pairs.
    payload["scorers"]["pickscore"]["is_disabled"] = False
    payload["require_all_scorers"] = True
    payload["all_scorers_are_real"] = True
    payload["sequential_postscoring"] = {
        "enabled": True,
        "reason": "keep only the optimization target resident during generation on a 24 GB GPU",
        "image_encoding": "lossless PNG produced by the official example",
        "phases": phase_records,
        "wall_seconds": time.perf_counter() - postscore_started,
    }
    if payload.get("recovery_provenance"):
        payload["wall_seconds"] = payload["sequential_postscoring"]["wall_seconds"]
        payload["wall_seconds_scope"] = "remote recovery post-scoring only; source generation duration unavailable"
    payload["cuda_max_memory_allocated_mb"] = max(
        [float(payload.get("cuda_max_memory_allocated_mb") or 0.0)]
        + [float(item["cuda_max_memory_allocated_mb"]) for item in phase_records]
    )
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("NULL_TTA_POSTSCORE_JSON=" + json.dumps(payload["sequential_postscoring"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
