#!/usr/bin/env python3
"""Build a fail-closed Null-TTA payload from a committed partial remote shard.

This is used only when Modal committed lossless image pairs and the target
PickScore CSV before an operator-cancelled app terminated.  The payload is
then completed by ``score_null_tta_images.py`` on a remote GPU.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


OFFICIAL_COMMIT = "337bf73037e9f24e9f844974d3287384abb610bf"
EXPECTED_CONFIG = {
    "min_inner_steps": 5,
    "max_inner_steps": 55,
    "num_particles": 3,
    "num_inference_steps": 100,
    "lambda_alpha": 100.0,
    "lambda_reg": 0.002,
    "phi_variance": 0.01,
    "lr_uncond": 0.01,
    "tampering_coef": 0.008,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--official-output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--prompt-start", required=True, type=int)
    parser.add_argument("--prompt-end", required=True, type=int)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--source-app-id", required=True)
    parser.add_argument("--source-profile", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: list[float]) -> dict[str, float | int]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Metric vector is empty or non-finite")
    sample_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean(values),
        "std": sample_std,
        "n": len(values),
        "se": sample_std / math.sqrt(len(values)),
    }


def official_prompts(example_path: Path) -> list[str]:
    tree = ast.parse(example_path.read_text(encoding="utf-8"), filename=str(example_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "prompt_list" for target in node.targets):
            prompts = ast.literal_eval(node.value)
            if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
                break
            return prompts
    raise RuntimeError(f"Could not recover literal prompt_list from {example_path}")


def finite(row: dict[str, str], field: str, row_index: int) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} at CSV row {row_index}")
    return value


def main() -> int:
    args = parse_args()
    if not (0 <= args.prompt_start < args.prompt_end <= 50):
        raise ValueError("Recovery slice must be within the official 50-prompt matrix")
    output = args.official_output.resolve()
    upstream = args.upstream_root.resolve()
    example_path = upstream / "examples/null_tta_sd.py"
    prompts = official_prompts(example_path)
    selected = prompts[args.prompt_start : args.prompt_end]
    if len(prompts) != 50 or len(selected) != args.prompt_end - args.prompt_start:
        raise RuntimeError("Official prompt-list cardinality drift")

    csv_paths = sorted(output.rglob("results_pickscore.csv"))
    if len(csv_paths) != 1:
        raise RuntimeError(f"Expected one committed PickScore CSV, found {csv_paths}")
    csv_path = csv_paths[0]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row.get("prompt_idx") != "avg"]
    if len(rows) != len(selected):
        raise RuntimeError(f"Expected {len(selected)} complete rows, found {len(rows)}")

    per_prompt: list[dict[str, Any]] = []
    source_files = [csv_path]
    for local_index, (row, expected_prompt) in enumerate(zip(rows, selected, strict=True)):
        if int(row["prompt_idx"]) != local_index or row["prompt"] != expected_prompt:
            raise RuntimeError(f"Committed row does not match official prompt {args.prompt_start + local_index}")
        base_path = csv_path.parent / f"base_prompt_{local_index:03d}_target-pickscore.png"
        opt_path = csv_path.parent / f"opt_prompt_{local_index:03d}_target-pickscore.png"
        if not base_path.is_file() or not opt_path.is_file():
            raise FileNotFoundError(f"Missing lossless recovery pair for local prompt {local_index}")
        source_files.extend((base_path, opt_path))
        baseline = finite(row, "baseline_pick", local_index)
        optimized = finite(row, "optimized_pick", local_index)
        per_prompt.append(
            {
                "global_prompt_index": args.prompt_start + local_index,
                "prompt": expected_prompt,
                "baseline": baseline,
                "optimized": optimized,
                "improvement": optimized - baseline,
                "metrics": {
                    "pickscore": {
                        "baseline": baseline,
                        "optimized": optimized,
                        "improvement": optimized - baseline,
                    },
                    "aesthetic": None,
                    "hpsv2": None,
                    "imagereward": None,
                },
            }
        )

    pick = {
        field: stats([float(row["metrics"]["pickscore"][field]) for row in per_prompt])
        for field in ("baseline", "optimized", "improvement")
    }
    payload = {
        "schema_version": 1,
        "paper": "Null-TTA",
        "paper_venue": "CVPR 2026",
        "upstream_commit": OFFICIAL_COMMIT,
        "official_example_sha256": sha256(example_path),
        "scope": "recovered complete prefix of an official SD-v1.5 remote shard",
        "seed": args.seed,
        "prompt_slice": [args.prompt_start, args.prompt_end],
        "prompt_count": len(per_prompt),
        "target_reward": "pickscore",
        "config": EXPECTED_CONFIG,
        "scorers": {
            "pickscore": {
                "type": "PickScoreScorer",
                "module": "das.rewards",
                "is_mock": False,
                "is_disabled": False,
                "provenance": "official generation CSV from source app",
            },
            **{
                metric: {
                    "type": "DisabledEvaluationScorer",
                    "module": "recovery_pending_remote_postscore",
                    "is_mock": False,
                    "is_disabled": True,
                }
                for metric in ("aesthetic", "hpsv2", "imagereward")
            },
        },
        "require_all_scorers": False,
        "target_only_generation": True,
        "all_scorers_are_real": False,
        "target_scorer_is_real": True,
        "metrics": {
            "pickscore": pick,
            "aesthetic": None,
            "hpsv2": None,
            "imagereward": None,
        },
        "target_metric": pick,
        "per_prompt": per_prompt,
        "wall_seconds": None,
        "wall_seconds_scope": "source generation unavailable; set to recovery post-scoring time after scoring",
        "cuda_max_memory_allocated_mb": None,
        "result_csv": str(csv_path.relative_to(output)),
        "recovery_provenance": {
            "source_run_id": args.source_run_id,
            "source_app_id": args.source_app_id,
            "source_profile": args.source_profile,
            "source_terminal_status": "operator_cancelled_after_partial_volume_commit",
            "transport": "lossless PNG and CSV copied between Modal workspaces via local orchestration only",
            "source_files": [
                {"path": str(path.relative_to(output)), "sha256": sha256(path), "bytes": path.stat().st_size}
                for path in source_files
            ],
        },
    }
    metrics_path = output / "null_tta_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("NULL_TTA_RECOVERY_PAYLOAD_JSON=" + json.dumps(payload["recovery_provenance"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
