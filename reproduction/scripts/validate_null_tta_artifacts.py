#!/usr/bin/env python3
"""Validate the exact remote Null-TTA shard matrix and persist provenance."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_COMMIT = "337bf73037e9f24e9f844974d3287384abb610bf"
EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_PROMPTS = set(range(50))
EXPECTED_SHARDS = {42: 8, 43: 8, 44: 9}
RECOVERY_SOURCE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "remote_fragments/null-tta-table1-s44-p07-11-source/official"
)
OFFICIAL_EXAMPLE = Path(__file__).resolve().parents[1] / "upstream/null-tta/examples/null_tta_sd.py"
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
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_prompts() -> list[str]:
    tree = ast.parse(OFFICIAL_EXAMPLE.read_text(encoding="utf-8"), filename=str(OFFICIAL_EXAMPLE))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "prompt_list" for target in node.targets
        ):
            prompts = ast.literal_eval(node.value)
            if isinstance(prompts, list) and len(prompts) == 50 and all(isinstance(item, str) for item in prompts):
                return prompts
    raise RuntimeError("Could not load the official 50-prompt literal")


def main() -> int:
    args = parse_args()
    if len(args.artifacts) != 25:
        raise RuntimeError(f"Expected exactly 25 canonical artifact directories, got {len(args.artifacts)}")

    coverage = {seed: set() for seed in EXPECTED_SEEDS}
    prompts = load_official_prompts()
    run_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for artifact_dir in args.artifacts:
        artifact_dir = artifact_dir.resolve()
        metrics_path = artifact_dir / "null_tta_metrics.json"
        summary_path = artifact_dir / "modal_summary.json"
        if not metrics_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(f"Missing metrics/summary pair under {artifact_dir}")
        metrics = load(metrics_path)
        summary = load(summary_path)

        run_id = str(summary["run_id"])
        seed = int(metrics["seed"])
        start, end = (int(value) for value in metrics["prompt_slice"])
        if run_id in run_ids:
            raise RuntimeError(f"Duplicate run id: {run_id}")
        run_ids.add(run_id)
        if seed not in EXPECTED_SEEDS or not (0 <= start < end <= 50):
            raise RuntimeError(f"Unexpected seed/slice in {artifact_dir}: {seed}, {[start, end]}")
        prompt_indices = {int(row["global_prompt_index"]) for row in metrics["per_prompt"]}
        expected_slice = set(range(start, end))
        if prompt_indices != expected_slice or metrics["prompt_count"] != end - start:
            raise RuntimeError(f"Prompt rows do not match slice for {run_id}")
        overlap = coverage[seed] & prompt_indices
        if overlap:
            raise RuntimeError(f"Overlapping prompts for seed {seed}: {sorted(overlap)}")
        coverage[seed].update(prompt_indices)

        if metrics["upstream_commit"] != EXPECTED_COMMIT or metrics["config"] != EXPECTED_CONFIG:
            raise RuntimeError(f"Commit/config drift for {run_id}")
        if metrics["target_reward"] != "pickscore" or not metrics["target_scorer_is_real"]:
            raise RuntimeError(f"Target scorer contract failed for {run_id}")
        if not metrics["all_scorers_are_real"] or any(
            item["is_mock"] or item["is_disabled"] for item in metrics["scorers"].values()
        ):
            raise RuntimeError(f"Four-real-scorer contract failed for {run_id}")
        for row in metrics["per_prompt"]:
            prompt_index = int(row["global_prompt_index"])
            if row["prompt"] != prompts[prompt_index]:
                raise RuntimeError(f"Official prompt-text drift at seed/index {(seed, prompt_index)}")
            for metric_name in ("pickscore", "hpsv2", "aesthetic", "imagereward"):
                metric = row["metrics"][metric_name]
                if any(not math.isfinite(float(metric[field])) for field in ("baseline", "optimized", "improvement")):
                    raise RuntimeError(f"Non-finite {metric_name} value in {run_id}")
        if summary["status"] != "COMPLETED" or summary["seed"] != seed:
            raise RuntimeError(f"Remote completion contract failed for {run_id}")
        if [int(value) for value in summary["prompt_slice"]] != [start, end]:
            raise RuntimeError(f"Summary slice drift for {run_id}")

        expected_sequential = seed == 44
        allowed_gpus = {"L40S"} if seed in (42, 43) else {"L4", "L40S"}
        if summary["gpu_requested"] not in allowed_gpus:
            raise RuntimeError(f"GPU request drift for {run_id}")
        if bool(summary["sequential_scorers"]) != expected_sequential:
            raise RuntimeError(f"Scoring-path drift for {run_id}")
        if bool(metrics["target_only_generation"]) != expected_sequential:
            raise RuntimeError(f"Generation-path drift for {run_id}")

        recovered = bool(summary.get("recovered_partial_shard"))
        recovery_record: dict[str, Any] | None = None
        if recovered:
            provenance = metrics.get("recovery_provenance")
            expected_source = {
                "source_run_id": "null-tta-table1-s44-p07-14-sequential-20260811T203200Z",
                "source_app_id": "ap-MPGcs8W1M9nPCsEWQkC3sv",
                "source_profile": "phamvanvuhoan",
                "source_terminal_status": "operator_cancelled_after_partial_volume_commit",
            }
            if seed != 44 or [start, end] != [7, 11] or not isinstance(provenance, dict):
                raise RuntimeError(f"Unexpected recovery scope for {run_id}")
            if any(provenance.get(key) != value for key, value in expected_source.items()):
                raise RuntimeError(f"Recovery source provenance drift for {run_id}")
            if any(summary.get(key) != value for key, value in expected_source.items() if key != "source_terminal_status"):
                raise RuntimeError(f"Recovery summary provenance drift for {run_id}")
            source_files = provenance.get("source_files")
            if not isinstance(source_files, list) or len(source_files) != 9:
                raise RuntimeError(f"Expected CSV plus eight lossless recovery PNGs for {run_id}")
            for item in source_files:
                source_path = RECOVERY_SOURCE_ROOT / str(item["path"])
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing preserved recovery source: {source_path}")
                if source_path.stat().st_size != int(item["bytes"]) or sha256(source_path) != item["sha256"]:
                    raise RuntimeError(f"Recovery source hash/size drift: {source_path}")
            recovery_record = {
                **expected_source,
                "source_generation_gpu_requested": summary.get("source_generation_gpu_requested"),
                "source_file_count": len(source_files),
                "source_root": str(RECOVERY_SOURCE_ROOT),
            }
        elif metrics.get("recovery_provenance"):
            raise RuntimeError(f"Metrics claim recovery but summary does not: {run_id}")

        records.append(
            {
                "run_id": run_id,
                "artifact_dir": str(artifact_dir),
                "seed": seed,
                "prompt_slice": [start, end],
                "prompt_count": end - start,
                "gpu_requested": summary["gpu_requested"],
                "gpu_observed": summary["environment"]["gpu"],
                "sequential_scorers": expected_sequential,
                "recovered_partial_shard": recovered,
                "recovery_source": recovery_record,
                "wall_seconds": float(metrics["wall_seconds"]),
                "wall_seconds_scope": metrics.get("wall_seconds_scope", "full_generation_and_scoring"),
                "cuda_max_memory_allocated_mb": float(metrics["cuda_max_memory_allocated_mb"]),
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256(metrics_path),
                "summary_path": str(summary_path),
                "summary_sha256": sha256(summary_path),
            }
        )

    for seed in EXPECTED_SEEDS:
        if coverage[seed] != EXPECTED_PROMPTS:
            raise RuntimeError(f"Incomplete prompt coverage for seed {seed}")
        if sum(record["seed"] == seed for record in records) != EXPECTED_SHARDS[seed]:
            raise RuntimeError(f"Unexpected shard count for seed {seed}")
    seed44_gpus = [record["gpu_requested"] for record in records if record["seed"] == 44]
    if seed44_gpus.count("L4") != 7 or seed44_gpus.count("L40S") != 2:
        raise RuntimeError(f"Expected guarded seed-44 split of seven L4 and two L40S artifacts: {seed44_gpus}")
    recovered_records = [record for record in records if record["recovered_partial_shard"]]
    if len(recovered_records) != 1:
        raise RuntimeError(f"Expected exactly one hash-validated recovered partial shard, got {len(recovered_records)}")

    output = {
        "schema_version": 1,
        "paper": "Null-TTA",
        "upstream_commit": EXPECTED_COMMIT,
        "config": EXPECTED_CONFIG,
        "canonical_artifact_count": len(records),
        "canonical_shard_count": len(records),
        "scientific_modal_app_count": len(records) + 1,
        "seeds": list(EXPECTED_SEEDS),
        "prompt_count_per_seed": 50,
        "total_seed_prompt_records": 150,
        "all_scorers_are_real": True,
        "coverage_complete_and_nonoverlapping": True,
        "runs": sorted(records, key=lambda item: (item["seed"], item["prompt_slice"][0])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "run_count": len(records), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
