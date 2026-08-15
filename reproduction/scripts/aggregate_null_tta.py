#!/usr/bin/env python3
"""Aggregate complete Null-TTA Table 1 shards without seed cherry-picking."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean, stdev
from typing import Any


EXPECTED_SEEDS = (42, 43, 44)
EXPECTED_PROMPTS = tuple(range(50))
METRICS = ("pickscore", "hpsv2", "aesthetic", "imagereward")
PAPER = {
    "baseline": {"pickscore": 0.218, "hpsv2": 0.279, "aesthetic": 5.232, "imagereward": 0.339},
    "null_tta_nmax55": {
        "pickscore": 0.315,
        "hpsv2": 0.294,
        "aesthetic": 5.431,
        "imagereward": 0.946,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    sample_std = stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean(values),
        "std": sample_std,
        "n": len(values),
        "se": sample_std / math.sqrt(len(values)),
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def main() -> int:
    args = parse_args()
    records: dict[tuple[int, int], dict[str, Any]] = {}
    source_files = []
    reference_config = None
    reference_commit = None

    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("all_scorers_are_real"):
            raise RuntimeError(f"All four real scorers are required: {path}")
        if payload.get("prompt_count") != payload["prompt_slice"][1] - payload["prompt_slice"][0]:
            raise RuntimeError(f"Prompt count/slice mismatch: {path}")
        if reference_config is None:
            reference_config = payload["config"]
            reference_commit = payload["upstream_commit"]
        if payload["config"] != reference_config or payload["upstream_commit"] != reference_commit:
            raise RuntimeError(f"Config or upstream drift: {path}")
        source_files.append(str(path))
        seed = int(payload["seed"])
        for row in payload["per_prompt"]:
            key = (seed, int(row["global_prompt_index"]))
            if key in records:
                raise RuntimeError(f"Duplicate seed/prompt record: {key}")
            if any(row["metrics"].get(metric) is None for metric in METRICS):
                raise RuntimeError(f"Mock or missing scorer row at {key}")
            records[key] = row

    expected = {(seed, prompt) for seed in EXPECTED_SEEDS for prompt in EXPECTED_PROMPTS}
    missing = sorted(expected - set(records))
    extra = sorted(set(records) - expected)
    if missing or extra:
        raise RuntimeError(f"Incomplete Table 1 matrix; missing={missing[:10]}, extra={extra[:10]}")

    aggregate: dict[str, Any] = {}
    per_seed: dict[str, Any] = {str(seed): {} for seed in EXPECTED_SEEDS}
    rng = random.Random(args.bootstrap_seed)
    for metric in METRICS:
        for seed in EXPECTED_SEEDS:
            rows = [records[(seed, prompt)]["metrics"][metric] for prompt in EXPECTED_PROMPTS]
            per_seed[str(seed)][metric] = {
                field: stats([float(row[field]) for row in rows])
                for field in ("baseline", "optimized", "improvement")
            }

        optimized_seed_means = [per_seed[str(seed)][metric]["optimized"]["mean"] for seed in EXPECTED_SEEDS]
        baseline_seed_means = [per_seed[str(seed)][metric]["baseline"]["mean"] for seed in EXPECTED_SEEDS]
        improvement_seed_means = [per_seed[str(seed)][metric]["improvement"]["mean"] for seed in EXPECTED_SEEDS]

        bootstrapped = []
        bootstrapped_improvement = []
        for _ in range(args.bootstrap_repeats):
            sampled_seeds = [rng.choice(EXPECTED_SEEDS) for _ in EXPECTED_SEEDS]
            sampled_prompts = [rng.choice(EXPECTED_PROMPTS) for _ in EXPECTED_PROMPTS]
            bootstrapped.append(
                mean(
                    float(records[(seed, prompt)]["metrics"][metric]["optimized"])
                    for seed in sampled_seeds
                    for prompt in sampled_prompts
                )
            )
            bootstrapped_improvement.append(
                mean(
                    float(records[(seed, prompt)]["metrics"][metric]["improvement"])
                    for seed in sampled_seeds
                    for prompt in sampled_prompts
                )
            )

        reproduced = mean(optimized_seed_means)
        paper_target = PAPER["null_tta_nmax55"][metric]
        symmetric_relative = abs(reproduced - paper_target) / ((abs(reproduced) + abs(paper_target)) / 2)
        aggregate[metric] = {
            "baseline_between_seed": stats(baseline_seed_means),
            "optimized_between_seed": stats(optimized_seed_means),
            "improvement_between_seed": stats(improvement_seed_means),
            "optimized_hierarchical_bootstrap_ci95": [
                percentile(bootstrapped, 0.025),
                percentile(bootstrapped, 0.975),
            ],
            "improvement_hierarchical_bootstrap_ci95": [
                percentile(bootstrapped_improvement, 0.025),
                percentile(bootstrapped_improvement, 0.975),
            ],
            "paper_baseline": PAPER["baseline"][metric],
            "paper_null_tta_nmax55": paper_target,
            "symmetric_relative_difference": symmetric_relative,
            "numeric_match_10pct": symmetric_relative <= 0.10,
            "paired_improvement_ci_excludes_zero": percentile(bootstrapped_improvement, 0.025) > 0.0,
            "worst_seed": min(EXPECTED_SEEDS, key=lambda seed: per_seed[str(seed)][metric]["optimized"]["mean"]),
            "best_seed": max(EXPECTED_SEEDS, key=lambda seed: per_seed[str(seed)][metric]["optimized"]["mean"]),
        }

    output = {
        "schema_version": 1,
        "paper": "Null-TTA",
        "experiment": "CVPR 2026 Table 1 SD-v1.5 PickScore-target Null-TTA nmax=55",
        "upstream_commit": reference_commit,
        "config": reference_config,
        "seeds": list(EXPECTED_SEEDS),
        "prompt_count_per_seed": len(EXPECTED_PROMPTS),
        "all_scorers_are_real": True,
        "source_files": source_files,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "primary_verdict": (
            "PASS"
            if aggregate["pickscore"]["numeric_match_10pct"]
            and aggregate["pickscore"]["paired_improvement_ci_excludes_zero"]
            else "FAIL"
        ),
        "primary_criterion": (
            "PickScore symmetric relative difference to the paper Table 1 point is at most 10%, "
            "and the paired baseline-to-optimized hierarchical-bootstrap CI95 is strictly positive"
        ),
        "bootstrap": {
            "method": "two-stage resampling of seeds and prompt indices with replacement",
            "repeats": args.bootstrap_repeats,
            "seed": args.bootstrap_seed,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("NULL_TTA_AGGREGATE_JSON=" + json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
