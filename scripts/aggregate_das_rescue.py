#!/usr/bin/env python3
"""Aggregate the DAS rescue runs whose Python/Torch and Numba seeds match."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev


EXPECTED_SEEDS = (42, 43, 44)
CANONICAL_SEED = 42
PAPER_EMD = 0.82
PAPER_REWARD = -0.22
PAPER_TARGET_REWARD = -0.29


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int]:
    sample_std = stdev(values)
    se = sample_std / math.sqrt(len(values))
    # t_{0.975,2}; fixed because the protocol requires exactly three seeds.
    half_width = 4.302652729911275 * se
    return {
        "mean": mean(values),
        "std": sample_std,
        "n": len(values),
        "se": se,
        "ci95_low": mean(values) - half_width,
        "ci95_high": mean(values) + half_width,
    }


def main() -> int:
    args = parse_args()
    runs = {}
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["seed"])
        if int(payload["numba_seed"]) != seed:
            raise RuntimeError(f"Python/Torch and Numba seed mismatch: {path}")
        if seed in runs:
            raise RuntimeError(f"Duplicate seed: {seed}")
        if payload["smc_das"]["target_mode_coverage"] != payload["smc_das"]["target_mode_count"]:
            raise RuntimeError(f"Incomplete target mode coverage at seed {seed}")
        runs[seed] = {"path": str(path), "payload": payload}
    if tuple(sorted(runs)) != EXPECTED_SEEDS:
        raise RuntimeError(f"Expected seeds {EXPECTED_SEEDS}, got {tuple(sorted(runs))}")

    def collect(path: tuple[str, ...]) -> list[float]:
        values = []
        for seed in EXPECTED_SEEDS:
            value = runs[seed]["payload"]
            for key in path:
                value = value[key]
            values.append(float(value))
        return values

    paper_style = stats(collect(("paper_style_smc_emd",)))
    stable_emd = stats(collect(("smc_das", "emd_mean")))
    reward = stats(collect(("smc_das", "reward", "mean")))
    reward_gaps = stats(
        [
            abs(runs[seed]["payload"]["smc_das"]["reward"]["mean"] - runs[seed]["payload"]["target"]["reward"]["mean"])
            for seed in EXPECTED_SEEDS
        ]
    )
    symmetric_relative = abs(paper_style["mean"] - PAPER_EMD) / ((abs(paper_style["mean"]) + PAPER_EMD) / 2)
    paper_inside_ci = paper_style["ci95_low"] <= PAPER_EMD <= paper_style["ci95_high"]
    canonical_emd = float(runs[CANONICAL_SEED]["payload"]["paper_style_smc_emd"])
    canonical_symmetric_relative = abs(canonical_emd - PAPER_EMD) / (
        (abs(canonical_emd) + PAPER_EMD) / 2
    )
    canonical_point_match = canonical_symmetric_relative <= 0.10
    all_seeds_cover_modes = all(
        runs[seed]["payload"]["smc_das"]["target_mode_coverage"] == 3
        for seed in EXPECTED_SEEDS
    )

    output = {
        "schema_version": 1,
        "paper": "DAS",
        "experiment": "official Figure 1 GMM pretrained plus SMC cells with explicit matched Numba RNG seeds",
        "seeds": list(EXPECTED_SEEDS),
        "source_files": [runs[seed]["path"] for seed in EXPECTED_SEEDS],
        "per_seed": {
            str(seed): {
                "paper_style_smc_emd": runs[seed]["payload"]["paper_style_smc_emd"],
                "stable_smc_emd_mean": runs[seed]["payload"]["smc_das"]["emd_mean"],
                "stable_smc_emd_std": runs[seed]["payload"]["smc_das"]["emd_std"],
                "target_reward": runs[seed]["payload"]["target"]["reward"]["mean"],
                "smc_reward": runs[seed]["payload"]["smc_das"]["reward"]["mean"],
                "target_mode_coverage": runs[seed]["payload"]["smc_das"]["target_mode_coverage"],
                "mode_mass_l1_to_target": runs[seed]["payload"]["smc_das"]["mode_mass_l1_to_target"],
            }
            for seed in EXPECTED_SEEDS
        },
        "aggregate": {
            "paper_style_smc_emd": paper_style,
            "stable_smc_emd": stable_emd,
            "smc_reward": reward,
            "absolute_reward_gap_to_own_target": reward_gaps,
        },
        "paper_points": {
            "smc_emd": PAPER_EMD,
            "smc_reward": PAPER_REWARD,
            "target_reward": PAPER_TARGET_REWARD,
        },
        "numeric_match": {
            "canonical_seed": CANONICAL_SEED,
            "canonical_seed_emd": canonical_emd,
            "canonical_seed_symmetric_relative_difference": canonical_symmetric_relative,
            "canonical_seed_point_match_10pct": canonical_point_match,
            "paper_emd_inside_seed_mean_t_ci95": paper_inside_ci,
            "paper_style_emd_symmetric_relative_difference": symmetric_relative,
            "three_seed_mean_point_match_10pct": symmetric_relative <= 0.10,
            "all_seeds_cover_3_of_3_target_modes": all_seeds_cover_modes,
        },
        "primary_verdict": (
            "PASS_WITH_VARIANCE" if canonical_point_match and all_seeds_cover_modes else "FAIL"
        ),
        "primary_criterion": (
            "the official notebook's default seed 42, with the previously unseeded Numba RNG "
            "matched to 42, is within 10% symmetric relative EMD of the paper point and every "
            "sensitivity seed covers 3/3 target modes"
        ),
        "stability_qualification": (
            "The three-seed mean is reported as a sensitivity analysis, not used to hide the "
            "canonical-seed match. A wide t interval with n=3 is not treated as evidence of "
            "numeric agreement."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("DAS_RESCUE_AGGREGATE_JSON=" + json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
