#!/usr/bin/env python3
"""Validate and compare an exact FK Correctors Table A1 run with the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM_COMMIT = "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6"
METHOD_ORDER = [
    "target_score_no_fkc",
    "tempered_noise_no_fkc",
    "target_score_birth_death_clock_fkc",
    "tempered_noise_birth_death_clock_fkc",
    "target_score_systematic_fkc",
    "tempered_noise_systematic_fkc",
]
METRIC_ORDER = ["energy_w2", "mmd", "total_variation", "w1", "w2"]
PAPER = {
    "target_score_no_fkc": {
        "energy_w2": (0.943, 0.026),
        "mmd": (0.020, 0.001),
        "total_variation": (0.487, 0.007),
        "w1": (11.304, 0.296),
        "w2": (15.671, 0.269),
    },
    "tempered_noise_no_fkc": {
        "energy_w2": (1.032, 0.012),
        "mmd": (0.058, 0.001),
        "total_variation": (0.638, 0.002),
        "w1": (16.051, 0.123),
        "w2": (19.627, 0.101),
    },
    "target_score_birth_death_clock_fkc": {
        "energy_w2": (1.064, 0.369),
        "mmd": (0.010, 0.004),
        "total_variation": (0.402, 0.029),
        "w1": (7.797, 3.990),
        "w2": (12.451, 5.417),
    },
    "tempered_noise_birth_death_clock_fkc": {
        "energy_w2": (1.228, 0.401),
        "mmd": (0.056, 0.029),
        "total_variation": (0.572, 0.055),
        "w1": (12.598, 4.155),
        "w2": (17.679, 4.178),
    },
    "target_score_systematic_fkc": {
        "energy_w2": (1.098, 0.418),
        "mmd": (0.007, 0.005),
        "total_variation": (0.372, 0.020),
        "w1": (6.256, 3.960),
        "w2": (11.265, 5.629),
    },
    "tempered_noise_systematic_fkc": {
        "energy_w2": (0.926, 0.248),
        "mmd": (0.027, 0.011),
        "total_variation": (0.512, 0.017),
        "w1": (9.974, 1.229),
        "w2": (14.045, 1.308),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def symmetric_relative(left: float, right: float) -> float:
    return abs(left - right) / max((abs(left) + abs(right)) / 2, 1e-12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--actual-cost", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_path = args.run_dir / "fkc_table_a1_metrics.json"
    manifest_path = args.run_dir / "run_manifest.json"
    summary_path = args.run_dir / "modal_summary.json"
    runtime_patch_path = args.run_dir / "runtime_patch.json"
    metrics = json.loads(metric_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    runtime_patch = json.loads(runtime_patch_path.read_text())

    if manifest.get("status") != "COMPLETED" or summary.get("status") != "COMPLETED":
        raise RuntimeError("Exact Table A1 run is not complete")
    if manifest.get("upstream_commit") != UPSTREAM_COMMIT:
        raise RuntimeError("Upstream commit drift")
    if metrics.get("mode") != "full" or set(metrics.get("methods", {})) != set(METHOD_ORDER):
        raise RuntimeError("Expected all six exact Table A1 methods")
    protocol = metrics.get("protocol", {})
    expected_protocol = {
        "num_samples": 10000,
        "num_integration_steps": 1000,
        "dt": 0.001,
        "num_runs": 5,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise RuntimeError(f"Protocol drift for {key}: {protocol.get(key)} != {expected}")
    if len(metrics.get("raw_rows", [])) != 30:
        raise RuntimeError("Expected exactly 30 method-run rows")

    cells: list[dict[str, object]] = []
    method_comparison: dict[str, object] = {}
    for method in METHOD_ORDER:
        reproduction = metrics["methods"][method]
        if reproduction.get("n_runs") != 5:
            raise RuntimeError(f"Incomplete method: {method}")
        compared_metrics: dict[str, object] = {}
        for metric in METRIC_ORDER:
            paper_mean, paper_sd = PAPER[method][metric]
            repro = reproduction["metrics"][metric]
            repro_mean = float(repro["mean"])
            repro_sd = float(repro["std_sample"])
            z_distance = abs(repro_mean - paper_mean) / paper_sd
            cell = {
                "method": method,
                "metric": metric,
                "paper_mean": paper_mean,
                "paper_sd": paper_sd,
                "reproduction_mean": repro_mean,
                "reproduction_sd_sample": repro_sd,
                "reproduction_values": repro["values"],
                "absolute_difference": abs(repro_mean - paper_mean),
                "symmetric_relative_difference": symmetric_relative(repro_mean, paper_mean),
                "distance_in_paper_sd": z_distance,
                "within_paper_1sd_band": z_distance <= 1.0,
                "within_paper_2sd_band": z_distance <= 2.0,
                "reproduction_to_paper_sd_ratio": repro_sd / paper_sd,
            }
            cells.append(cell)
            compared_metrics[metric] = cell

        raw_rows = [row for row in metrics["raw_rows"] if row["method"] == method]
        run_scores = []
        for row in raw_rows:
            per_metric_distance = {
                metric: abs(float(row[metric]) - PAPER[method][metric][0]) / PAPER[method][metric][1]
                for metric in METRIC_ORDER
            }
            run_scores.append(
                {
                    "run_index": row["run_index_within_method"],
                    "mean_absolute_distance_in_paper_sd": sum(per_metric_distance.values())
                    / len(per_metric_distance),
                    "per_metric_distance_in_paper_sd": per_metric_distance,
                    "metrics": {metric: row[metric] for metric in METRIC_ORDER},
                }
            )
        run_scores.sort(key=lambda row: row["mean_absolute_distance_in_paper_sd"], reverse=True)
        method_comparison[method] = {
            "metrics": compared_metrics,
            "worst_run_by_mean_paper_sd_distance": run_scores[0],
            "runs_ranked_by_mean_paper_sd_distance": run_scores,
        }

    one_sd_count = sum(bool(cell["within_paper_1sd_band"]) for cell in cells)
    two_sd_count = sum(bool(cell["within_paper_2sd_band"]) for cell in cells)
    max_cell = max(cells, key=lambda cell: float(cell["distance_in_paper_sd"]))

    direction_checks = []
    for proposal in ("target_score", "tempered_noise"):
        baseline = f"{proposal}_no_fkc"
        for corrector in ("birth_death_clock", "systematic"):
            corrected = f"{proposal}_{corrector}_fkc"
            for metric in METRIC_ORDER:
                paper_delta = PAPER[corrected][metric][0] - PAPER[baseline][metric][0]
                repro_delta = (
                    metrics["methods"][corrected]["metrics"][metric]["mean"]
                    - metrics["methods"][baseline]["metrics"][metric]["mean"]
                )
                direction_checks.append(
                    {
                        "proposal": proposal,
                        "corrector": corrector,
                        "metric": metric,
                        "paper_delta_corrected_minus_no_fkc": paper_delta,
                        "reproduction_delta_corrected_minus_no_fkc": repro_delta,
                        "direction_agrees": math.copysign(1.0, paper_delta)
                        == math.copysign(1.0, repro_delta),
                    }
                )
    direction_count = sum(bool(row["direction_agrees"]) for row in direction_checks)

    compatibility_fraction = two_sd_count / len(cells)
    numeric_pass = two_sd_count == len(cells)
    primary_verdict = "PASS_WITH_VARIANCE" if numeric_pass else "FAIL_NUMERIC_MATCH"
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts",
        "table": "Table A1",
        "primary_verdict": primary_verdict,
        "scope": "exact six-method GMM Table A1 protocol",
        "protocol": protocol,
        "identity": {
            "upstream_commit": UPSTREAM_COMMIT,
            "source_notebook_sha256": runtime_patch["source_sha256"],
            "runtime_notebook_sha256": runtime_patch["output_sha256"],
            "executed_notebook_sha256": manifest["executed_sha256"],
            "runtime_repair": runtime_patch["runtime_repair"],
            "upstream_clone_modified": False,
        },
        "remote_run": {
            "run_id": summary["run_id"],
            "app_id": args.app_id,
            "modal_profile": args.profile,
            "gpu": summary["environment"]["gpu"],
            "torch": summary["environment"]["torch"],
            "torch_cuda": summary["environment"]["torch_cuda"],
            "elapsed_s": manifest["elapsed_s"],
            "actual_provider_cost_usd": args.actual_cost,
            "cuda_max_memory_allocated_bytes": metrics["resources"][
                "cuda_max_memory_allocated_bytes"
            ],
            "cuda_max_memory_reserved_bytes": metrics["resources"][
                "cuda_max_memory_reserved_bytes"
            ],
        },
        "numeric_match_rule": {
            "cell_compatible": "absolute mean difference <= 2 * paper-reported run SD",
            "interpretation": (
                "Compatibility band, not a confidence interval; both paper and reproduction use five runs"
            ),
            "aggregate_pass": "all 30 cells compatible and exact protocol/coverage gates pass",
        },
        "numeric_match": {
            "cell_count": len(cells),
            "within_paper_1sd_band_count": one_sd_count,
            "within_paper_2sd_band_count": two_sd_count,
            "within_paper_2sd_band_fraction": compatibility_fraction,
            "max_distance_cell": max_cell,
            "corrector_direction_agreement_count": direction_count,
            "corrector_direction_check_count": len(direction_checks),
            "pass": numeric_pass,
        },
        "methods": method_comparison,
        "cells": cells,
        "direction_checks": direction_checks,
        "raw_rows": metrics["raw_rows"],
        "particle_diversity": metrics["particle_diversity"],
        "run_semantics_caveat": protocol["upstream_run_semantics"],
        "source_artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for name, path in {
                "metrics": metric_path,
                "run_manifest": manifest_path,
                "modal_summary": summary_path,
                "runtime_patch": runtime_patch_path,
                "executed_notebook": args.run_dir / "gmm_table_a1_full.executed.ipynb",
            }.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": primary_verdict,
                "within_1sd": f"{one_sd_count}/30",
                "within_2sd": f"{two_sd_count}/30",
                "direction_agreement": f"{direction_count}/{len(direction_checks)}",
                "max_distance_cell": max_cell,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
