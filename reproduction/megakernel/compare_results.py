#!/usr/bin/env python3
"""Compare eager/optimized diffusion benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = str(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [_load(path) for path in args.input]
    real = [
        item
        for item in payloads
        if item.get("benchmark") == "fk_steering_real_pipeline_denoising"
    ]
    eager = next(
        (
            item
            for item in real
            if item.get("mode") == "eager"
            and item.get("device_map_requested") == "single"
        ),
        None,
    )
    comparisons: list[dict[str, Any]] = []
    for item in real:
        comparison: dict[str, Any] = {
            "path": item["_path"],
            "mode": item.get("mode"),
            "device_map": item.get("device_map_requested"),
            "status": item.get("status"),
            "median_s": item.get("median_s"),
            "particle_steps_per_s": item.get("particle_steps_per_s"),
            "peak_allocated_bytes_by_device": {
                key: value.get("peak_allocated_bytes")
                for key, value in (item.get("device_memory") or {}).items()
            },
        }
        if eager and item.get("median_s") and eager.get("median_s"):
            comparison["speedup_vs_single_eager"] = eager["median_s"] / item["median_s"]
            baseline_probe = np.asarray(
                eager["output_probes_first_256"][0], dtype=np.float32
            )
            candidate_probe = np.asarray(
                item["output_probes_first_256"][0], dtype=np.float32
            )
            delta = np.abs(candidate_probe - baseline_probe)
            comparison["probe_max_abs_error_vs_eager"] = float(delta.max())
            comparison["probe_mean_abs_error_vs_eager"] = float(delta.mean())
            comparison["exact_hash_match_vs_eager"] = (
                item["deterministic_output_hashes"][0]
                == eager["deterministic_output_hashes"][0]
            )
        comparisons.append(comparison)

    accepted = [
        row
        for row in comparisons
        if row.get("mode") != "eager"
        and row.get("status") == "ok"
        and (row.get("speedup_vs_single_eager") or 0.0) >= 1.05
        and (row.get("probe_max_abs_error_vs_eager") or 0.0) <= 5e-3
    ]
    result = {
        "schema_version": 1,
        "inputs": [str(path) for path in args.input],
        "comparison": comparisons,
        "acceptance_gate": {
            "minimum_speedup": 1.05,
            "maximum_probe_abs_error": 5e-3,
            "requires_no_fallback": True,
        },
        "accepted_modes": [row["mode"] for row in accepted],
        "recommendation": (
            "adopt fastest accepted mode for this exact hardware/model shape"
            if accepted
            else "keep eager; optimization did not clear both speed and correctness gates"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("MEGAKERNEL_COMPARISON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
