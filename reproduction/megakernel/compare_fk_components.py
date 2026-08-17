#!/usr/bin/env python3
"""Compare single- and split-device FK component-placement canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def pixel_difference(
    reference: np.ndarray, candidate: np.ndarray, *, atol: float, rtol: float
) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "allclose": False,
            "exact": False,
            "violation_fraction": 1.0,
        }
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    absolute = np.abs(candidate64 - reference64)
    tolerance = atol + rtol * np.abs(reference64)
    violations = absolute > tolerance
    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "allclose": not bool(np.any(violations)),
        "exact": bool(np.array_equal(reference, candidate)),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(absolute**2))) if absolute.size else 0.0,
        "violation_fraction": float(violations.mean()) if violations.size else 0.0,
    }


def trace_difference(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        return {
            "length_match": False,
            "reference_length": len(reference),
            "candidate_length": len(candidate),
            "allclose": False,
        }
    if not reference:
        return {
            "length_match": True,
            "reference_length": 0,
            "candidate_length": 0,
            "allclose": False,
            "reason": "FK trace is missing",
        }

    discrete_match = True
    max_abs = 0.0
    mean_abs_values: list[float] = []
    first_failed_checkpoint = None
    for position, (reference_row, candidate_row) in enumerate(
        zip(reference, candidate)
    ):
        row_discrete_match = all(
            reference_row.get(key) == candidate_row.get(key)
            for key in ("sampling_idx", "resampled", "indices")
        )
        row_numeric_match = True
        for key in ("raw_rewards", "candidate_rewards", "weights"):
            difference = pixel_difference(
                np.asarray(reference_row.get(key, []), dtype=np.float64),
                np.asarray(candidate_row.get(key, []), dtype=np.float64),
                atol=atol,
                rtol=rtol,
            )
            row_numeric_match = row_numeric_match and difference["allclose"]
            max_abs = max(max_abs, difference.get("max_abs", 0.0))
            mean_abs_values.append(difference.get("mean_abs", 0.0))

        reference_ess = reference_row.get("ess")
        candidate_ess = candidate_row.get("ess")
        if reference_ess is None or candidate_ess is None:
            ess_match = reference_ess is candidate_ess
        else:
            ess_match = bool(
                np.isclose(reference_ess, candidate_ess, atol=atol, rtol=rtol)
            )
            max_abs = max(max_abs, abs(candidate_ess - reference_ess))
        row_match = row_discrete_match and row_numeric_match and ess_match
        if not row_match and first_failed_checkpoint is None:
            first_failed_checkpoint = position
        discrete_match = discrete_match and row_discrete_match

    return {
        "length_match": True,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "discrete_match": discrete_match,
        "allclose": first_failed_checkpoint is None,
        "first_failed_checkpoint": first_failed_checkpoint,
        "max_abs": max_abs,
        "mean_abs": (
            float(np.mean(mean_abs_values)) if mean_abs_values else 0.0
        ),
        "atol": atol,
        "rtol": rtol,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--single-array", type=Path, required=True)
    parser.add_argument("--split-array", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    single = json.loads(args.single_json.read_text(encoding="utf-8"))
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "single_status": single.get("status", "missing"),
        "split_status": split.get("status", "missing"),
        "single_json": str(args.single_json),
        "split_json": str(args.split_json),
        "split_fits": split.get("status") == "ok",
        "single_fits": single.get("status") == "ok",
        "numerical_comparison_available": False,
        "numerical_match": None,
    }

    if single.get("status") == "ok" and split.get("status") == "ok":
        reference = np.load(args.single_array, allow_pickle=False)
        candidate = np.load(args.split_array, allow_pickle=False)
        pixels = pixel_difference(reference, candidate, atol=args.atol, rtol=args.rtol)
        trace = trace_difference(
            single.get("fkd_trace", []),
            split.get("fkd_trace", []),
            atol=args.atol,
            rtol=args.rtol,
        )
        payload["pixel_difference"] = pixels
        payload["trace_difference"] = trace
        payload["numerical_comparison_available"] = True
        payload["numerical_match"] = pixels["allclose"] and trace["allclose"]
        payload["hash_match"] = single.get("output_sha256") == split.get(
            "output_sha256"
        )

    if payload["split_fits"] and payload["single_fits"]:
        payload["verdict"] = (
            "split_matches_single"
            if payload["numerical_match"]
            else "split_output_mismatch"
        )
    elif payload["split_fits"] and single.get("status") == "oom":
        payload["verdict"] = "split_fits_single_oom"
    elif split.get("status") == "oom":
        payload["verdict"] = "split_oom"
    else:
        payload["verdict"] = "inconclusive"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("FK_COMPONENT_COMPARISON " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
