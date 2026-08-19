#!/usr/bin/env python3
"""Create the scientific decision record for the SDXL T4x2 canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parity = json.loads(args.parity.read_text(encoding="utf-8"))
    full = json.loads(args.full.read_text(encoding="utf-8"))
    parity_pass = bool(
        parity.get("numerical_comparison_available")
        and parity.get("numerical_match")
    )
    full_split_fits = bool(full.get("split_fits"))
    full_single_fits = bool(full.get("single_fits"))
    if parity_pass and full_split_fits and full_single_fits:
        decision = "microbatch-single-and-split-feasible"
    elif parity_pass and full_split_fits:
        decision = "split-components-feasible"
    else:
        decision = "do-not-enable"
    payload = {
        "schema_version": 1,
        "benchmark": "fk_steering_sdxl_t4x2_component_canary",
        "parity_pass": parity_pass,
        "full_split_fits": full_split_fits,
        "full_single_fits": full_single_fits,
        "full_single_vs_split_verdict": full.get("verdict"),
        "passed": parity_pass and full_split_fits,
        "decision": decision,
        "evidence_boundary": (
            "One prompt canary only; passing does not reproduce GenEval or the "
            "paper's aggregate metrics."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("FK_COMPONENT_CANARY_SUMMARY " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
