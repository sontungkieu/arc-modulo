#!/usr/bin/env python3
"""Create a documented, scaled copy of the official FK Correctors notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} occurrence, found {count}")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-runs", type=int, default=5)
    args = parser.parse_args()

    notebook = json.loads(args.source.read_text())
    cells = notebook["cells"]

    parameters = "".join(cells[25]["source"])
    parameters = replace_once(parameters, "num_int_steps = 1000", f"num_int_steps = {args.num_steps}", "step parameter")
    parameters = replace_once(parameters, "num_samples = 10000", f"num_samples = {args.num_samples}", "sample parameter")
    parameters = replace_once(parameters, "num_seeds = 5", f"num_seeds = {args.num_runs}", "run parameter")
    cells[25]["source"] = parameters.splitlines(keepends=True)

    cells[26]["source"] = [
        "# Runtime-scaled reproduction: skip the standalone Ito density diagnostic.\n",
        "# This cell does not contribute samples or metrics to Table A1.\n",
    ]
    cells[26]["outputs"] = []
    cells[26]["execution_count"] = None

    strategy_old = 'for strategy in ["birth_death_clock", "systematic"]:'
    strategy_new = 'for strategy in ["systematic"]:'
    for index in (27, 28):
        source = "".join(cells[index]["source"])
        cells[index]["source"] = replace_once(
            source, strategy_old, strategy_new, f"strategy list in cell {index}"
        ).splitlines(keepends=True)

    cells[51]["source"] = [
        "# Runtime portability: skip downloading an optional plotting font.\n",
        "# The upstream GitHub URL currently returns HTTP 404.\n",
    ]
    cells[51]["outputs"] = []
    cells[51]["execution_count"] = None
    cells[52]["source"] = [
        "# Runtime-scaled reproduction: skip the optional publication figure.\n",
        "# Table A1 metrics have already been computed and rendered above.\n",
    ]
    cells[52]["outputs"] = []
    cells[52]["execution_count"] = None

    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output": str(args.output),
        "scale": {
            "num_samples": args.num_samples,
            "num_integration_steps": args.num_steps,
            "num_runs": args.num_runs,
        },
        "method_subset": [
            "target_score_no_fkc",
            "tempered_noise_no_fkc",
            "target_score_systematic_fkc",
            "tempered_noise_systematic_fkc",
        ],
        "omissions": {
            "standalone_ito_density_diagnostic": "Not used by Table A1 metrics",
            "birth_death_clock": "Omitted to prioritize the paper's main systematic FKC result",
            "publication_figure": "Skipped because its optional GitHub font URL returns HTTP 404; Table A1 metrics are unaffected",
        },
    }
    notebook.setdefault("metadata", {})["diffusion_smc_runtime_patch"] = provenance

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n")
    provenance["output_sha256"] = sha256(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
