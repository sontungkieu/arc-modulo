#!/usr/bin/env python3
"""Prepare an auditable FK Correctors Table A1 runtime notebook.

The upstream notebook remains untouched.  This transformer can create either a
small BDC canary or the exact six-method, 10k-sample Table A1 workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


UPSTREAM_COMMIT = "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6"


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


def code_cell(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--num-runs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = {
        "canary": {"num_samples": 1000, "num_steps": 100, "num_runs": 2},
        "full": {"num_samples": 10000, "num_steps": 1000, "num_runs": 5},
    }[args.mode]
    num_samples = args.num_samples or defaults["num_samples"]
    num_steps = args.num_steps or defaults["num_steps"]
    num_runs = args.num_runs or defaults["num_runs"]
    if min(num_samples, num_steps, num_runs) <= 0:
        raise ValueError("Sample, step, and run counts must be positive")
    if args.mode == "full" and (num_samples, num_steps, num_runs) != (10000, 1000, 5):
        raise ValueError("Full mode is locked to the exact Table A1 protocol: 10000/1000/5")

    notebook = json.loads(args.source.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    parameters = "".join(cells[25]["source"])
    parameters = replace_once(
        parameters, "num_int_steps = 1000", f"num_int_steps = {num_steps}", "step parameter"
    )
    parameters = replace_once(
        parameters, "num_samples = 10000", f"num_samples = {num_samples}", "sample parameter"
    )
    parameters = replace_once(
        parameters, "num_seeds = 5", f"num_seeds = {num_runs}", "run parameter"
    )
    cells[25]["source"] = parameters.splitlines(keepends=True)

    cells[26] = code_cell(
        "# Runtime portability: skip the standalone Ito density diagnostic.\n"
        "# It does not feed any sample or metric reported in Table A1.\n"
    )

    integrator = "".join(cells[21]["source"])
    old_call = """sample_birth_death_clocks(
            x.shape[0], accum_birth, accum_death, clock_thresholds
        )"""
    new_call = """sample_birth_death_clocks(
            x.shape[0],
            accum_birth,
            accum_death,
            clock_thresholds,
            reset_transition_per_index=reset_transition_per_index,
        )"""
    cells[21]["source"] = replace_once(
        integrator, old_call, new_call, "BDC reset-transition forwarding repair"
    ).splitlines(keepends=True)

    strategy_old = 'for strategy in ["birth_death_clock", "systematic"]:'
    strategy_new = (
        strategy_old
        if args.mode == "full"
        else 'for strategy in ["birth_death_clock"]:'
    )
    for index in (27, 28):
        source = "".join(cells[index]["source"])
        if strategy_new != strategy_old:
            source = replace_once(
                source, strategy_old, strategy_new, f"canary strategy list in cell {index}"
            )
        cells[index]["source"] = source.splitlines(keepends=True)

    if args.mode == "canary":
        for index in (30, 31):
            cells[index] = code_cell(
                "# BDC canary: skip no-FKC baselines; full mode executes this upstream cell.\n"
            )
        for index in (32, 33):
            cells[index] = code_cell(
                "# BDC canary: skip the auxiliary log-weight plot for omitted no-FKC baselines.\n"
                "# This plotting-only cell does not feed Table A1 metrics.\n"
            )

    cells[51] = code_cell(
        "# Runtime portability: skip the optional plotting-font download.\n"
        "# It is downstream of all Table A1 metrics and the upstream URL returns HTTP 404.\n"
    )
    cells[52] = code_cell(
        "# Runtime portability: skip the optional publication-only sample figure.\n"
        "# Table A1 metrics have already been computed above.\n"
    )

    result_cell = r'''# Machine-readable Table A1 artifact and resource diagnostics.
import json as _json
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path as _Path

_method_names = {
    "not_resampled": "target_score_no_fkc",
    "not_resampled scale_diffusion": "tempered_noise_no_fkc",
    "resampled birth_death_clock": "target_score_birth_death_clock_fkc",
    "resampled birth_death_clock scale_diffusion": "tempered_noise_birth_death_clock_fkc",
    "resampled systematic": "target_score_systematic_fkc",
    "resampled systematic scale_diffusion": "tempered_noise_systematic_fkc",
}
_metric_names = {
    "W1": "w1",
    "W2": "w2",
    "MMD": "mmd",
    "Total Var": "total_variation",
    "Energy W2": "energy_w2",
}
_raw_rows = []
for _run_index, _row in df.reset_index(drop=True).iterrows():
    _method = str(_row["Method"])
    _raw_rows.append({
        "method": _method_names[_method],
        "upstream_method_key": _method,
        "run_index_within_method": int((_run_index % num_seeds)),
        **{_metric_names[_metric]: float(_row[_metric]) for _metric in metrics},
    })

_methods = {}
for _upstream_method, _group in df.groupby("Method", sort=False):
    _method = _method_names[str(_upstream_method)]
    _methods[_method] = {
        "upstream_method_key": str(_upstream_method),
        "n_runs": int(len(_group)),
        "metrics": {
            _metric_names[_metric]: {
                "mean": float(_group[_metric].mean()),
                "std_sample": float(_group[_metric].std(ddof=1)),
                "values": [float(_value) for _value in _group[_metric].tolist()],
            }
            for _metric in metrics
        },
    }

_choice_diagnostics = {}
for _key, _value in choices.items():
    _choice_tensor = torch.as_tensor(_value, dtype=torch.float32)
    _choice_diagnostics[_method_names[_key]] = {
        "min_unique_particles": float(_choice_tensor.min().item()),
        "mean_unique_particles": float(_choice_tensor.mean().item()),
        "final_unique_particles_per_run": [float(x) for x in _choice_tensor[:, -1].tolist()],
    }

_payload = {
    "schema_version": 1,
    "created_at_utc": _datetime.now(_timezone.utc).isoformat(),
    "paper": "Feynman-Kac Correctors in Diffusion: Annealing, Guidance, and Product of Experts",
    "table": "Table A1",
    "upstream_commit": "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6",
    "mode": "__MODE__",
    "protocol": {
        "num_samples": int(num_samples),
        "num_integration_steps": int(num_int_steps),
        "dt": float(1 / num_int_steps),
        "num_runs": int(num_seeds),
        "upstream_run_semantics": "sequential stochastic runs; upstream loop variable is named seed but does not reseed",
    },
    "runtime_repair": {
        "name": "forward_reset_transition_per_index_to_birth_death_sampler",
        "scope": "one runtime-notebook call site; upstream clone unchanged",
    },
    "methods": _methods,
    "raw_rows": _raw_rows,
    "particle_diversity": _choice_diagnostics,
    "resources": {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
    },
}
_Path("fkc_table_a1_metrics.json").write_text(_json.dumps(_payload, indent=2) + "\n")
print("FKC_TABLE_A1_METRICS_JSON=" + _json.dumps(_payload, sort_keys=True))
'''.replace("__MODE__", args.mode)
    cells.append(code_cell(result_cell))

    expected_methods = (
        [
            "target_score_no_fkc",
            "tempered_noise_no_fkc",
            "target_score_birth_death_clock_fkc",
            "tempered_noise_birth_death_clock_fkc",
            "target_score_systematic_fkc",
            "tempered_noise_systematic_fkc",
        ]
        if args.mode == "full"
        else [
            "target_score_birth_death_clock_fkc",
            "tempered_noise_birth_death_clock_fkc",
        ]
    )
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "source_upstream_commit": UPSTREAM_COMMIT,
        "output": str(args.output),
        "mode": args.mode,
        "protocol": {
            "num_samples": num_samples,
            "num_integration_steps": num_steps,
            "dt": 1 / num_steps,
            "num_runs": num_runs,
        },
        "expected_methods": expected_methods,
        "runtime_repair": {
            "cell_index": 21,
            "reason": (
                "The caller creates one-dimensional accum_birth for "
                "reset_transition_per_index=False, but the upstream call omitted that flag and "
                "therefore selected the sampler's incompatible True default."
            ),
            "old": old_call,
            "new": new_call,
        },
        "non_metric_omissions": {
            "standalone_ito_density_diagnostic": "Does not feed Table A1 samples or metrics",
            "canary_only_log_weight_plots": (
                "Canary omits plotting cells 32-33 because their no-FKC variables are absent; "
                "full mode executes them unchanged"
            ),
            "plotting_font_and_publication_figure": "Downstream of all Table A1 metrics",
        },
    }
    notebook.setdefault("metadata", {})["diffusion_smc_runtime_patch"] = provenance

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    provenance["output_sha256"] = sha256(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, sort_keys=True))


if __name__ == "__main__":
    main()
