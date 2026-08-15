#!/usr/bin/env python3
"""Create a seeded, instrumented runtime copy of the official DAS GMM notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat


SEED_LINES = (
    "torch.manual_seed(42)",
    "torch.cuda.manual_seed(42)",
    "torch.cuda.manual_seed_all(42)",
    "np.random.seed(42)",
    "random.seed(42)",
)


DETERMINISM_CELL = r'''# Runtime-only deterministic rescue controls (not part of upstream DAS)
import json as _repro_json
import os as _repro_os
import time as _repro_time
from numba import njit as _repro_njit

_repro_started_at = _repro_time.perf_counter()
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

@_repro_njit
def _repro_seed_numba(seed):
    # Numba maintains RNG state independently from Python/NumPy.
    np.random.seed(seed)

_repro_seed_numba(__REPRO_NUMBA_SEED__)
print("DAS_DETERMINISM_JSON=" + _repro_json.dumps({
    "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "cublas_workspace_config": _repro_os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    "numba_seed": __REPRO_NUMBA_SEED__,
}, sort_keys=True))
'''


METRICS_CELL = r'''# Runtime-only reproduction instrumentation (not part of upstream DAS)
import json as _repro_json
from pathlib import Path as _ReproPath
from scipy.stats import wasserstein_distance_nd as _repro_wd

_repro_seed = __REPRO_SEED__
_repro_methods = {
    "pretrained": np.asarray(pre_trained_samples),
    "rl_ddpo": np.asarray(rl_samples),
    "direct_backprop": np.asarray(backprop_samples),
    "approx_guidance": np.asarray(guidance_samples),
    "smc_das": np.asarray(smc_samples),
}
_repro_target = np.asarray(target_samples)
_repro_means = np.asarray(means, dtype=float)
_repro_rng = np.random.default_rng(_repro_seed + 10000)

def _repro_reward_stats(samples):
    values = np.asarray(compute_reward(np.asarray(samples)), dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "n": int(values.size),
    }

def _repro_mode_mass(samples):
    samples = np.asarray(samples)
    nearest = np.square(samples[:, None, :] - _repro_means[None, :, :]).sum(axis=2).argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(_repro_means))
    return counts.astype(float) / counts.sum()

def _repro_emd_repeats(samples, repeats=5, n=500):
    values = []
    samples = np.asarray(samples)
    for _ in range(repeats):
        sample_idx = _repro_rng.integers(0, len(samples), size=n)
        target_idx = _repro_rng.integers(0, len(_repro_target), size=n)
        values.append(float(_repro_wd(samples[sample_idx], _repro_target[target_idx])))
    return values

_repro_target_reward = _repro_reward_stats(_repro_target)
_repro_target_mass = _repro_mode_mass(_repro_target)
_repro_active = _repro_target_mass >= 0.01
_repro_metrics = {
    "schema_version": 1,
    "paper": "DAS",
    "experiment": "official GMM notebook Figure 1",
    "seed": _repro_seed,
    "upstream_seed_replaced": 42,
    "emd_estimator": {
        "implementation": "scipy.stats.wasserstein_distance_nd",
        "subsample_n": 500,
        "replacement": True,
        "repeats": 5,
        "paper_notebook_repeats": 1,
    },
    "mode_assignment": {
        "centers": _repro_means.tolist(),
        "active_target_threshold": 0.01,
        "target_active_modes": np.flatnonzero(_repro_active).tolist(),
    },
    "target": {
        "reward": _repro_target_reward,
        "mode_mass": _repro_target_mass.tolist(),
    },
    "methods": {},
}

for _repro_name, _repro_samples in _repro_methods.items():
    _repro_reward = _repro_reward_stats(_repro_samples)
    _repro_mass = _repro_mode_mass(_repro_samples)
    _repro_emd = _repro_emd_repeats(_repro_samples)
    _repro_metrics["methods"][_repro_name] = {
        "reward": _repro_reward,
        "reward_gap_abs_to_target": abs(_repro_reward["mean"] - _repro_target_reward["mean"]),
        "emd_repeats": _repro_emd,
        "emd_mean": float(np.mean(_repro_emd)),
        "emd_std": float(np.std(_repro_emd, ddof=1)),
        "mode_mass": _repro_mass.tolist(),
        "mode_mass_l1_to_target": float(np.abs(_repro_mass - _repro_target_mass).sum()),
        "target_mode_coverage": int(np.logical_and(_repro_mass >= 0.01, _repro_active).sum()),
        "target_mode_count": int(_repro_active.sum()),
        "spurious_mass_outside_target_modes": float(_repro_mass[~_repro_active].sum()),
    }

_ReproPath("das_metrics.json").write_text(
    _repro_json.dumps(_repro_metrics, indent=2) + "\n", encoding="utf-8"
)
np.savez_compressed(
    "das_samples.npz",
    target=_repro_target,
    pretrained=_repro_methods["pretrained"],
    rl_ddpo=_repro_methods["rl_ddpo"],
    direct_backprop=_repro_methods["direct_backprop"],
    approx_guidance=_repro_methods["approx_guidance"],
    smc_das=_repro_methods["smc_das"],
)
print("DAS_REPRO_METRICS_JSON=" + _repro_json.dumps(_repro_metrics, sort_keys=True))
'''


SMC_ONLY_METRICS_CELL = r'''# Runtime-only DAS SMC rescue instrumentation
import json as _repro_json
import os as _repro_os
import time as _repro_time
from pathlib import Path as _ReproPath
from scipy.stats import wasserstein_distance_nd as _repro_wd

_repro_seed = __REPRO_SEED__
_repro_numba_seed = __REPRO_NUMBA_SEED__
_repro_smc = np.asarray(smc_samples)
_repro_pretrained = np.asarray(pre_trained_samples)
_repro_target = np.asarray(target_samples)
_repro_means = np.asarray(means, dtype=float)

def _repro_reward_stats(samples):
    values = np.asarray(compute_reward(np.asarray(samples)), dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "n": int(values.size),
    }

def _repro_mode_mass(samples):
    samples = np.asarray(samples)
    nearest = np.square(samples[:, None, :] - _repro_means[None, :, :]).sum(axis=2).argmin(axis=1)
    counts = np.bincount(nearest, minlength=len(_repro_means))
    return counts.astype(float) / counts.sum()

def _repro_emd_repeats(samples, repeats=100, n=500):
    rng = np.random.default_rng(_repro_seed * 1000003 + _repro_numba_seed + 10000)
    values = []
    samples = np.asarray(samples)
    for _ in range(repeats):
        sample_idx = rng.integers(0, len(samples), size=n)
        target_idx = rng.integers(0, len(_repro_target), size=n)
        values.append(float(_repro_wd(samples[sample_idx], _repro_target[target_idx])))
    return values

# Match the paper notebook's single stochastic 500-vs-500 estimator exactly.
_repro_paper_sample_idx = np.random.choice(np.arange(len(_repro_smc)), 500)
_repro_paper_target_idx = np.random.choice(np.arange(len(_repro_target)), 500)
_repro_paper_emd = float(_repro_wd(
    _repro_smc[_repro_paper_sample_idx],
    _repro_target[_repro_paper_target_idx],
))

_repro_target_reward = _repro_reward_stats(_repro_target)
_repro_target_mass = _repro_mode_mass(_repro_target)
_repro_active = _repro_target_mass >= 0.01
_repro_smc_reward = _repro_reward_stats(_repro_smc)
_repro_smc_mass = _repro_mode_mass(_repro_smc)
_repro_emd = _repro_emd_repeats(_repro_smc)
_repro_metrics = {
    "schema_version": 2,
    "paper": "DAS",
    "experiment": "official GMM notebook Figure 1 - SMC-only rescue",
    "scope": "official cells through pretrained model plus official SMC cells; unrelated fine-tuning baselines skipped",
    "seed": _repro_seed,
    "numba_seed": _repro_numba_seed,
    "strict_determinism": {
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cublas_workspace_config": _repro_os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    },
    "wall_seconds": float(_repro_time.perf_counter() - _repro_started_at),
    "emd_estimator": {
        "implementation": "scipy.stats.wasserstein_distance_nd",
        "paper_style_subsample_n": 500,
        "paper_style_repeats": 1,
        "stability_subsample_n": 500,
        "stability_repeats": 100,
        "replacement": True,
    },
    "paper_style_smc_emd": _repro_paper_emd,
    "target": {
        "reward": _repro_target_reward,
        "mode_mass": _repro_target_mass.tolist(),
    },
    "pretrained": {
        "reward": _repro_reward_stats(_repro_pretrained),
        "mode_mass": _repro_mode_mass(_repro_pretrained).tolist(),
    },
    "smc_das": {
        "reward": _repro_smc_reward,
        "reward_gap_abs_to_target": abs(_repro_smc_reward["mean"] - _repro_target_reward["mean"]),
        "emd_repeats": _repro_emd,
        "emd_mean": float(np.mean(_repro_emd)),
        "emd_std": float(np.std(_repro_emd, ddof=1)),
        "emd_q025": float(np.quantile(_repro_emd, 0.025)),
        "emd_q975": float(np.quantile(_repro_emd, 0.975)),
        "mode_mass": _repro_smc_mass.tolist(),
        "mode_mass_l1_to_target": float(np.abs(_repro_smc_mass - _repro_target_mass).sum()),
        "target_mode_coverage": int(np.logical_and(_repro_smc_mass >= 0.01, _repro_active).sum()),
        "target_mode_count": int(_repro_active.sum()),
        "spurious_mass_outside_target_modes": float(_repro_smc_mass[~_repro_active].sum()),
    },
}

_ReproPath("das_metrics.json").write_text(
    _repro_json.dumps(_repro_metrics, indent=2) + "\n", encoding="utf-8"
)
np.savez_compressed(
    "das_samples.npz",
    target=_repro_target,
    pretrained=_repro_pretrained,
    smc_das=_repro_smc,
)
print("DAS_REPRO_METRICS_JSON=" + _repro_json.dumps(_repro_metrics, sort_keys=True))
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--numba-seed", type=int)
    parser.add_argument("--smc-only", action="store_true")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    numba_seed = args.seed if args.numba_seed is None else args.numba_seed
    notebook = nbformat.read(args.source, as_version=4)
    if not notebook.cells or notebook.cells[0].cell_type != "code":
        raise RuntimeError("Expected the official seed setup in code cell 0")

    source_lines = notebook.cells[0].source.splitlines(keepends=True)
    replacements: list[dict[str, str]] = []
    for old in SEED_LINES:
        matching = [index for index, line in enumerate(source_lines) if line.rstrip("\r\n") == old]
        if len(matching) != 1:
            raise RuntimeError(f"Expected exactly one seed line: {old!r}")
        new = old.replace("42", str(args.seed))
        index = matching[0]
        newline = source_lines[index][len(source_lines[index].rstrip("\r\n")) :]
        source_lines[index] = new + newline
        replacements.append({"old": old, "new": new})
    notebook.cells[0].source = "".join(source_lines)
    notebook.cells.insert(
        1,
        nbformat.v4.new_code_cell(
            DETERMINISM_CELL.replace("__REPRO_NUMBA_SEED__", str(numba_seed))
        ),
    )
    if args.smc_only:
        # Original indices shift by one after the determinism cell insertion.
        keep_original = set(range(0, 17)) | set(range(30, 35))
        notebook.cells = [
            cell
            for index, cell in enumerate(notebook.cells)
            if index == 1 or (index - (1 if index > 1 else 0)) in keep_original
        ]
        metrics_source = SMC_ONLY_METRICS_CELL
    else:
        metrics_source = METRICS_CELL
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            metrics_source
            .replace("__REPRO_SEED__", str(args.seed))
            .replace("__REPRO_NUMBA_SEED__", str(numba_seed))
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output)
    manifest = {
        "schema_version": 1,
        "purpose": "runtime-only seed substitution and metric persistence",
        "source": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "seed": args.seed,
        "numba_seed": numba_seed,
        "smc_only": args.smc_only,
        "seed_replacements": replacements,
        "determinism_cell_inserted": True,
        "appended_metric_cell": True,
        "upstream_modified": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
