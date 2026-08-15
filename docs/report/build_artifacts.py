#!/usr/bin/env python3
"""Aggregate remote results and build publication-quality reproduction figures."""

from __future__ import annotations

import json
import hashlib
import math
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "report" / "figures"
RESULTS_DIR = ROOT / "results"
KAGGLE_DAS_RUN_DIR = ROOT / "outputs" / "kaggle_jobs" / "das_seed42_t4x2_20260811T155327Z"
KAGGLE_DAS_DIAGNOSTICS = (
    KAGGLE_DAS_RUN_DIR
    / "output"
    / "kaggle_job_ops"
    / "das-seed42-kaggle-t4x2-20260811"
    / "diagnostics"
)
KAGGLE_CANARY_RUN_DIR = ROOT / "outputs" / "kaggle_jobs" / "t4x2_canary_r4_20260811T1530Z"
KAGGLE_CANARY_DIAGNOSTICS = (
    KAGGLE_CANARY_RUN_DIR / "output_direct" / "kaggle_job_ops" / "standalone" / "diagnostics"
)
MODAL_DAS_SEED42_DIR = (
    ROOT
    / "remote_artifacts"
    / "das-gmm-seed42-20260811-r2"
    / "output"
    / "das-gmm-seed42-20260811-r2"
)
FINAL_DAS_RUNS = {
    0: "das-gmm-seed0-20260811-r2",
    1: "das-gmm-seed1-20260811-r2",
    2: "das-gmm-seed2-20260811-r1",
    3: "das-gmm-seed3-20260811-r1",
    42: "das-gmm-seed42-20260811-r2",
}
PAPER_DAS = {
    "target_reward": -0.29,
    "reward": {
        "rl_ddpo": -0.07,
        "direct_backprop": -0.96,
        "approx_guidance": -338.0,
        "smc_das": -0.22,
    },
    "emd": {
        "rl_ddpo": 1.58,
        "direct_backprop": 3.69,
        "approx_guidance": 15.49,
        "smc_das": 0.82,
    },
}
METHODS = ["rl_ddpo", "direct_backprop", "approx_guidance", "smc_das"]
METHOD_LABELS = ["RL/DDPO", "Direct BP", "Approx. guidance", "DAS/SMC"]
COLORS = {
    "rl_ddpo": "#0077BB",
    "direct_backprop": "#EE7733",
    "approx_guidance": "#CC3311",
    "smc_das": "#009988",
    "target": "#000000",
}
T95_DF4 = 2.7764451051977987
ENVIRONMENT_SENSITIVE_TOLERANCE = 0.10


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def one_match(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename} under {root}, found {matches}")
    return matches[0]


def mean_sd_ci(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    sd = statistics.stdev(values)
    half = T95_DF4 * sd / math.sqrt(len(values))
    return {"mean": mean, "sd": sd, "ci95_low": mean - half, "ci95_high": mean + half}


def save_figure(fig: mpl.figure.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png")
    plt.close(fig)


def notebook_semantic_sha256(path: Path) -> str:
    """Hash notebook semantics while ignoring random per-cell nbformat IDs."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    normalized = {
        "nbformat": notebook["nbformat"],
        "nbformat_minor": notebook["nbformat_minor"],
        "metadata": notebook.get("metadata", {}),
        "cells": [
            {key: value for key, value in cell.items() if key != "id"}
            for cell in notebook["cells"]
        ],
    }
    encoded = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def symmetric_relative_difference(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-12)


def load_kaggle_portability() -> dict:
    kaggle_metrics_path = KAGGLE_DAS_DIAGNOSTICS / "notebook_output" / "das_metrics.json"
    kaggle_summary_path = KAGGLE_DAS_DIAGNOSTICS / "das_kaggle_summary.json"
    kaggle_patch_path = KAGGLE_DAS_DIAGNOSTICS / "runtime_patch.json"
    kaggle_notebook_path = (
        KAGGLE_DAS_DIAGNOSTICS / "notebook_output" / "GMM.seed-42.instrumented.ipynb"
    )
    kaggle_accelerator_path = (
        KAGGLE_DAS_RUN_DIR / "output" / "kaggle_job_ops" / "standalone" / "accelerator_summary.json"
    )
    canary_path = KAGGLE_CANARY_DIAGNOSTICS / "t4x2_canary_summary.json"
    modal_metrics_path = MODAL_DAS_SEED42_DIR / "das_metrics.json"
    modal_summary_path = MODAL_DAS_SEED42_DIR / "modal_summary.json"
    modal_patch_path = MODAL_DAS_SEED42_DIR / "runtime_patch.json"
    modal_notebook_path = MODAL_DAS_SEED42_DIR / "GMM.seed-42.instrumented.ipynb"

    paths = [
        kaggle_metrics_path,
        kaggle_summary_path,
        kaggle_patch_path,
        kaggle_notebook_path,
        kaggle_accelerator_path,
        canary_path,
        modal_metrics_path,
        modal_summary_path,
        modal_patch_path,
        modal_notebook_path,
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing Kaggle portability evidence: {missing}")

    kaggle_metrics = json.loads(kaggle_metrics_path.read_text(encoding="utf-8"))
    kaggle_summary = json.loads(kaggle_summary_path.read_text(encoding="utf-8"))
    kaggle_patch = json.loads(kaggle_patch_path.read_text(encoding="utf-8"))
    kaggle_accelerator = json.loads(kaggle_accelerator_path.read_text(encoding="utf-8"))
    canary = json.loads(canary_path.read_text(encoding="utf-8"))
    modal_metrics = json.loads(modal_metrics_path.read_text(encoding="utf-8"))
    modal_summary = json.loads(modal_summary_path.read_text(encoding="utf-8"))
    modal_patch = json.loads(modal_patch_path.read_text(encoding="utf-8"))

    if kaggle_metrics["seed"] != 42 or modal_metrics["seed"] != 42:
        raise RuntimeError("Kaggle/Modal portability comparison must use seed 42 on both backends")

    kaggle_das = kaggle_metrics["methods"]["smc_das"]
    modal_das = modal_metrics["methods"]["smc_das"]
    target_reward_equal = (
        kaggle_metrics["target"]["reward"]["mean"]
        == modal_metrics["target"]["reward"]["mean"]
    )
    target_mode_mass_equal = (
        kaggle_metrics["target"]["mode_mass"] == modal_metrics["target"]["mode_mass"]
    )
    source_sha_equal = kaggle_patch["source_sha256"] == modal_patch["source_sha256"]
    kaggle_semantic_sha = notebook_semantic_sha256(kaggle_notebook_path)
    modal_semantic_sha = notebook_semantic_sha256(modal_notebook_path)
    semantic_notebook_equal = kaggle_semantic_sha == modal_semantic_sha

    kaggle_best_baseline_emd = min(
        item["emd_mean"]
        for name, item in kaggle_metrics["methods"].items()
        if name != "smc_das"
    )
    kaggle_claim_pass = (
        kaggle_das["emd_mean"] < kaggle_best_baseline_emd
        and kaggle_das["target_mode_coverage"] == kaggle_das["target_mode_count"]
    )
    emd_symmetric_difference = symmetric_relative_difference(
        kaggle_das["emd_mean"], modal_das["emd_mean"]
    )
    paper_emd_symmetric_difference = symmetric_relative_difference(
        kaggle_das["emd_mean"], PAPER_DAS["emd"]["smc_das"]
    )

    return {
        "schema_version": 1,
        "classification": "environment-sensitive same-seed portability probe",
        "seed": 42,
        "paper_point": {
            "smc_emd": PAPER_DAS["emd"]["smc_das"],
            "smc_reward": PAPER_DAS["reward"]["smc_das"],
            "target_reward": PAPER_DAS["target_reward"],
            "reward_gap_abs_to_target": abs(
                PAPER_DAS["reward"]["smc_das"] - PAPER_DAS["target_reward"]
            ),
        },
        "modal_l4": {
            "run_id": modal_summary["run_id"],
            "environment": modal_summary["environment"],
            "target_reward": modal_metrics["target"]["reward"]["mean"],
            "smc_reward": modal_das["reward"]["mean"],
            "smc_reward_gap_abs_to_target": modal_das["reward_gap_abs_to_target"],
            "smc_emd_mean": modal_das["emd_mean"],
            "smc_emd_subsample_sd": modal_das["emd_std"],
            "target_mode_coverage": modal_das["target_mode_coverage"],
            "target_mode_count": modal_das["target_mode_count"],
            "mode_mass_l1_to_target": modal_das["mode_mass_l1_to_target"],
            "spurious_mass_outside_target_modes": modal_das[
                "spurious_mass_outside_target_modes"
            ],
        },
        "kaggle_t4x2": {
            "kernel_id": "codemaivanngu/diffusion-smc-das-seed42-t4x2-20260811",
            "kernel_url": "https://www.kaggle.com/code/codemaivanngu/diffusion-smc-das-seed42-t4x2-20260811",
            "environment": kaggle_summary["environment"],
            "uv_version": kaggle_summary["uv_version"],
            "uv_sync_elapsed_s": kaggle_summary["uv_sync_elapsed_s"],
            "notebook_elapsed_s": kaggle_summary["run_manifest"]["elapsed_s"],
            "target_reward": kaggle_metrics["target"]["reward"]["mean"],
            "smc_reward": kaggle_das["reward"]["mean"],
            "smc_reward_gap_abs_to_target": kaggle_das["reward_gap_abs_to_target"],
            "smc_emd_mean": kaggle_das["emd_mean"],
            "smc_emd_subsample_sd": kaggle_das["emd_std"],
            "target_mode_coverage": kaggle_das["target_mode_coverage"],
            "target_mode_count": kaggle_das["target_mode_count"],
            "mode_mass_l1_to_target": kaggle_das["mode_mass_l1_to_target"],
            "spurious_mass_outside_target_modes": kaggle_das[
                "spurious_mass_outside_target_modes"
            ],
            "best_baseline_emd": kaggle_best_baseline_emd,
            "allocation_gpu_count": kaggle_accelerator["gpu_device_count"],
            "allocation_gpu_names": kaggle_accelerator["gpu_device_names"],
            "scientific_workload_gpu_count": 1,
        },
        "identity_checks": {
            "target_reward_exact_match": target_reward_equal,
            "target_mode_mass_exact_match": target_mode_mass_equal,
            "official_source_sha256_exact_match": source_sha_equal,
            "official_source_sha256": kaggle_patch["source_sha256"],
            "instrumented_notebook_semantic_exact_match": semantic_notebook_equal,
            "instrumented_notebook_semantic_sha256": kaggle_semantic_sha,
            "note": "Byte hashes differ only because nbformat generated a random cell id; semantic hash excludes cell id.",
        },
        "differences": {
            "smc_emd_absolute": abs(kaggle_das["emd_mean"] - modal_das["emd_mean"]),
            "smc_emd_kaggle_percent_higher_than_modal": (
                (kaggle_das["emd_mean"] - modal_das["emd_mean"])
                / abs(modal_das["emd_mean"])
                * 100
            ),
            "smc_emd_symmetric_relative": emd_symmetric_difference,
            "smc_reward_absolute": abs(
                kaggle_das["reward"]["mean"] - modal_das["reward"]["mean"]
            ),
            "paper_emd_symmetric_relative": paper_emd_symmetric_difference,
        },
        "verdicts": {
            "t4x2_canary": canary["status"],
            "scientific_execution": kaggle_summary["status"],
            "within_kaggle_seed_claim": "PASS" if kaggle_claim_pass else "FAIL",
            "paper_numeric_match_10pct": (
                "PASS"
                if paper_emd_symmetric_difference < ENVIRONMENT_SENSITIVE_TOLERANCE
                else "FAIL"
            ),
            "cross_backend_numeric_portability_10pct": (
                "REPRODUCIBLE"
                if emd_symmetric_difference < ENVIRONMENT_SENSITIVE_TOLERANCE
                else "NOT_REPRODUCIBLE"
            ),
        },
        "tolerance": {
            "environment_sensitive_symmetric_relative": ENVIRONMENT_SENSITIVE_TOLERANCE,
            "source": "experiment-agent reproducibility protocol default",
        },
        "canary": {
            "kernel_id": "codemaivanngu/diffusion-smc-t4x2-canary-r4-20260811",
            "kernel_url": "https://www.kaggle.com/code/codemaivanngu/diffusion-smc-t4x2-canary-r4-20260811",
            "summary": canary,
        },
        "source_artifacts": [str(path.relative_to(ROOT)) for path in paths],
        "qualification": (
            "This same-seed comparison is not an additional independent seed and is excluded "
            "from the five-seed Modal aggregate. The T4x2 allocation was verified on both GPUs, "
            "but the official DAS notebook itself uses one CUDA device."
        ),
    }


def load_das() -> tuple[list[dict], dict[int, Path]]:
    rows = []
    sample_paths: dict[int, Path] = {}
    for seed, run_id in FINAL_DAS_RUNS.items():
        root = ROOT / "remote_artifacts" / run_id
        metric_path = one_match(root, "das_metrics.json")
        sample_paths[seed] = one_match(root, "das_samples.npz")
        payload = json.loads(metric_path.read_text())
        if payload["seed"] != seed:
            raise RuntimeError(f"Seed mismatch for {run_id}: {payload['seed']} != {seed}")
        rows.append({"run_id": run_id, "metric_path": str(metric_path.relative_to(ROOT)), **payload})
    rows.sort(key=lambda row: row["seed"])
    return rows, sample_paths


def aggregate_das(rows: list[dict]) -> dict:
    method_summary = {}
    for method in ["pretrained", *METHODS]:
        method_summary[method] = {
            "emd": mean_sd_ci([row["methods"][method]["emd_mean"] for row in rows]),
            "reward": mean_sd_ci([row["methods"][method]["reward"]["mean"] for row in rows]),
            "reward_gap_abs_to_target": mean_sd_ci(
                [row["methods"][method]["reward_gap_abs_to_target"] for row in rows]
            ),
            "mode_mass_l1_to_target": mean_sd_ci(
                [row["methods"][method]["mode_mass_l1_to_target"] for row in rows]
            ),
        }

    primary_seed_pass = []
    for row in rows:
        das = row["methods"]["smc_das"]
        baseline_min = min(row["methods"][method]["emd_mean"] for method in METHODS[:-1])
        primary_seed_pass.append(
            {
                "seed": row["seed"],
                "pass": das["emd_mean"] < baseline_min
                and das["target_mode_coverage"] == das["target_mode_count"],
                "das_emd": das["emd_mean"],
                "best_baseline_emd": baseline_min,
                "mode_coverage": das["target_mode_coverage"],
                "target_mode_count": das["target_mode_count"],
            }
        )
    pass_count = sum(item["pass"] for item in primary_seed_pass)
    emd_ci = method_summary["smc_das"]["emd"]
    gap_ci = method_summary["smc_das"]["reward_gap_abs_to_target"]

    if emd_ci["ci95_high"] < PAPER_DAS["emd"]["smc_das"]:
        emd_improvement = "PASS"
    elif emd_ci["ci95_low"] > PAPER_DAS["emd"]["smc_das"]:
        emd_improvement = "FAIL"
    else:
        emd_improvement = "INCONCLUSIVE"

    paper_gap = abs(PAPER_DAS["reward"]["smc_das"] - PAPER_DAS["target_reward"])
    if gap_ci["ci95_high"] < paper_gap:
        reward_gap_improvement = "PASS"
    elif gap_ci["ci95_low"] > paper_gap:
        reward_gap_improvement = "FAIL"
    else:
        reward_gap_improvement = "INCONCLUSIVE"

    return {
        "schema_version": 1,
        "paper": "Test-time Alignment of Diffusion Models without Reward Over-optimization",
        "upstream_commit": "2f4b2239f29ee59f80359bdfa5ed747b6d855a1e",
        "seeds": [row["seed"] for row in rows],
        "run_ids": [row["run_id"] for row in rows],
        "paper_figure1_top_row": PAPER_DAS,
        "method_summary_across_seeds": method_summary,
        "primary_claim": {
            "criterion": "DAS EMD is lower than every reproduced baseline and all target-active modes are covered, per seed",
            "required_passes": 4,
            "observed_passes": pass_count,
            "verdict": "PASS" if pass_count >= 4 else "FAIL",
            "per_seed": primary_seed_pass,
        },
        "improvement_over_paper_point_estimate": {
            "emd_lower_is_better": {
                "paper": PAPER_DAS["emd"]["smc_das"],
                "criterion": "95% t CI for mean lies entirely below paper point estimate",
                "verdict": emd_improvement,
            },
            "absolute_reward_gap_lower_is_better": {
                "paper": paper_gap,
                "criterion": "95% t CI for mean gap lies entirely below paper point estimate",
                "verdict": reward_gap_improvement,
            },
        },
        "qualification": (
            "Five independent training/sampling seeds. EMD is the mean of five paper-style 500-sample "
            "subsample estimates within each seed; between-seed statistics use the five seed means. "
            "Paper Figure 1 provides point estimates but no seed-level uncertainty."
        ),
        "per_seed_payloads": rows,
    }


def plot_das_primary(rows: list[dict], summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35))
    x = np.arange(len(METHODS))
    offsets = np.linspace(-0.12, 0.12, len(rows))

    ax = axes[0]
    for offset, row in zip(offsets, rows):
        values = [row["methods"][method]["emd_mean"] for method in METHODS]
        ax.scatter(x + offset, values, marker="o", s=22, facecolor="white", edgecolor="#555555", linewidth=0.7)
    paper = [PAPER_DAS["emd"][method] for method in METHODS]
    means = [summary["method_summary_across_seeds"][method]["emd"]["mean"] for method in METHODS]
    ax.scatter(x, paper, marker="X", s=48, color="#CC3311", label="Paper point")
    ax.scatter(x, means, marker="D", s=34, color="#0077BB", label="Repro mean")
    ax.set_yscale("log")
    ax.set_xticks(x, METHOD_LABELS, rotation=22, ha="right")
    ax.set_ylabel("EMD to target (log scale; lower is better)")
    ax.set_title("(a) Distributional match")
    ax.legend(frameon=False)

    ax = axes[1]
    for offset, row in zip(offsets, rows):
        values = [row["methods"][method]["reward_gap_abs_to_target"] for method in METHODS]
        ax.scatter(x + offset, values, marker="o", s=22, facecolor="white", edgecolor="#555555", linewidth=0.7)
    paper_gap = [abs(PAPER_DAS["reward"][method] - PAPER_DAS["target_reward"]) for method in METHODS]
    means = [
        summary["method_summary_across_seeds"][method]["reward_gap_abs_to_target"]["mean"]
        for method in METHODS
    ]
    ax.scatter(x, paper_gap, marker="X", s=48, color="#CC3311", label="Paper point")
    ax.scatter(x, means, marker="D", s=34, color="#0077BB", label="Repro mean")
    ax.set_yscale("log")
    ax.set_xticks(x, METHOD_LABELS, rotation=22, ha="right")
    ax.set_ylabel(r"Absolute reward gap $|\bar r-r_{target}|$ (log)")
    ax.set_title("(b) Reward fidelity (secondary)")

    ax = axes[2]
    seeds = [row["seed"] for row in rows]
    emd = [row["methods"]["smc_das"]["emd_mean"] for row in rows]
    estimator_sd = [row["methods"]["smc_das"]["emd_std"] for row in rows]
    worst = int(np.argmax(emd))
    colors = ["#009988"] * len(rows)
    colors[worst] = "#CC3311"
    ax.errorbar(np.arange(len(rows)), emd, yerr=estimator_sd, fmt="none", ecolor="#666666", capsize=3)
    ax.scatter(np.arange(len(rows)), emd, c=colors, s=40, edgecolor="black", linewidth=0.5)
    ax.axhline(PAPER_DAS["emd"]["smc_das"], color="#CC3311", linestyle="--", linewidth=1, label="Paper DAS = 0.82")
    ax.set_xticks(np.arange(len(rows)), [str(seed) for seed in seeds])
    ax.set_xlabel("Training/sampling seed")
    ax.set_ylabel("DAS EMD (mean ± subsample SD)")
    ax.set_title("(c) Seed and estimator variability")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_das_primary")


def plot_das_modes(rows: list[dict]) -> None:
    target_mass = np.mean([row["target"]["mode_mass"] for row in rows], axis=0)
    das_mass = np.asarray([row["methods"]["smc_das"]["mode_mass"] for row in rows])
    matrix = np.vstack([target_mass, das_mass])
    labels = ["Target mean", *[f"DAS seed {row['seed']}" for row in rows]]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    im = ax.imshow(matrix, cmap="cividis", vmin=0, vmax=max(0.36, float(matrix.max())), aspect="auto")
    ax.set_xticks(np.arange(9), ["(-4,4)", "(0,4)", "(4,4)", "(-4,0)", "(0,0)", "(4,0)", "(-4,-4)", "(0,-4)", "(4,-4)"], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("(a) Nearest-mode mass")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > 0.18 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Assigned sample mass")

    ax = axes[1]
    seeds = [str(row["seed"]) for row in rows]
    l1 = [row["methods"]["smc_das"]["mode_mass_l1_to_target"] for row in rows]
    spurious = [row["methods"]["smc_das"]["spurious_mass_outside_target_modes"] for row in rows]
    pos = np.arange(len(rows))
    width = 0.36
    ax.bar(pos - width / 2, l1, width, color="#0077BB", edgecolor="black", linewidth=0.4, label="Mode-mass L1")
    ax.bar(pos + width / 2, spurious, width, color="#EE7733", edgecolor="black", linewidth=0.4, label="Spurious mass")
    ax.set_xticks(pos, seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mass discrepancy (lower is better)")
    ax.set_title("(b) Mode fidelity by seed")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig_das_modes")


def plot_das_samples(rows: list[dict], sample_paths: dict[int, Path]) -> None:
    emd_by_seed = {row["seed"]: row["methods"]["smc_das"]["emd_mean"] for row in rows}
    ranked = sorted(emd_by_seed, key=emd_by_seed.get)
    chosen = [("Best", ranked[0]), ("Median", ranked[len(ranked) // 2]), ("Worst", ranked[-1])]
    reference_seed = rows[0]["seed"]
    ref = np.load(sample_paths[reference_seed])
    rng = np.random.default_rng(20260811)

    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.75), sharex=True, sharey=True)
    idx = rng.choice(len(ref["target"]), size=min(2500, len(ref["target"])), replace=False)
    axes[0].scatter(ref["target"][idx, 0], ref["target"][idx, 1], s=2, alpha=0.35, color="#000000")
    axes[0].set_title("Target")
    axes[0].set_ylabel("Y")
    for ax, (rank, seed) in zip(axes[1:], chosen):
        data = np.load(sample_paths[seed])["smc_das"]
        idx = rng.choice(len(data), size=min(2500, len(data)), replace=False)
        ax.scatter(data[idx, 0], data[idx, 1], s=2, alpha=0.35, color="#009988")
        ax.set_title(f"{rank}: seed {seed}\nEMD={emd_by_seed[seed]:.3f}")
    for ax in axes:
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_xlabel("X")
        ax.set_aspect("equal")
    fig.tight_layout()
    save_figure(fig, "fig_das_samples")


def plot_fk_steering() -> None:
    payload = json.loads((RESULTS_DIR / "fk_steering_one_prompt.json").read_text())
    seeds = payload["seeds"]
    series = payload["image_reward_across_seed_runs"]
    names = [("fk_max", "FK-max", "o", "#0077BB"), ("fk_diff", "FK-diff", "s", "#009988"), ("base", "Base", "^", "#EE7733")]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for key, label, marker, color in names:
        values = series[key]["per_seed_mean"]
        ax.plot(seeds, values, marker=marker, color=color, linewidth=1.2, markersize=5, label=f"{label} (SD={series[key]['sample_sd_of_means']:.2f})")
    ax.set_xticks(seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Mean ImageReward over 4 particles")
    ax.set_title("One-prompt FK Steering reproduction: large seed variance")
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    fig.tight_layout()
    save_figure(fig, "fig_fk_steering_variance")


def plot_fkc() -> None:
    payload = json.loads((RESULTS_DIR / "fk_correctors_table_a1_scaled.json").read_text())
    methods = list(payload["metrics_mean_pm_sample_sd"])
    metrics = ["energy_w2", "mmd", "total_variation", "w1", "w2"]
    labels = ["Target / no FKC", "Tempered / no FKC", "Target / systematic", "Tempered / systematic"]
    matrix = np.empty((len(methods), len(metrics)))
    for i, method in enumerate(methods):
        item = payload["metrics_mean_pm_sample_sd"][method]
        for j, metric in enumerate(metrics):
            repro = item["reproduction"][metric][0]
            paper = item["paper"][metric][0]
            matrix[i, j] = (repro - paper) / paper * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    im = ax.imshow(np.clip(matrix, -200, 200), cmap="RdBu_r", vmin=-200, vmax=200, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), ["Energy-W2", "MMD", "TV", "W1", "W2"])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_title("Scaled FK Correctors reproduction: relative difference from Table A1")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if abs(matrix[i, j]) > 90 else "black"
            ax.text(j, i, f"{matrix[i, j]:+.0f}%", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("(scaled repro − paper) / paper (%)")
    fig.tight_layout()
    save_figure(fig, "fig_fkc_relative_error")


def plot_das_kaggle_portability(portability: dict) -> None:
    paper = portability["paper_point"]
    modal = portability["modal_l4"]
    kaggle = portability["kaggle_t4x2"]
    labels = ["Paper point", "Modal L4\nseed 42", "Kaggle T4x2\nseed 42"]
    colors = ["#BBBBBB", "#0077BB", "#EE7733"]
    hatches = ["//", "", "xx"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.55))

    emd = [paper["smc_emd"], modal["smc_emd_mean"], kaggle["smc_emd_mean"]]
    emd_error = [0.0, modal["smc_emd_subsample_sd"], kaggle["smc_emd_subsample_sd"]]
    bars = axes[0].bar(
        x,
        emd,
        yerr=emd_error,
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("DAS EMD to target (lower is better)")
    axes[0].set_title("(a) Same method, same seed, different backend")
    axes[0].set_ylim(0, max(emd) * 1.28)
    for index, value in enumerate(emd):
        axes[0].text(index, value + 0.07, f"{value:.3f}", ha="center", va="bottom")

    reward_gap = [
        paper["reward_gap_abs_to_target"],
        modal["smc_reward_gap_abs_to_target"],
        kaggle["smc_reward_gap_abs_to_target"],
    ]
    bars = axes[1].bar(
        x,
        reward_gap,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel(r"Absolute reward gap $|\bar r-r_{target}|$ (log)")
    axes[1].set_title("(b) Reward fidelity diagnostic")
    axes[1].set_ylim(0.025, 6.0)
    for index, value in enumerate(reward_gap):
        axes[1].text(index, value * 1.18, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    save_figure(fig, "fig_das_kaggle_portability")


def write_generated_tex(summary: dict, portability: dict) -> None:
    das = summary["method_summary_across_seeds"]["smc_das"]
    emd = das["emd"]
    reward = das["reward"]
    gap = das["reward_gap_abs_to_target"]
    per_seed = summary["primary_claim"]["per_seed"]
    worst = max(per_seed, key=lambda item: item["das_emd"])
    best = min(per_seed, key=lambda item: item["das_emd"])
    kaggle = portability["kaggle_t4x2"]
    modal = portability["modal_l4"]
    differences = portability["differences"]
    verdicts = portability["verdicts"]
    macros = f"""% Generated by report/build_artifacts.py; do not edit manually.
\\newcommand{{\\DASSeedCount}}{{{len(summary['seeds'])}}}
\\newcommand{{\\DASEMDMean}}{{{emd['mean']:.3f}}}
\\newcommand{{\\DASEMDSD}}{{{emd['sd']:.3f}}}
\\newcommand{{\\DASEMDCILow}}{{{emd['ci95_low']:.3f}}}
\\newcommand{{\\DASEMDCIHigh}}{{{emd['ci95_high']:.3f}}}
\\newcommand{{\\DASRewardMean}}{{{reward['mean']:.3f}}}
\\newcommand{{\\DASRewardSD}}{{{reward['sd']:.3f}}}
\\newcommand{{\\DASRewardGapMean}}{{{gap['mean']:.3f}}}
\\newcommand{{\\DASRewardGapSD}}{{{gap['sd']:.3f}}}
\\newcommand{{\\DASPrimaryPassCount}}{{{summary['primary_claim']['observed_passes']}}}
\\newcommand{{\\DASPrimaryVerdict}}{{{summary['primary_claim']['verdict']}}}
\\newcommand{{\\DASEMDImprovementVerdict}}{{{summary['improvement_over_paper_point_estimate']['emd_lower_is_better']['verdict']}}}
\\newcommand{{\\DASRewardImprovementVerdict}}{{{summary['improvement_over_paper_point_estimate']['absolute_reward_gap_lower_is_better']['verdict']}}}
\\newcommand{{\\DASWorstSeed}}{{{worst['seed']}}}
\\newcommand{{\\DASWorstEMD}}{{{worst['das_emd']:.3f}}}
\\newcommand{{\\DASBestSeed}}{{{best['seed']}}}
\\newcommand{{\\DASBestEMD}}{{{best['das_emd']:.3f}}}
\\newcommand{{\\KaggleDASEMD}}{{{kaggle['smc_emd_mean']:.3f}}}
\\newcommand{{\\KaggleDASEMDSD}}{{{kaggle['smc_emd_subsample_sd']:.3f}}}
\\newcommand{{\\KaggleDASReward}}{{{kaggle['smc_reward']:.3f}}}
\\newcommand{{\\KaggleDASRewardGap}}{{{kaggle['smc_reward_gap_abs_to_target']:.3f}}}
\\newcommand{{\\KaggleDASModeLone}}{{{kaggle['mode_mass_l1_to_target']:.3f}}}
\\newcommand{{\\ModalSeedFortyTwoEMD}}{{{modal['smc_emd_mean']:.3f}}}
\\newcommand{{\\ModalSeedFortyTwoReward}}{{{modal['smc_reward']:.3f}}}
\\newcommand{{\\ModalSeedFortyTwoRewardGap}}{{{modal['smc_reward_gap_abs_to_target']:.3f}}}
\\newcommand{{\\KaggleModalEMDAbsoluteDiff}}{{{differences['smc_emd_absolute']:.3f}}}
\\newcommand{{\\KaggleModalEMDPercentHigher}}{{{differences['smc_emd_kaggle_percent_higher_than_modal']:.1f}\\%}}
\\newcommand{{\\KaggleModalEMDSymmetricDiff}}{{{differences['smc_emd_symmetric_relative'] * 100:.1f}\\%}}
\\newcommand{{\\KagglePaperEMDSymmetricDiff}}{{{differences['paper_emd_symmetric_relative'] * 100:.1f}\\%}}
\\newcommand{{\\KaggleWithinSeedVerdict}}{{{verdicts['within_kaggle_seed_claim']}}}
\\newcommand{{\\KagglePaperNumericVerdict}}{{{verdicts['paper_numeric_match_10pct']}}}
\\newcommand{{\\KaggleCrossBackendVerdict}}{{{verdicts['cross_backend_numeric_portability_10pct'].replace('_', ' ')}}}
"""
    (ROOT / "report" / "generated_metrics.tex").write_text(macros, encoding="utf-8")
    rows = []
    payload_by_seed = {item["seed"]: item for item in summary["per_seed_payloads"]}
    for item in per_seed:
        payload = payload_by_seed[item["seed"]]
        das_item = payload["methods"]["smc_das"]
        rows.append(
            f"{item['seed']} & {payload['target']['reward']['mean']:.3f} & "
            f"{das_item['reward']['mean']:.3f} & {das_item['reward_gap_abs_to_target']:.3f} & "
            f"{das_item['emd_mean']:.3f} $\\pm$ {das_item['emd_std']:.3f} & "
            f"{das_item['target_mode_coverage']}/{das_item['target_mode_count']} & "
            f"{das_item['mode_mass_l1_to_target']:.3f} & {'PASS' if item['pass'] else 'FAIL'} \\\\"
        )
    (ROOT / "report" / "generated_das_seed_rows.tex").write_text(
        "\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8"
    )


def write_figure_trace() -> None:
    script_path = Path(__file__).resolve()
    script_hash = hashlib.sha256(script_path.read_bytes()).hexdigest()
    transformation = {"script": str(script_path.relative_to(ROOT)), "sha256": script_hash}
    entries = [
        {
            "artifact_id": "fig_das_primary",
            "source_data": ["results/das_gmm_multiseed.json"],
            "transformation": transformation,
            "caption_claim": "Shows paper points, five reproduction seeds, aggregate means, and within-seed EMD subsample variability without treating reward as the primary metric.",
            "supported_manuscript_claims": [
                {"locator": "DAS: tái lập GMM nhiều seed / Kết quả nhiều seed", "claim": "DAS is evaluated primarily by EMD and target-mode coverage, with reward gap secondary."}
            ],
            "limitations": ["The paper provides point estimates without seed-level uncertainty.", "The EMD y-axis is logarithmic."],
        },
        {
            "artifact_id": "fig_das_modes",
            "source_data": ["results/das_gmm_multiseed.json"],
            "transformation": transformation,
            "caption_claim": "Reports nearest-center mode mass and per-seed mode-fidelity diagnostics for DAS.",
            "supported_manuscript_claims": [
                {"locator": "DAS: tái lập GMM nhiều seed / Kết quả nhiều seed", "claim": "Mode fidelity is checked rather than inferred from raw reward."}
            ],
            "limitations": ["Nearest-center assignment and the 1 percent active-mode threshold are reproduction diagnostics, not preregistered paper metrics."],
        },
        {
            "artifact_id": "fig_das_samples",
            "source_data": ["remote_artifacts/das-gmm-seed*/**/das_samples.npz", "results/das_gmm_multiseed.json"],
            "transformation": transformation,
            "caption_claim": "Displays target and objectively selected best, median, and worst DAS seeds by EMD.",
            "supported_manuscript_claims": [
                {"locator": "DAS: tái lập GMM nhiều seed / Kết quả nhiều seed", "claim": "The report does not cherry-pick a visually favorable seed."}
            ],
            "limitations": ["Scatter plots are subsampled for legibility and are diagnostic rather than the verdict source."],
        },
        {
            "artifact_id": "fig_fk_steering_variance",
            "source_data": ["results/fk_steering_one_prompt.json"],
            "transformation": transformation,
            "caption_claim": "Shows large seed-to-seed ImageReward variation and rank changes among three samplers for one prompt.",
            "supported_manuscript_claims": [
                {"locator": "FK Steering: một prompt và variance", "claim": "One prompt and three seeds do not support a benchmark-wide superiority claim."}
            ],
            "limitations": ["Only one prompt and three seeds were evaluated; model runtime used a public community mirror."],
        },
        {
            "artifact_id": "fig_fkc_relative_error",
            "source_data": ["results/fk_correctors_table_a1_scaled.json"],
            "transformation": transformation,
            "caption_claim": "Shows relative cell-wise differences between the scaled reproduction and the full paper Table A1.",
            "supported_manuscript_claims": [
                {"locator": "FK Correctors: Table A1 subset thu nhỏ", "claim": "Some cells are close, but the scaled run is not an exact numeric reproduction."}
            ],
            "limitations": ["The reproduction uses 2000 samples and 200 steps instead of 10000 and 1000, and omits BDC.", "Color values are clipped at plus or minus 200 percent while annotations retain actual values."],
        },
        {
            "artifact_id": "fig_das_kaggle_portability",
            "source_data": [
                "results/das_kaggle_t4x2_portability.json",
                "remote_artifacts/das-gmm-seed42-20260811-r2/**/das_metrics.json",
                "outputs/kaggle_jobs/das_seed42_t4x2_20260811T155327Z/**/das_metrics.json",
            ],
            "transformation": transformation,
            "caption_claim": "Compares paper, Modal L4, and Kaggle T4x2 DAS EMD and reward-gap diagnostics without merging the duplicate seed into the multi-seed aggregate.",
            "supported_manuscript_claims": [
                {
                    "locator": "DAS: portability trên Kaggle T4x2",
                    "claim": "The Kaggle run passes the within-run ranking criterion but fails paper numeric match and same-seed cross-backend numeric portability."
                }
            ],
            "limitations": [
                "Modal and Kaggle share seed 42, so this is an environment-sensitive portability probe rather than an independent replicate.",
                "EMD error bars are within-seed subsample SD, not cross-seed uncertainty.",
                "The Kaggle allocation exposes two T4 GPUs, but the official DAS notebook uses one CUDA device.",
            ],
        },
    ]
    (ROOT / "report" / "figure_table_trace.json").write_text(json.dumps({"figure_table_trace": entries}, indent=2) + "\n")


def main() -> int:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, sample_paths = load_das()
    summary = aggregate_das(rows)
    portability = load_kaggle_portability()
    (RESULTS_DIR / "das_gmm_multiseed.json").write_text(json.dumps(summary, indent=2) + "\n")
    (RESULTS_DIR / "das_kaggle_t4x2_portability.json").write_text(
        json.dumps(portability, indent=2) + "\n", encoding="utf-8"
    )
    plot_das_primary(rows, summary)
    plot_das_modes(rows)
    plot_das_samples(rows, sample_paths)
    plot_das_kaggle_portability(portability)
    plot_fk_steering()
    plot_fkc()
    write_generated_tex(summary, portability)
    write_figure_trace()
    print(
        json.dumps(
            {
                "das_summary": summary["method_summary_across_seeds"]["smc_das"],
                "verdict": summary["primary_claim"],
                "kaggle_portability_verdicts": portability["verdicts"],
                "figures": sorted(path.name for path in FIGURE_DIR.glob("*.pdf")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
