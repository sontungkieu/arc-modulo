#!/usr/bin/env python3
"""Build the final, evidence-gated figures and TeX fragments.

The script intentionally fails until the complete Null-TTA 3x50 matrix and the
successful Kaggle T4x2 rescue artifacts are present.  This prevents a report
build from silently falling back to canaries or failed engineering attempts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
RESULTS = ROOT / "results"

NULL_PATH = RESULTS / "null_tta_table1.json"
NULL_RUN_MANIFEST = RESULTS / "null_tta_run_manifest.json"
DAS_PATH = RESULTS / "das_rescue_multiseed.json"
FK_PATH = RESULTS / "fk_steering_one_prompt.json"
FKC_PATH = RESULTS / "fk_correctors_table_a1_exact.json"
KAGGLE_RUN = (
    ROOT
    / "outputs/kaggle_jobs/das_rescue_seed42_t4x2_r6_20260811T201300Z"
    / "output"
)
BILLING_PATH = RESULTS / "modal_billing_final.json"
PAPER_NULL = {
    "pickscore": (0.218, 0.315),
    "hpsv2": (0.279, 0.294),
    "aesthetic": (5.232, 5.431),
    "imagereward": (0.339, 0.946),
}
METRIC_LABELS = {
    "pickscore": "PickScore",
    "hpsv2": "HPS v2",
    "aesthetic": "Aesthetic",
    "imagereward": "ImageReward",
}
FKC_METHOD_LABELS = {
    "target_score_no_fkc": "Target score / no FKC",
    "tempered_noise_no_fkc": "Tempered noise / no FKC",
    "target_score_birth_death_clock_fkc": "Target score / BDC",
    "tempered_noise_birth_death_clock_fkc": "Tempered noise / BDC",
    "target_score_systematic_fkc": "Target score / systematic",
    "tempered_noise_systematic_fkc": "Tempered noise / systematic",
}
FKC_METRIC_LABELS = {
    "energy_w2": "Energy-W2",
    "mmd": "MMD",
    "total_variation": "TV",
    "w1": "W1",
    "w2": "W2",
}


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


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def one_match(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename)) if root.is_dir() else []
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename} under {root}, found {matches}")
    return matches[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig: mpl.figure.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def symmetric_relative(left: float, right: float) -> float:
    return abs(left - right) / max((abs(left) + abs(right)) / 2, 1e-12)


def validate_null(payload: dict[str, Any]) -> None:
    if payload.get("seeds") != [42, 43, 44]:
        raise RuntimeError("Null-TTA report requires seeds 42, 43, 44")
    if payload.get("prompt_count_per_seed") != 50 or not payload.get("all_scorers_are_real"):
        raise RuntimeError("Null-TTA report requires 50 prompts/seed and four real scorers")
    if payload.get("upstream_commit") != "337bf73037e9f24e9f844974d3287384abb610bf":
        raise RuntimeError("Unexpected Null-TTA upstream commit")
    for metric, (paper_base, paper_opt) in PAPER_NULL.items():
        item = payload["aggregate"][metric]
        if item["paper_baseline"] != paper_base or item["paper_null_tta_nmax55"] != paper_opt:
            raise RuntimeError(f"Paper target drift for {metric}")


def validate_null_manifest(payload: dict[str, Any]) -> None:
    if payload.get("canonical_artifact_count") != 25:
        raise RuntimeError("Null-TTA report requires exactly 25 canonical artifact records")
    if payload.get("scientific_modal_app_count") != 26:
        raise RuntimeError("Null-TTA report requires exactly 26 scientific Modal app ids")
    if payload.get("total_seed_prompt_records") != 150:
        raise RuntimeError("Null-TTA report requires exactly 150 seed-prompt records")
    if not payload.get("coverage_complete_and_nonoverlapping") or not payload.get("all_scorers_are_real"):
        raise RuntimeError("Null-TTA shard coverage/scorer manifest failed")
    if payload.get("upstream_commit") != "337bf73037e9f24e9f844974d3287384abb610bf":
        raise RuntimeError("Null-TTA manifest commit drift")


def validate_das(payload: dict[str, Any]) -> None:
    if payload.get("seeds") != [42, 43, 44]:
        raise RuntimeError("DAS report requires matched-RNG seeds 42, 43, 44")
    numeric = payload.get("numeric_match", {})
    if numeric.get("canonical_seed") != 42:
        raise RuntimeError("DAS canonical seed must be 42")
    if not numeric.get("all_seeds_cover_3_of_3_target_modes"):
        raise RuntimeError("DAS mode coverage guard failed")


def validate_fkc(payload: dict[str, Any]) -> None:
    if payload.get("primary_verdict") != "PASS_WITH_VARIANCE":
        raise RuntimeError("FK Correctors exact report requires PASS_WITH_VARIANCE")
    protocol = payload.get("protocol", {})
    expected_protocol = {
        "num_samples": 10000,
        "num_integration_steps": 1000,
        "dt": 0.001,
        "num_runs": 5,
    }
    if any(protocol.get(key) != value for key, value in expected_protocol.items()):
        raise RuntimeError("FK Correctors exact protocol drift")
    numeric = payload.get("numeric_match", {})
    if (
        numeric.get("cell_count") != 30
        or numeric.get("within_paper_2sd_band_count") != 30
        or not numeric.get("pass")
    ):
        raise RuntimeError("FK Correctors numeric compatibility gate failed")
    if len(payload.get("methods", {})) != 6 or len(payload.get("raw_rows", [])) != 30:
        raise RuntimeError("FK Correctors exact method/run coverage failed")


def validate_billing(payload: dict[str, Any]) -> None:
    expected = {"kieusontung6", "kieusontung8", "phamvanvuhoan"}
    snapshots = payload.get("snapshots", {})
    if set(snapshots) != expected:
        raise RuntimeError(f"Final Modal billing requires exactly {sorted(expected)}")
    if payload.get("guard", {}).get("hard_limit_per_workspace") != 29.0:
        raise RuntimeError("Final Modal billing must preserve the USD 29 hard guard")
    if not payload.get("all_under_hard_limit"):
        raise RuntimeError("Final Modal billing hard-limit check failed")
    expected_app_counts = {"kieusontung6": 10, "kieusontung8": 8, "phamvanvuhoan": 8}
    for profile, row in snapshots.items():
        usage = float(row["final_reported_usage"])
        if usage >= 29.0:
            raise RuntimeError(f"Modal hard guard exceeded for {profile}: {usage}")
        if (
            row.get("canonical_app_count") != expected_app_counts[profile]
            or row.get("provider_billed_canonical_app_count") != expected_app_counts[profile]
            or float(row.get("canonical_run_cost", 0.0)) <= 0.0
        ):
            raise RuntimeError(f"Incomplete canonical billing join for {profile}")
        if not row.get("reported_at_utc"):
            raise RuntimeError(f"Missing provider snapshot timestamp for {profile}")


def plot_null(payload: dict[str, Any]) -> None:
    colors = {"paper": "#7A5195", "repro": "#00876C"}
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    for ax, metric in zip(axes.flat, PAPER_NULL, strict=True):
        item = payload["aggregate"][metric]
        per_seed = payload["per_seed"]
        base = item["baseline_between_seed"]["mean"]
        opt = item["optimized_between_seed"]["mean"]
        ci_low, ci_high = item["optimized_hierarchical_bootstrap_ci95"]
        paper_base, paper_opt = PAPER_NULL[metric]
        ax.bar([0, 1], [paper_base, paper_opt], width=0.32, color=colors["paper"], alpha=0.82, label="Paper")
        ax.bar([0.38, 1.38], [base, opt], width=0.32, color=colors["repro"], alpha=0.9, label="Reproduction")
        ax.errorbar(
            [1.38],
            [opt],
            yerr=[[opt - ci_low], [ci_high - opt]],
            color="black",
            capsize=3,
            linewidth=1,
            zorder=4,
        )
        for seed, marker in zip((42, 43, 44), ("o", "s", "D"), strict=True):
            y = per_seed[str(seed)][metric]["optimized"]["mean"]
            ax.scatter(1.38, y, marker=marker, s=28, facecolor="white", edgecolor="black", zorder=5)
        ax.set_xticks([0.19, 1.19], ["SD-v1.5", "Null-TTA"])
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.2)
        ax.text(
            0.98,
            0.04,
            f"sym. diff={100 * item['symmetric_relative_difference']:.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
    axes.flat[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Null-TTA Table 1: paper points and complete 3-seed reproduction", y=1.01)
    fig.tight_layout()
    save(fig, "fig_null_tta_table1")

    pick = payload["aggregate"]["pickscore"]
    fig, ax = plt.subplots(figsize=(7.5, 4.1))
    seeds = [42, 43, 44]
    baselines = [payload["per_seed"][str(s)]["pickscore"]["baseline"]["mean"] for s in seeds]
    optimized = [payload["per_seed"][str(s)]["pickscore"]["optimized"]["mean"] for s in seeds]
    for i, seed in enumerate(seeds):
        ax.plot([0, 1], [baselines[i], optimized[i]], marker="o", linewidth=1.8, label=f"seed {seed}")
    ax.scatter([1.12], [pick["optimized_between_seed"]["mean"]], marker="D", s=58, color="#00876C", label="3-seed mean")
    ax.axhline(PAPER_NULL["pickscore"][0], color="#7A5195", linestyle=":", label="paper baseline")
    ax.axhline(PAPER_NULL["pickscore"][1], color="#7A5195", linestyle="--", label="paper Null-TTA")
    ax.set_xlim(-0.12, 1.25)
    ax.set_xticks([0, 1], ["SD-v1.5", "Null-TTA"])
    ax.set_ylabel("PickScore")
    ax.set_title("Null-TTA seed sensitivity on the primary metric")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_null_tta_seed_variance")


def plot_das(payload: dict[str, Any]) -> None:
    seeds = [42, 43, 44]
    paper_style = [payload["per_seed"][str(s)]["paper_style_smc_emd"] for s in seeds]
    stable = [payload["per_seed"][str(s)]["stable_smc_emd_mean"] for s in seeds]
    stable_sd = [payload["per_seed"][str(s)]["stable_smc_emd_std"] for s in seeds]
    targets = [payload["per_seed"][str(s)]["target_reward"] for s in seeds]
    rewards = [payload["per_seed"][str(s)]["smc_reward"] for s in seeds]

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    x = np.arange(len(seeds))
    axes[0].scatter(x - 0.08, paper_style, marker="o", s=48, color="#D55E00", label="paper-style 1x EMD")
    axes[0].errorbar(x + 0.08, stable, yerr=stable_sd, marker="s", capsize=3, color="#0072B2", label="100-repeat mean ± SD")
    axes[0].axhline(payload["paper_points"]["smc_emd"], linestyle="--", color="black", label="paper 0.82")
    axes[0].set_xticks(x, [str(s) for s in seeds])
    axes[0].set_xlabel("matched Python/Torch/Numba seed")
    axes[0].set_ylabel("EMD (lower is better)")
    axes[0].set_title("Distributional fidelity and bad seed")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)

    width = 0.35
    axes[1].bar(x - width / 2, targets, width, color="#999999", label="own target")
    axes[1].bar(x + width / 2, rewards, width, color="#009E73", label="DAS/SMC")
    axes[1].axhline(payload["paper_points"]["target_reward"], linestyle=":", color="black", label="paper target")
    axes[1].axhline(payload["paper_points"]["smc_reward"], linestyle="--", color="#D55E00", label="paper DAS")
    axes[1].set_xticks(x, [str(s) for s in seeds])
    axes[1].set_xlabel("seed")
    axes[1].set_ylabel("mean reward")
    axes[1].set_title("Reward is a guardrail, not the sole target")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, "fig_das_rescue_variance")

    sample_paths = [one_match(ROOT / Path(payload["source_files"][i]).parents[1], "das_samples.npz") for i in range(3)]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), sharex=True, sharey=True)
    for ax, seed, sample_path in zip(axes, seeds, sample_paths, strict=True):
        samples = np.load(sample_path)
        target = samples["target"]
        smc = samples["smc_das"]
        ax.scatter(target[::10, 0], target[::10, 1], s=5, alpha=0.24, color="#666666", label="target")
        ax.scatter(smc[::2, 0], smc[::2, 1], s=6, alpha=0.36, color="#0072B2", label="DAS/SMC")
        ax.set_title(f"seed {seed}; EMD={paper_style[seeds.index(seed)]:.3f}")
        ax.set_xlabel("x")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False, markerscale=2)
    fig.suptitle("DAS target samples versus SMC outputs", y=1.01)
    fig.tight_layout()
    save(fig, "fig_das_rescue_samples")


def load_kaggle(das: dict[str, Any]) -> tuple[dict[str, Any], list[Path]]:
    metrics_path = one_match(KAGGLE_RUN, "das_metrics.json")
    summary_path = one_match(KAGGLE_RUN, "das_kaggle_summary.json")
    accelerator_path = one_match(KAGGLE_RUN, "accelerator_summary.json")
    metrics = load_json(metrics_path)
    summary = load_json(summary_path)
    accelerator = load_json(accelerator_path)
    if summary.get("status") != "PASS" or metrics.get("seed") != 42 or metrics.get("numba_seed") != 42:
        raise RuntimeError("Kaggle scientific payload is incomplete or not matched-RNG seed 42")
    names = accelerator.get("gpu_device_names", [])
    if accelerator.get("gpu_device_count") != 2 or len(names) != 2 or any("T4" not in name for name in names):
        raise RuntimeError(f"Kaggle T4x2 accelerator contract failed: {accelerator}")
    modal = das["per_seed"]["42"]
    paper_emd = float(das["paper_points"]["smc_emd"])
    kaggle_paper_style = float(metrics["paper_style_smc_emd"])
    kaggle_stable = float(metrics["smc_das"]["emd_mean"])
    output = {
        "schema_version": 1,
        "kernel_id": "codemaivanngu/diffusion-smc-das-seed42-numba-rescue-t4x2-r6",
        "kernel_url": "https://www.kaggle.com/code/codemaivanngu/diffusion-smc-das-seed42-numba-rescue-t4x2-r6",
        "seed": 42,
        "numba_seed": 42,
        "allocation": {"gpu_count": 2, "gpu_names": names},
        "scientific_workload_gpu_count": 1,
        "paper": {"paper_style_smc_emd": paper_emd, "smc_reward": -0.22, "target_reward": -0.29},
        "modal_l4": modal,
        "kaggle_t4x2": {
            "paper_style_smc_emd": kaggle_paper_style,
            "stable_smc_emd_mean": kaggle_stable,
            "stable_smc_emd_std": float(metrics["smc_das"]["emd_std"]),
            "target_reward": float(metrics["target"]["reward"]["mean"]),
            "smc_reward": float(metrics["smc_das"]["reward"]["mean"]),
            "target_mode_coverage": int(metrics["smc_das"]["target_mode_coverage"]),
            "target_mode_count": int(metrics["smc_das"]["target_mode_count"]),
        },
        "numeric": {
            "kaggle_to_paper_symmetric_relative": symmetric_relative(kaggle_paper_style, paper_emd),
            "kaggle_to_modal_symmetric_relative": symmetric_relative(kaggle_paper_style, float(modal["paper_style_smc_emd"])),
            "paper_match_10pct": symmetric_relative(kaggle_paper_style, paper_emd) <= 0.10,
            "modal_match_10pct": symmetric_relative(kaggle_paper_style, float(modal["paper_style_smc_emd"])) <= 0.10,
        },
        "source_files": [str(p.relative_to(ROOT)) for p in (metrics_path, summary_path, accelerator_path)],
    }
    (RESULTS / "das_kaggle_t4x2_rescue.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output, [metrics_path, summary_path, accelerator_path]


def plot_kaggle(payload: dict[str, Any]) -> None:
    labels = ["Paper", "Modal L4", "Kaggle T4x2"]
    emd = [
        payload["paper"]["paper_style_smc_emd"],
        payload["modal_l4"]["paper_style_smc_emd"],
        payload["kaggle_t4x2"]["paper_style_smc_emd"],
    ]
    reward = [payload["paper"]["smc_reward"], payload["modal_l4"]["smc_reward"], payload["kaggle_t4x2"]["smc_reward"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].bar(labels, emd, color=["#7A5195", "#0072B2", "#E69F00"])
    axes[0].set_ylabel("paper-style EMD")
    axes[0].set_title("Matched seed 42 numeric portability")
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(labels, reward, color=["#7A5195", "#0072B2", "#E69F00"])
    axes[1].axhline(-0.29, color="black", linestyle=":", label="paper target")
    axes[1].set_ylabel("SMC reward")
    axes[1].set_title("Reward comparison")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    save(fig, "fig_das_kaggle_rescue")


def plot_fk(payload: dict[str, Any]) -> None:
    methods = [("fk_max", "FK-max"), ("fk_diff", "FK-difference"), ("base", "Base k=4")]
    seeds = payload["seeds"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    x = np.arange(len(seeds))
    for key, label in methods:
        ax.plot(x, payload["image_reward_across_seed_runs"][key]["per_seed_mean"], marker="o", label=label)
    ax.set_xticks(x, [str(s) for s in seeds])
    ax.set_xlabel("seed")
    ax.set_ylabel("mean ImageReward")
    ax.set_title("FK Steering one-prompt seed variance")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_fk_steering_variance_final")


def plot_fkc(payload: dict[str, Any]) -> None:
    methods = list(FKC_METHOD_LABELS)
    metrics = list(FKC_METRIC_LABELS)
    by_cell = {(row["method"], row["metric"]): row for row in payload["cells"]}
    matrix = np.array(
        [
            [
                (by_cell[(method, metric)]["reproduction_mean"] - by_cell[(method, metric)]["paper_mean"])
                / by_cell[(method, metric)]["paper_sd"]
                for metric in metrics
            ]
            for method in methods
        ]
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(metrics)), [FKC_METRIC_LABELS[m] for m in metrics])
    ax.set_yticks(range(len(methods)), [FKC_METHOD_LABELS[m] for m in methods])
    ax.set_title("FK Correctors exact Table A1: signed distance from paper mean")
    fig.colorbar(image, ax=ax, label="(reproduction mean - paper mean) / paper SD")
    fig.tight_layout()
    save(fig, "fig_fkc_relative_error_final")

    run_matrix = np.array(
        [
            [
                next(
                    row["mean_absolute_distance_in_paper_sd"]
                    for row in payload["methods"][method]["runs_ranked_by_mean_paper_sd_distance"]
                    if row["run_index"] == run_index
                )
                for run_index in range(5)
            ]
            for method in methods
        ]
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    image = ax.imshow(
        run_matrix,
        cmap="YlOrRd",
        vmin=0,
        vmax=max(4.5, float(run_matrix.max())),
        aspect="auto",
    )
    for i in range(run_matrix.shape[0]):
        for j in range(run_matrix.shape[1]):
            ax.text(j, i, f"{run_matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(5), [f"run {i}" for i in range(5)])
    ax.set_yticks(range(len(methods)), [FKC_METHOD_LABELS[m] for m in methods])
    ax.set_title("FK Correctors run sensitivity across all five metrics")
    fig.colorbar(image, ax=ax, label="mean absolute distance from paper mean (paper SD units)")
    fig.tight_layout()
    save(fig, "fig_fkc_run_variance_final")


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def write_tex(
    null: dict[str, Any],
    das: dict[str, Any],
    kaggle: dict[str, Any],
    fkc: dict[str, Any],
    billing: dict[str, Any],
) -> list[Path]:
    pick = null["aggregate"]["pickscore"]
    das_emd = das["aggregate"]["paper_style_smc_emd"]
    macros = {
        "NullVerdict": null["primary_verdict"],
        "NullPickMean": f"{pick['optimized_between_seed']['mean']:.3f}",
        "NullPickSD": f"{pick['optimized_between_seed']['std']:.3f}",
        "NullPickCILow": f"{pick['optimized_hierarchical_bootstrap_ci95'][0]:.3f}",
        "NullPickCIHigh": f"{pick['optimized_hierarchical_bootstrap_ci95'][1]:.3f}",
        "NullPickSymDiff": f"{100 * pick['symmetric_relative_difference']:.1f}\\%",
        "NullPickImprovement": f"{pick['improvement_between_seed']['mean']:.3f}",
        "NullPickImprovementCILow": f"{pick['improvement_hierarchical_bootstrap_ci95'][0]:.3f}",
        "NullPickImprovementCIHigh": f"{pick['improvement_hierarchical_bootstrap_ci95'][1]:.3f}",
        "DASVerdict": tex_escape(das["primary_verdict"]),
        "DASCanonicalEMD": f"{das['numeric_match']['canonical_seed_emd']:.3f}",
        "DASCanonicalSymDiff": f"{100 * das['numeric_match']['canonical_seed_symmetric_relative_difference']:.1f}\\%",
        "DASThreeMean": f"{das_emd['mean']:.3f}",
        "DASThreeSD": f"{das_emd['std']:.3f}",
        "DASWorstSeed": "43",
        "DASWorstEMD": f"{das['per_seed']['43']['paper_style_smc_emd']:.3f}",
        "KaggleEMD": f"{kaggle['kaggle_t4x2']['paper_style_smc_emd']:.3f}",
        "KaggleReward": f"{kaggle['kaggle_t4x2']['smc_reward']:.3f}",
        "KagglePaperSymDiff": f"{100 * kaggle['numeric']['kaggle_to_paper_symmetric_relative']:.1f}\\%",
        "KaggleModalSymDiff": f"{100 * kaggle['numeric']['kaggle_to_modal_symmetric_relative']:.1f}\\%",
        "KagglePaperVerdict": "PASS" if kaggle["numeric"]["paper_match_10pct"] else "FAIL",
        "KaggleModalVerdict": "PASS" if kaggle["numeric"]["modal_match_10pct"] else "FAIL",
        "FKCVerdict": tex_escape(fkc["primary_verdict"]),
        "FKCWithinOneSD": str(fkc["numeric_match"]["within_paper_1sd_band_count"]),
        "FKCWithinTwoSD": str(fkc["numeric_match"]["within_paper_2sd_band_count"]),
        "FKCDirectionAgreement": str(fkc["numeric_match"]["corrector_direction_agreement_count"]),
        "FKCElapsedMinutes": f"{fkc['remote_run']['elapsed_s'] / 60:.1f}",
        "FKCCost": f"{fkc['remote_run']['actual_provider_cost_usd']:.3f}",
        "FKCMaxVRAM": f"{fkc['remote_run']['cuda_max_memory_allocated_bytes'] / 2**30:.2f}",
    }
    billing_names = {
        "kieusontung6": "TungSix",
        "kieusontung8": "TungEight",
        "phamvanvuhoan": "VuHoan",
    }
    for profile, suffix in billing_names.items():
        row = billing["snapshots"][profile]
        macros[f"Billing{suffix}Usage"] = f"{float(row['final_reported_usage']):.2f}"
        macros[f"Billing{suffix}Cost"] = f"{float(row['canonical_run_cost']):.2f}"
        macros[f"Billing{suffix}Headroom"] = f"{29.0 - float(row['final_reported_usage']):.2f}"
    macro_path = REPORT / "generated_final_metrics.tex"
    macro_path.write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n", encoding="utf-8")

    null_rows = []
    for seed in null["seeds"]:
        row = null["per_seed"][str(seed)]
        null_rows.append(
            f"{seed} & {row['pickscore']['baseline']['mean']:.3f} & {row['pickscore']['optimized']['mean']:.3f} & "
            f"{row['hpsv2']['optimized']['mean']:.3f} & {row['aesthetic']['optimized']['mean']:.3f} & "
            f"{row['imagereward']['optimized']['mean']:.3f} \\\\"
        )
    null_rows_path = REPORT / "generated_null_seed_rows.tex"
    # Keep the closing booktabs rule in the same input stream as the final row.
    # A rule immediately after an external \input boundary can be parsed after
    # a non-alignment token and trigger ``Misplaced \noalign`` in pdfTeX.
    null_rows_path.write_text("\n".join(null_rows) + "\n\\bottomrule\n", encoding="utf-8")

    # Put methods/sources on rows and metrics on columns, matching the layout of
    # the original paper table.  A second, metric-oriented table carries CIs and
    # validation diagnostics so the main point-estimate table stays readable.
    null_method_specs = [
        ("Paper / SD-v1.5", lambda item: item["paper_baseline"]),
        ("Reproduction / SD-v1.5", lambda item: item["baseline_between_seed"]["mean"]),
        (r"Paper / Null-TTA $n_{\max}=55$", lambda item: item["paper_null_tta_nmax55"]),
        (
            r"Reproduction / Null-TTA $n_{\max}=55$",
            lambda item: item["optimized_between_seed"]["mean"],
        ),
    ]
    null_method_rows = []
    for row_index, (label, getter) in enumerate(null_method_specs):
        values = [getter(null["aggregate"][metric]) for metric in PAPER_NULL]
        prefix = r"\addlinespace[2pt] " if row_index == 2 else ""
        null_method_rows.append(
            prefix + label + " & " + " & ".join(f"{value:.3f}" for value in values) + r" \\"
        )
    null_method_rows_path = REPORT / "generated_null_method_rows.tex"
    null_method_rows_path.write_text(
        "\n".join(null_method_rows) + "\n\\bottomrule\n", encoding="utf-8"
    )

    null_validation_rows = []
    for metric in PAPER_NULL:
        item = null["aggregate"][metric]
        opt_ci = item["optimized_hierarchical_bootstrap_ci95"]
        improvement_ci = item["improvement_hierarchical_bootstrap_ci95"]
        verdict = r"\PASS" if item["numeric_match_10pct"] else r"\FAIL"
        null_validation_rows.append(
            f"{METRIC_LABELS[metric]} & "
            f"{item['optimized_between_seed']['mean']:.3f} [{opt_ci[0]:.3f}, {opt_ci[1]:.3f}] & "
            f"{item['improvement_between_seed']['mean']:.3f} "
            f"[{improvement_ci[0]:.3f}, {improvement_ci[1]:.3f}] & "
            f"{100 * item['symmetric_relative_difference']:.1f}\\% & {verdict} \\\\"
        )
    null_validation_rows_path = REPORT / "generated_null_validation_rows.tex"
    null_validation_rows_path.write_text(
        "\n".join(null_validation_rows) + "\n\\bottomrule\n", encoding="utf-8"
    )

    das_rows = []
    for seed in das["seeds"]:
        row = das["per_seed"][str(seed)]
        relative = symmetric_relative(row["paper_style_smc_emd"], das["paper_points"]["smc_emd"])
        das_rows.append(
            f"{seed} & {row['paper_style_smc_emd']:.3f} & {row['stable_smc_emd_mean']:.3f} $\\pm$ {row['stable_smc_emd_std']:.3f} & "
            f"{row['smc_reward']:.3f} & {row['target_mode_coverage']}/3 & {100 * relative:.1f}\\% \\\\"
        )
    das_rows_path = REPORT / "generated_das_rescue_rows.tex"
    das_rows_path.write_text("\n".join(das_rows) + "\n\\bottomrule\n", encoding="utf-8")

    fkc_rows = []
    for method in FKC_METHOD_LABELS:
        cells = fkc["methods"][method]["metrics"]
        paper_values = [
            f"{cells[metric]['paper_mean']:.3f}$\\pm${cells[metric]['paper_sd']:.3f}"
            for metric in FKC_METRIC_LABELS
        ]
        reproduction_values = [
            f"{cells[metric]['reproduction_mean']:.3f}$\\pm$"
            f"{cells[metric]['reproduction_sd_sample']:.3f}"
            for metric in FKC_METRIC_LABELS
        ]
        fkc_rows.append(
            f"{FKC_METHOD_LABELS[method]} & Paper & " + " & ".join(paper_values) + " \\\\"
        )
        fkc_rows.append(
            " & Repro & " + " & ".join(reproduction_values) + " \\\\ \\addlinespace[1.5pt]"
        )
    fkc_rows_path = REPORT / "generated_fkc_rows.tex"
    fkc_rows_path.write_text("\n".join(fkc_rows) + "\n\\bottomrule\n", encoding="utf-8")
    return [
        macro_path,
        null_rows_path,
        null_method_rows_path,
        null_validation_rows_path,
        das_rows_path,
        fkc_rows_path,
    ]


def write_markdown_summary(
    null: dict[str, Any],
    das: dict[str, Any],
    kaggle: dict[str, Any],
    fkc: dict[str, Any],
    billing: dict[str, Any],
) -> Path:
    pick = null["aggregate"]["pickscore"]
    pick_ci = pick["optimized_hierarchical_bootstrap_ci95"]
    improvement_ci = pick["improvement_hierarchical_bootstrap_ci95"]
    method_rows = []
    method_specs = [
        ("Paper / SD-v1.5", lambda item: item["paper_baseline"]),
        ("Reproduction / SD-v1.5", lambda item: item["baseline_between_seed"]["mean"]),
        ("Paper / Null-TTA nmax=55", lambda item: item["paper_null_tta_nmax55"]),
        (
            "Reproduction / Null-TTA nmax=55",
            lambda item: item["optimized_between_seed"]["mean"],
        ),
    ]
    for label, getter in method_specs:
        values = [getter(null["aggregate"][metric]) for metric in PAPER_NULL]
        method_rows.append(f"| {label} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    validation_rows = []
    for metric in PAPER_NULL:
        item = null["aggregate"][metric]
        opt_ci = item["optimized_hierarchical_bootstrap_ci95"]
        improvement_ci_metric = item["improvement_hierarchical_bootstrap_ci95"]
        validation_rows.append(
            f"| {METRIC_LABELS[metric]} | "
            f"{item['optimized_between_seed']['mean']:.3f} [{opt_ci[0]:.3f}, {opt_ci[1]:.3f}] | "
            f"{item['improvement_between_seed']['mean']:.3f} "
            f"[{improvement_ci_metric[0]:.3f}, {improvement_ci_metric[1]:.3f}] | "
            f"{100 * item['symmetric_relative_difference']:.1f}% | "
            f"{'PASS' if item['numeric_match_10pct'] else 'FAIL'} |"
        )
    seed_rows = []
    for seed in null["seeds"]:
        row = null["per_seed"][str(seed)]["pickscore"]
        seed_rows.append(f"| {seed} | {row['baseline']['mean']:.6f} | {row['optimized']['mean']:.6f} |")
    cost_rows = []
    for profile in ("kieusontung6", "kieusontung8", "phamvanvuhoan"):
        row = billing["snapshots"][profile]
        cost_rows.append(
            f"| {profile} | {row['final_reported_usage']:.2f} | {row['canonical_run_cost']:.2f} | "
            f"{29.0 - row['final_reported_usage']:.2f} |"
        )

    markdown = f"""# Audited reproduction summary

This file is generated from the same gated JSON artifacts as the LaTeX report.
The canonical manuscript is `report/diffusion_smc_reproduction_report.tex`; the
verified release PDF is `output/pdf/diffusion_smc_reproduction_report.pdf`.

## Verdicts

| Work | Scope | Result |
|---|---|---|
| Null-TTA | Null-TTA nmax=55 on the SD-v1.5 backbone, 3 seeds x 50 prompts, 4 real scorers | **{null['primary_verdict']}** |
| DAS | Official GMM Figure 1, matched Python/Torch/Numba seed | **{das['primary_verdict']}** |
| DAS Kaggle | Seed 42 on verified 2 x Tesla T4 allocation; official workload uses one GPU | Infrastructure PASS; paper numeric {'PASS' if kaggle['numeric']['paper_match_10pct'] else 'FAIL'} |
| FK Steering | One prompt, 3 seeds, k=4 | **PARTIAL** |
| FK Correctors | Exact Table A1, 10k samples / 1000 steps / 5 runs / 6 methods | **{fkc['primary_verdict']}** |

## Null-TTA

Primary PickScore is {pick['optimized_between_seed']['mean']:.6f} across seeds
(paper 0.315; symmetric relative difference
{100 * pick['symmetric_relative_difference']:.2f}%; hierarchical-bootstrap 95%
CI [{pick_ci[0]:.6f}, {pick_ci[1]:.6f}]). Paired improvement is
{pick['improvement_between_seed']['mean']:.6f}, with 95% CI
[{improvement_ci[0]:.6f}, {improvement_ci[1]:.6f}]. Primary PASS requires both
the <=10% point match and an improvement CI strictly above zero.

Direct mapping to the original Table 1: `Paper/Repro SD-v1.5` reproduces the
first row (the unaligned backbone), while `Paper/Repro Null-TTA` reproduces the
last row (the same backbone after null-text embedding optimization).

| Source / method | PickScore | HPS v2 | Aesthetic | ImageReward |
|---|---:|---:|---:|---:|
{chr(10).join(method_rows)}

| Metric | Repro Null-TTA [95% CI] | Paired improvement [95% CI] | Symmetric diff | Match |
|---|---:|---:|---:|---|
{chr(10).join(validation_rows)}

| Seed | PickScore base | PickScore optimized |
|---:|---:|---:|
{chr(10).join(seed_rows)}

## DAS and Kaggle

The canonical Modal seed 42 paper-style EMD is
{das['numeric_match']['canonical_seed_emd']:.6f} versus paper 0.82
({100 * das['numeric_match']['canonical_seed_symmetric_relative_difference']:.2f}%
symmetric difference). The three-seed mean is
{das['aggregate']['paper_style_smc_emd']['mean']:.6f} +/-
{das['aggregate']['paper_style_smc_emd']['std']:.6f}; seed 43 is the bad seed at
{das['per_seed']['43']['paper_style_smc_emd']:.6f}. Kaggle r6 completes on a
verified T4x2 allocation but produces EMD
{kaggle['kaggle_t4x2']['paper_style_smc_emd']:.6f} and reward
{kaggle['kaggle_t4x2']['smc_reward']:.6f}; this is a numeric portability failure,
not an infrastructure failure.

## FK Correctors exact Table A1

The exact six-method protocol completed on an NVIDIA L4 in
{fkc['remote_run']['elapsed_s'] / 60:.2f} minutes for USD
{fkc['remote_run']['actual_provider_cost_usd']:.6f}. All
{fkc['numeric_match']['within_paper_2sd_band_count']}/30 reproduction means lie
inside the paper mean +/- 2 reported run-SD compatibility bands;
{fkc['numeric_match']['within_paper_1sd_band_count']}/30 lie inside one SD.
Corrector-versus-no-FKC directions agree in
{fkc['numeric_match']['corrector_direction_agreement_count']}/20 comparisons.
The only direction disagreement is target-score systematic Energy-W2, whose
paper and reproduction distributions overlap strongly. Upstream calls the loop
index a seed but does not reseed; this report therefore labels them sequential
runs rather than claiming known seed identities.

## Modal guard

| Profile | Final reported usage (USD) | Canonical app cost | Headroom to 29 |
|---|---:|---:|---:|
{chr(10).join(cost_rows)}

Provider billing can lag. Canonical cost is joined by the exact 26 scientific app IDs
behind 25 artifact records (one recovered artifact uses a source-generation app plus a
remote post-scoring app). Canaries and canceled attempts are retained separately in
`results/modal_billing_final.json`.

## Evidence

- `results/null_tta_run_manifest.json`: exact 25-artifact coverage, scorer, GPU and hashes.
- `results/null_tta_table1.json`: 150-record aggregate and bootstrap.
- `results/das_rescue_multiseed.json`: matched-RNG DAS results.
- `results/das_kaggle_t4x2_rescue.json`: Kaggle T4x2 portability result.
- `results/fk_correctors_table_a1_exact.json`: exact Table A1 cells, raw runs, variance and provenance.
- `report/final_figure_table_trace.json`: hashes for report inputs.

No local GPU experiment is represented as remote evidence. Upstream repositories
remain pinned and clean; engineering retries without scientific metrics are not
counted as model failures.
"""
    path = ROOT / "REPORT.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def main() -> int:
    null = load_json(NULL_PATH)
    null_manifest = load_json(NULL_RUN_MANIFEST)
    das = load_json(DAS_PATH)
    fk = load_json(FK_PATH)
    fkc = load_json(FKC_PATH)
    billing = load_json(BILLING_PATH)
    validate_null(null)
    validate_null_manifest(null_manifest)
    validate_das(das)
    validate_fkc(fkc)
    validate_billing(billing)
    kaggle, kaggle_paths = load_kaggle(das)

    plot_null(null)
    plot_das(das)
    plot_kaggle(kaggle)
    plot_fk(fk)
    plot_fkc(fkc)
    generated_tex = write_tex(null, das, kaggle, fkc, billing)
    markdown_summary = write_markdown_summary(null, das, kaggle, fkc, billing)

    sources = [NULL_PATH, NULL_RUN_MANIFEST, DAS_PATH, FK_PATH, FKC_PATH, BILLING_PATH, *kaggle_paths]
    trace = {
        "schema_version": 1,
        "sources": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sources
        ],
        "generated_tex": [str(path.relative_to(ROOT)) for path in generated_tex],
        "generated_documents": [str(markdown_summary.relative_to(ROOT))],
        "figures": sorted(str(path.relative_to(ROOT)) for path in FIGURES.glob("fig_*_final.*"))
        + sorted(str(path.relative_to(ROOT)) for path in FIGURES.glob("fig_null_tta_*.*"))
        + sorted(str(path.relative_to(ROOT)) for path in FIGURES.glob("fig_das_rescue_*.*"))
        + sorted(str(path.relative_to(ROOT)) for path in FIGURES.glob("fig_das_kaggle_rescue.*")),
    }
    trace_path = REPORT / "final_figure_table_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "trace": str(trace_path), "source_count": len(sources)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
