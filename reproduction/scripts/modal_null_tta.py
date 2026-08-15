#!/usr/bin/env python3
"""Standalone Modal launcher for Null-TTA with one selectable GPU type."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/repro")
UV_VERSION = "0.10.2"
GPU_TYPE = os.environ.get("NULL_TTA_MODAL_GPU", "L40S")
if GPU_TYPE not in {"L40S", "L4"}:
    raise ValueError(f"NULL_TTA_MODAL_GPU must be L40S or L4, got {GPU_TYPE!r}")

app = modal.App(f"null-tta-reproduction-{GPU_TYPE.lower()}")
artifacts = modal.Volume.from_name("diffusion-smc-repro-artifacts", create_if_missing=True)

env_root = ROOT / "envs/null-tta"
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "tk")
    .uv_pip_install(f"uv=={UV_VERSION}", uv_version=UV_VERSION)
    .add_local_file(env_root / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
    .add_local_file(env_root / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
    .run_commands(
        "cd /opt/repro && UV_PROJECT_ENVIRONMENT=/opt/repro/.venv uv sync --frozen --no-install-project",
    )
    .add_local_file(
        ROOT / "scripts/install_hpsv2_bpe.py",
        str(REMOTE_ROOT / "scripts/install_hpsv2_bpe.py"),
        copy=True,
    )
    .run_commands(
        "/opt/repro/.venv/bin/python /opt/repro/scripts/install_hpsv2_bpe.py",
    )
    .add_local_file(
        ROOT / "scripts/run_null_tta.py",
        str(REMOTE_ROOT / "scripts/run_null_tta.py"),
        copy=True,
    )
    .add_local_file(
        ROOT / "scripts/score_null_tta_images.py",
        str(REMOTE_ROOT / "scripts/score_null_tta_images.py"),
        copy=True,
    )
    .add_local_file(
        ROOT / "scripts/build_partial_null_tta_metrics.py",
        str(REMOTE_ROOT / "scripts/build_partial_null_tta_metrics.py"),
        copy=True,
    )
    .add_local_dir(
        ROOT / "remote_fragments/null-tta-table1-s44-p07-11-source/official",
        str(REMOTE_ROOT / "recovery/null-tta-table1-s44-p07-11-source/official"),
        copy=True,
    )
    .add_local_dir(
        ROOT / "upstream/null-tta",
        str(REMOTE_ROOT / "upstream/null-tta"),
        copy=True,
        ignore=[".git/**", "logs/**", "**/__pycache__/**"],
    )
)

RUNTIME_ENV = {
    "HOME": "/vol/cache/home",
    "HF_HOME": "/vol/cache/home/.cache/huggingface",
    "TORCH_HOME": "/vol/cache/home/.cache/torch",
    "XDG_CACHE_HOME": "/vol/cache/home/.cache",
    "MPLBACKEND": "Agg",
    "TOKENIZERS_PARALLELISM": "false",
    # The remote module is imported in a fresh process; carry the locally
    # selected decorator GPU type into that import so persisted provenance
    # cannot fall back to the default value.
    "NULL_TTA_MODAL_GPU": GPU_TYPE,
}


def prepare_run(run_id: str) -> Path:
    run_dir = Path("/vol/runs") / run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    return run_dir


def environment_payload() -> dict[str, object]:
    probe = """import json, sys, torch
print(json.dumps({
    'python': sys.version,
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'cuda_available': torch.cuda.is_available(),
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_count': torch.cuda.device_count(),
}))
"""
    completed = subprocess.run(
        ["/opt/repro/.venv/bin/python", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def finish(run_id: str, run_dir: Path, extra: dict[str, object]) -> dict[str, object]:
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_payload(),
        **extra,
    }
    (run_dir / "modal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifacts.commit()
    print(f"NULL_TTA_MODAL_SUMMARY_JSON={json.dumps(payload, sort_keys=True)}")
    return payload


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": artifacts},
    timeout=43200,
    env=RUNTIME_ENV,
)
def run_null_tta(
    run_id: str,
    seed: int,
    prompt_start: int = 0,
    prompt_end: int = 50,
    min_inner_steps: int = 5,
    max_inner_steps: int = 55,
    sequential_scorers: bool = False,
) -> dict[str, object]:
    run_dir = prepare_run(run_id)
    official_output = run_dir / "official"
    command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/run_null_tta.py",
        "--upstream-root",
        "/opt/repro/upstream/null-tta",
        "--output-dir",
        str(official_output),
        "--seed",
        str(seed),
        "--prompt-start",
        str(prompt_start),
        "--prompt-end",
        str(prompt_end),
        "--target-reward",
        "pickscore",
        "--min-inner-steps",
        str(min_inner_steps),
        "--max-inner-steps",
        str(max_inner_steps),
        "--num-particles",
        "3",
        "--num-inference-steps",
        "100",
        "--lambda-alpha",
        "100",
        "--lambda-reg",
        "0.002",
        "--phi-variance",
        "0.01",
        "--lr-uncond",
        "0.01",
        "--tampering-coef",
        "0.008",
    ]
    if sequential_scorers:
        command.append("--target-only-generation")
    else:
        command.append("--require-all-scorers")
    print(f"NULL_TTA_COMMAND_JSON={json.dumps(command)}")
    try:
        subprocess.run(command, check=True)
    except Exception as error:
        failure = {
            "paper": "null-tta",
            "status": "FAILED",
            "gpu_requested": GPU_TYPE,
            "seed": seed,
            "prompt_slice": [prompt_start, prompt_end],
            "min_inner_steps": min_inner_steps,
            "max_inner_steps": max_inner_steps,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        (run_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        artifacts.commit()
        raise

    metrics_path = official_output / "null_tta_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected persisted Null-TTA metrics: {metrics_path}")
    if sequential_scorers:
        postscore_command = [
            "/opt/repro/.venv/bin/python",
            "/opt/repro/scripts/score_null_tta_images.py",
            "--upstream-root",
            "/opt/repro/upstream/null-tta",
            "--metrics",
            str(metrics_path),
        ]
        print(f"NULL_TTA_POSTSCORE_COMMAND_JSON={json.dumps(postscore_command)}")
        subprocess.run(postscore_command, check=True)
    metrics = json.loads(metrics_path.read_text())
    if not metrics.get("target_scorer_is_real") or not metrics.get("all_scorers_are_real"):
        raise RuntimeError("Null-TTA Table 1 requires all four real scorers")
    return finish(
        run_id,
        run_dir,
        {
            "paper": "null-tta",
            "status": "COMPLETED",
            "gpu_requested": GPU_TYPE,
            "seed": seed,
            "prompt_slice": [prompt_start, prompt_end],
            "min_inner_steps": min_inner_steps,
            "max_inner_steps": max_inner_steps,
            "target_reward": "PickScore",
            "sequential_scorers": sequential_scorers,
            "target_metric": metrics["target_metric"],
            "metrics": metrics["metrics"],
            "all_scorers_are_real": True,
            "wall_seconds": metrics["wall_seconds"],
            "cuda_max_memory_allocated_mb": metrics["cuda_max_memory_allocated_mb"],
        },
    )


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={"/vol": artifacts},
    timeout=7200,
    env=RUNTIME_ENV,
)
def recover_partial_null_tta(
    run_id: str,
    source_run_id: str,
    source_app_id: str,
    source_profile: str,
    seed: int = 44,
    prompt_start: int = 7,
    prompt_end: int = 11,
) -> dict[str, object]:
    """Post-score a hash-recorded prefix committed by a cancelled source app."""

    run_dir = prepare_run(run_id)
    official_output = run_dir / "official"
    source = REMOTE_ROOT / "recovery/null-tta-table1-s44-p07-11-source/official"
    shutil.copytree(source, official_output)
    build_command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/build_partial_null_tta_metrics.py",
        "--upstream-root",
        "/opt/repro/upstream/null-tta",
        "--official-output",
        str(official_output),
        "--seed",
        str(seed),
        "--prompt-start",
        str(prompt_start),
        "--prompt-end",
        str(prompt_end),
        "--source-run-id",
        source_run_id,
        "--source-app-id",
        source_app_id,
        "--source-profile",
        source_profile,
    ]
    print(f"NULL_TTA_RECOVERY_BUILD_COMMAND_JSON={json.dumps(build_command)}")
    subprocess.run(build_command, check=True)
    metrics_path = official_output / "null_tta_metrics.json"
    postscore_command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/score_null_tta_images.py",
        "--upstream-root",
        "/opt/repro/upstream/null-tta",
        "--metrics",
        str(metrics_path),
    ]
    print(f"NULL_TTA_RECOVERY_POSTSCORE_COMMAND_JSON={json.dumps(postscore_command)}")
    subprocess.run(postscore_command, check=True)
    metrics = json.loads(metrics_path.read_text())
    if not metrics.get("target_scorer_is_real") or not metrics.get("all_scorers_are_real"):
        raise RuntimeError("Recovered Null-TTA shard requires all four real scorers")
    return finish(
        run_id,
        run_dir,
        {
            "paper": "null-tta",
            "status": "COMPLETED",
            "gpu_requested": GPU_TYPE,
            "seed": seed,
            "prompt_slice": [prompt_start, prompt_end],
            "min_inner_steps": 5,
            "max_inner_steps": 55,
            "target_reward": "PickScore",
            "sequential_scorers": True,
            "recovered_partial_shard": True,
            "source_run_id": source_run_id,
            "source_app_id": source_app_id,
            "source_profile": source_profile,
            "source_generation_gpu_requested": "L4",
            "target_metric": metrics["target_metric"],
            "metrics": metrics["metrics"],
            "all_scorers_are_real": True,
            "wall_seconds": metrics["wall_seconds"],
            "wall_seconds_scope": metrics["wall_seconds_scope"],
            "cuda_max_memory_allocated_mb": metrics["cuda_max_memory_allocated_mb"],
        },
    )
