#!/usr/bin/env python3
"""Run the three official-code reproductions on isolated Modal workspaces."""

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

app = modal.App("diffusion-smc-reproduction")
artifacts = modal.Volume.from_name("diffusion-smc-repro-artifacts", create_if_missing=True)

base = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_pip_install(f"uv=={UV_VERSION}", uv_version=UV_VERSION)
)

toy_env = (
    base.add_local_file(ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
    .add_local_file(ROOT / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
    .run_commands(
        "cd /opt/repro && UV_PROJECT_ENVIRONMENT=/opt/repro/.venv uv sync --frozen --no-install-project",
    )
)

das_image = (
    toy_env.add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_dir(ROOT / "upstream/das/das", str(REMOTE_ROOT / "upstream/das/das"), copy=True)
    .add_local_file(
        ROOT / "upstream/das/notebooks/GMM.ipynb",
        str(REMOTE_ROOT / "upstream/das/notebooks/GMM.ipynb"),
        copy=True,
    )
)

fkc_image = (
    toy_env.add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=True)
    .add_local_dir(
        ROOT / "upstream/fk-correctors/applications/temperature_annealing/runner/src",
        str(REMOTE_ROOT / "upstream/fk-correctors/applications/temperature_annealing/runner/src"),
        copy=True,
    )
    .add_local_dir(
        ROOT / "upstream/fk-correctors/applications/temperature_annealing/fab/fab",
        str(REMOTE_ROOT / "upstream/fk-correctors/applications/temperature_annealing/fab/fab"),
        copy=True,
    )
    .add_local_file(
        ROOT
        / "upstream/fk-correctors/applications/temperature_annealing/runner/notebooks/gmm_temp_annealed_birth_death.ipynb",
        str(
            REMOTE_ROOT
            / "upstream/fk-correctors/applications/temperature_annealing/runner/notebooks/gmm_temp_annealed_birth_death.ipynb"
        ),
        copy=True,
    )
)

fk_env_root = ROOT / "envs/fk-steering"
fk_image = (
    base.add_local_file(
        fk_env_root / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True
    )
    .add_local_file(fk_env_root / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
    .run_commands(
        "cd /opt/repro && UV_PROJECT_ENVIRONMENT=/opt/repro/.venv uv sync --frozen --no-install-project",
    )
    .add_local_dir(
        ROOT / "upstream/fk-steering/text_to_image",
        str(REMOTE_ROOT / "upstream/fk-steering/text_to_image"),
        copy=True,
        ignore=["samples_for_paper/**", "playground_fksteering.ipynb"],
    )
    .add_local_file(
        ROOT / "configs/fk_steering_one_prompt.json",
        str(REMOTE_ROOT / "configs/fk_steering_one_prompt.json"),
        copy=True,
    )
)

null_tta_env_root = ROOT / "envs/null-tta"
null_tta_image = (
    base.apt_install("tk")
    .add_local_file(
        null_tta_env_root / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True
    )
    .add_local_file(null_tta_env_root / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
    .run_commands(
        "cd /opt/repro && UV_PROJECT_ENVIRONMENT=/opt/repro/.venv uv sync --frozen --no-install-project",
    )
    .add_local_file(
        ROOT / "scripts/run_null_tta.py",
        str(REMOTE_ROOT / "scripts/run_null_tta.py"),
        copy=True,
    )
    .add_local_dir(
        ROOT / "upstream/null-tta",
        str(REMOTE_ROOT / "upstream/null-tta"),
        copy=True,
        ignore=[".git/**", "logs/**", "**/__pycache__/**"],
    )
)


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


def finish(run_id: str, run_dir: Path, extra: dict[str, object] | None = None) -> dict[str, object]:
    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_payload(),
    }
    if extra:
        payload.update(extra)
    (run_dir / "modal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifacts.commit()
    print(f"DIFFUSION_SMC_ENV_JSON={json.dumps(payload['environment'], sort_keys=True)}")
    print(f"DIFFUSION_SMC_SUMMARY_JSON={json.dumps(payload, sort_keys=True)}")
    return payload


@app.function(image=das_image, gpu="L4", volumes={"/vol": artifacts}, timeout=7200)
def run_das_gmm(run_id: str) -> dict[str, object]:
    run_dir = prepare_run(run_id)
    command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/run_notebook.py",
        "/opt/repro/upstream/das/notebooks/GMM.ipynb",
        "--output-dir",
        str(run_dir),
        "--upstream-root",
        "/opt/repro/upstream/das",
        "--upstream-commit",
        "2f4b2239f29ee59f80359bdfa5ed747b6d855a1e",
        "--pythonpath",
        "/opt/repro/upstream/das",
        "--timeout",
        "7000",
    ]
    print(f"DAS_COMMAND_JSON={json.dumps(command)}")
    subprocess.run(command, check=True)
    return finish(run_id, run_dir, {"paper": "das", "status": "COMPLETED"})


@app.function(image=das_image, gpu="L4", volumes={"/vol": artifacts}, timeout=7200)
def run_das_gmm_seed(run_id: str, seed: int) -> dict[str, object]:
    run_dir = prepare_run(run_id)
    source_notebook = Path("/opt/repro/upstream/das/notebooks/GMM.ipynb")
    runtime_notebook = Path("/tmp") / f"GMM.seed-{seed}.instrumented.ipynb"
    patch_command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/prepare_das_notebook.py",
        str(source_notebook),
        str(runtime_notebook),
        "--seed",
        str(seed),
        "--manifest",
        str(run_dir / "runtime_patch.json"),
    ]
    print(f"DAS_PATCH_COMMAND_JSON={json.dumps(patch_command)}")
    subprocess.run(patch_command, check=True)
    artifacts.commit()
    command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/run_notebook.py",
        str(runtime_notebook),
        "--output-dir",
        str(run_dir),
        "--upstream-root",
        "/opt/repro/upstream/das",
        "--upstream-commit",
        "2f4b2239f29ee59f80359bdfa5ed747b6d855a1e",
        "--pythonpath",
        "/opt/repro/upstream/das",
        "--timeout",
        "7000",
    ]
    print(f"DAS_COMMAND_JSON={json.dumps(command)}")
    subprocess.run(command, check=True)
    metrics_path = run_dir / "das_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected persisted DAS metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    return finish(
        run_id,
        run_dir,
        {
            "paper": "das",
            "status": "COMPLETED",
            "seed": seed,
            "reproduction_scope": "official GMM notebook with runtime-only metric persistence",
            "das_primary_metrics": {
                "target_reward": metrics["target"]["reward"]["mean"],
                "smc_reward": metrics["methods"]["smc_das"]["reward"]["mean"],
                "smc_emd_mean": metrics["methods"]["smc_das"]["emd_mean"],
                "smc_target_mode_coverage": metrics["methods"]["smc_das"]["target_mode_coverage"],
            },
        },
    )


@app.function(
    image=das_image,
    gpu="L4",
    volumes={"/vol": artifacts},
    timeout=7200,
    env={
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MPLBACKEND": "Agg",
    },
)
def run_das_smc_rescue(run_id: str, seed: int, numba_seed: int) -> dict[str, object]:
    """Run the official DAS pretrained-model and SMC cells with controlled RNGs."""
    run_dir = prepare_run(run_id)
    source_notebook = Path("/opt/repro/upstream/das/notebooks/GMM.ipynb")
    runtime_notebook = Path("/tmp") / f"GMM.seed-{seed}.numba-{numba_seed}.smc-rescue.ipynb"
    patch_command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/prepare_das_notebook.py",
        str(source_notebook),
        str(runtime_notebook),
        "--seed",
        str(seed),
        "--numba-seed",
        str(numba_seed),
        "--smc-only",
        "--manifest",
        str(run_dir / "runtime_patch.json"),
    ]
    print(f"DAS_RESCUE_PATCH_COMMAND_JSON={json.dumps(patch_command)}")
    subprocess.run(patch_command, check=True)
    artifacts.commit()
    command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/run_notebook.py",
        str(runtime_notebook),
        "--output-dir",
        str(run_dir),
        "--upstream-root",
        "/opt/repro/upstream/das",
        "--upstream-commit",
        "2f4b2239f29ee59f80359bdfa5ed747b6d855a1e",
        "--pythonpath",
        "/opt/repro/upstream/das",
        "--timeout",
        "7000",
    ]
    print(f"DAS_RESCUE_COMMAND_JSON={json.dumps(command)}")
    subprocess.run(command, check=True)
    metrics_path = run_dir / "das_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected persisted DAS rescue metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    return finish(
        run_id,
        run_dir,
        {
            "paper": "das",
            "status": "COMPLETED",
            "seed": seed,
            "numba_seed": numba_seed,
            "reproduction_scope": metrics["scope"],
            "das_primary_metrics": {
                "paper_style_smc_emd": metrics["paper_style_smc_emd"],
                "smc_emd_mean": metrics["smc_das"]["emd_mean"],
                "smc_emd_std": metrics["smc_das"]["emd_std"],
                "target_reward": metrics["target"]["reward"]["mean"],
                "smc_reward": metrics["smc_das"]["reward"]["mean"],
                "smc_target_mode_coverage": metrics["smc_das"]["target_mode_coverage"],
            },
        },
    )


NULL_TTA_ENV = {
    "HOME": "/vol/cache/home",
    "HF_HOME": "/vol/cache/home/.cache/huggingface",
    "TORCH_HOME": "/vol/cache/home/.cache/torch",
    "XDG_CACHE_HOME": "/vol/cache/home/.cache",
    "MPLBACKEND": "Agg",
    "TOKENIZERS_PARALLELISM": "false",
}


def _run_null_tta_sd_impl(
    run_id: str,
    seed: int,
    prompt_start: int = 0,
    prompt_end: int = 50,
    min_inner_steps: int = 5,
    max_inner_steps: int = 55,
) -> dict[str, object]:
    """Run an official-prompt slice of Null-TTA Table 1 on an L40S."""
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
        "--require-all-scorers",
    ]
    print(f"NULL_TTA_COMMAND_JSON={json.dumps(command)}")
    try:
        subprocess.run(command, check=True)
    except Exception as error:
        failure = {
            "paper": "null-tta",
            "status": "FAILED",
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
    metrics = json.loads(metrics_path.read_text())
    if not metrics.get("target_scorer_is_real"):
        raise RuntimeError("Null-TTA target scorer was not real PickScore")
    if not metrics.get("all_scorers_are_real"):
        raise RuntimeError("Null-TTA Table 1 run requires all four real scorers")
    return finish(
        run_id,
        run_dir,
        {
            "paper": "null-tta",
            "status": "COMPLETED",
            "seed": seed,
            "prompt_slice": [prompt_start, prompt_end],
            "min_inner_steps": min_inner_steps,
            "max_inner_steps": max_inner_steps,
            "target_reward": "PickScore",
            "target_metric": metrics["target_metric"],
            "metrics": metrics["metrics"],
            "target_scorer_is_real": True,
            "all_scorers_are_real": True,
            "wall_seconds": metrics["wall_seconds"],
            "cuda_max_memory_allocated_mb": metrics["cuda_max_memory_allocated_mb"],
        },
    )


@app.function(
    image=null_tta_image,
    gpu="L40S",
    volumes={"/vol": artifacts},
    timeout=86400,
    env=NULL_TTA_ENV,
)
def run_null_tta_sd(
    run_id: str,
    seed: int,
    prompt_start: int = 0,
    prompt_end: int = 50,
    min_inner_steps: int = 5,
    max_inner_steps: int = 55,
) -> dict[str, object]:
    return _run_null_tta_sd_impl(
        run_id, seed, prompt_start, prompt_end, min_inner_steps, max_inner_steps
    )


@app.function(
    image=null_tta_image,
    gpu="L4",
    volumes={"/vol": artifacts},
    timeout=43200,
    env=NULL_TTA_ENV,
)
def run_null_tta_sd_l4(
    run_id: str,
    seed: int,
    prompt_start: int = 0,
    prompt_end: int = 50,
    min_inner_steps: int = 5,
    max_inner_steps: int = 55,
) -> dict[str, object]:
    return _run_null_tta_sd_impl(
        run_id, seed, prompt_start, prompt_end, min_inner_steps, max_inner_steps
    )


@app.function(image=fkc_image, gpu="L4", volumes={"/vol": artifacts}, timeout=7200)
def run_fkc_gmm(run_id: str) -> dict[str, object]:
    run_dir = prepare_run(run_id)
    source_notebook = Path(
        "/opt/repro/upstream/fk-correctors/applications/temperature_annealing/runner/notebooks/gmm_temp_annealed_birth_death.ipynb"
    )
    scaled_notebook = Path("/tmp/gmm_temp_annealed_systematic_scaled.ipynb")
    patch_command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/prepare_fkc_notebook.py",
        str(source_notebook),
        str(scaled_notebook),
        "--manifest",
        str(run_dir / "runtime_patch.json"),
        "--num-samples",
        "2000",
        "--num-steps",
        "200",
        "--num-runs",
        "5",
    ]
    print(f"FKC_PATCH_COMMAND_JSON={json.dumps(patch_command)}")
    subprocess.run(patch_command, check=True)
    artifacts.commit()
    command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/run_notebook.py",
        str(scaled_notebook),
        "--output-dir",
        str(run_dir),
        "--upstream-root",
        "/opt/repro/upstream/fk-correctors",
        "--upstream-commit",
        "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6",
        "--pythonpath",
        "/opt/repro/upstream/fk-correctors/applications/temperature_annealing/runner",
        "--pythonpath",
        "/opt/repro/upstream/fk-correctors/applications/temperature_annealing/fab",
        "--timeout",
        "7000",
    ]
    print(f"FKC_COMMAND_JSON={json.dumps(command)}")
    subprocess.run(command, check=True)
    return finish(
        run_id,
        run_dir,
        {
            "paper": "fk-correctors",
            "status": "COMPLETED",
            "reproduction_scope": "scaled Table A1 subset",
            "num_samples": 2000,
            "num_integration_steps": 200,
            "num_runs": 5,
            "fkc_method": "systematic",
        },
    )


@app.function(
    image=fkc_image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    volumes={"/vol": artifacts},
    timeout=5200,
)
def run_fkc_table_a1(run_id: str, mode: str = "canary") -> dict[str, object]:
    if mode not in {"canary", "full"}:
        raise ValueError(f"Unsupported FK Correctors Table A1 mode: {mode}")

    configuration = {
        "canary": {"num_samples": 1000, "num_steps": 100, "num_runs": 2},
        "full": {"num_samples": 10000, "num_steps": 1000, "num_runs": 5},
    }[mode]
    expected_method_count = 2 if mode == "canary" else 6
    run_dir = prepare_run(run_id)
    source_notebook = Path(
        "/opt/repro/upstream/fk-correctors/applications/temperature_annealing/runner/notebooks/gmm_temp_annealed_birth_death.ipynb"
    )
    runtime_notebook = Path("/tmp") / f"gmm_table_a1_{mode}.ipynb"
    patch_command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/prepare_fkc_table_a1.py",
        str(source_notebook),
        str(runtime_notebook),
        "--manifest",
        str(run_dir / "runtime_patch.json"),
        "--mode",
        mode,
    ]
    print(f"FKC_TABLE_A1_PATCH_COMMAND_JSON={json.dumps(patch_command)}")
    subprocess.run(patch_command, check=True)
    artifacts.commit()

    command = [
        "/opt/repro/.venv/bin/python",
        "/opt/repro/scripts/run_notebook.py",
        str(runtime_notebook),
        "--output-dir",
        str(run_dir),
        "--upstream-root",
        "/opt/repro/upstream/fk-correctors",
        "--upstream-commit",
        "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6",
        "--pythonpath",
        "/opt/repro/upstream/fk-correctors/applications/temperature_annealing/runner",
        "--pythonpath",
        "/opt/repro/upstream/fk-correctors/applications/temperature_annealing/fab",
        "--timeout",
        "5000",
    ]
    print(f"FKC_TABLE_A1_COMMAND_JSON={json.dumps(command)}")
    subprocess.run(command, check=True)

    metrics_path = run_dir / "fkc_table_a1_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing Table A1 metrics artifact: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("mode") != mode:
        raise RuntimeError(f"Table A1 artifact mode mismatch: {metrics.get('mode')} != {mode}")
    if len(metrics.get("methods", {})) != expected_method_count:
        raise RuntimeError(
            f"Expected {expected_method_count} Table A1 methods, found "
            f"{len(metrics.get('methods', {}))}"
        )
    if any(method["n_runs"] != configuration["num_runs"] for method in metrics["methods"].values()):
        raise RuntimeError("At least one Table A1 method has an incomplete run count")

    return finish(
        run_id,
        run_dir,
        {
            "paper": "fk-correctors",
            "status": "COMPLETED",
            "reproduction_scope": (
                "exact Table A1 six-method protocol" if mode == "full" else "BDC runtime canary"
            ),
            "mode": mode,
            **configuration,
            "method_count": expected_method_count,
            "runtime_repair": "forward reset_transition_per_index at the BDC sampler call site",
        },
    )


@app.function(
    image=fk_image,
    gpu="L4",
    volumes={"/vol": artifacts},
    timeout=7200,
    env={
        "HF_HOME": "/vol/cache/huggingface",
        "TORCH_HOME": "/vol/cache/torch",
        "XDG_CACHE_HOME": "/vol/cache/xdg",
        "MPLBACKEND": "Agg",
    },
)
def run_fk_steering(run_id: str) -> dict[str, object]:
    run_dir = prepare_run(run_id)
    upstream_workdir = Path("/opt/repro/upstream/fk-steering/text_to_image")
    workdir = Path("/tmp") / f"fk-steering-{run_id}"
    shutil.copytree(upstream_workdir, workdir)
    launch_path = workdir / "launch_eval_runs.py"
    launch_source = launch_path.read_text()
    upstream_model = "stabilityai/stable-diffusion-2-1"
    runtime_model = "sd2-community/stable-diffusion-2-1"
    if upstream_model not in launch_source:
        raise RuntimeError(f"Expected upstream model id not found: {upstream_model}")
    launch_path.write_text(launch_source.replace(upstream_model, runtime_model))
    prompt_path = Path("/opt/repro/configs/fk_steering_one_prompt.json")
    shared = [
        "/opt/repro/.venv/bin/python",
        "launch_eval_runs.py",
        "--prompt_path",
        str(prompt_path),
        "--model_idx",
        "2",
        "--num_inference_steps",
        "100",
        "--lmbda",
        "2.0",
        "--resample_frequency",
        "20",
        "--resample_t_start",
        "20",
        "--resample_t_end",
        "80",
        "--guidance_reward_fn",
        "ImageReward",
        "--metrics_to_compute",
        "ImageReward",
    ]
    commands = [
        shared + ["--use_smc", "--adaptive_resampling", "--potential_type", "max"],
        shared + ["--use_smc", "--adaptive_resampling", "--potential_type", "diff"],
        shared + ["--potential_type", "diff"],
    ]
    scratch_prompt = run_dir / "prompt.json"
    shutil.copy2(prompt_path, scratch_prompt)
    for index, command in enumerate(commands):
        command[command.index(str(prompt_path))] = str(scratch_prompt)
        print(f"FK_STEERING_COMMAND_{index}_JSON={json.dumps(command)}")
        subprocess.run(command, cwd=workdir, check=True)
    return finish(
        run_id,
        run_dir,
        {
            "paper": "fk-steering",
            "status": "COMPLETED",
            "model_upstream": upstream_model,
            "model_runtime": runtime_model,
            "model_override_reason": "The deprecated upstream Hugging Face repo returned 401; using its public community mirror.",
            "prompt_count": 1,
            "seeds": [42, 43, 44],
            "samplers": ["fk-max-k4", "fk-diff-k4", "base-k4"],
        },
    )
