"""Run the DAS SMC rescue at matched Python/Torch and Numba seed 42 on T4x2."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO_ROOT = Path("/kaggle/working/repo")
KJO_RUN_ID = "das-seed42-numba42-rescue-t4x2-r6-20260811T201300Z"
DIAGNOSTICS = Path("/kaggle/working/kaggle_job_ops") / KJO_RUN_ID / "diagnostics"
UV_VERSION = "0.10.2"
UPSTREAM_COMMIT = "2f4b2239f29ee59f80359bdfa5ed747b6d855a1e"


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: int = 7200,
) -> dict[str, object]:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    result = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_s": time.monotonic() - started,
        "stdout_tail": completed.stdout[-6000:],
        "stderr_tail": completed.stderr[-6000:],
    }
    print("DAS_KAGGLE_COMMAND_SUMMARY " + json.dumps(result, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {command}")
    return result


for required in (
    REPO_ROOT / "pyproject.toml",
    REPO_ROOT / "uv.lock",
    REPO_ROOT / "scripts/prepare_das_notebook.py",
    REPO_ROOT / "scripts/run_notebook.py",
    REPO_ROOT / "upstream/das/notebooks/GMM.ipynb",
):
    if not required.is_file():
        raise FileNotFoundError(required)

# Match the Modal tool version rather than silently accepting Kaggle's preinstalled uv.
uv_bin = shutil.which("uv")
uv_version = ""
if uv_bin:
    uv_version = subprocess.run(
        [uv_bin, "--version"], check=False, capture_output=True, text=True
    ).stdout.strip()
if uv_version != f"uv {UV_VERSION} (x86_64-unknown-linux-gnu)":
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--upgrade",
            f"uv=={UV_VERSION}",
        ],
        timeout=300,
    )
    uv_bin = shutil.which("uv")
if not uv_bin:
    raise RuntimeError("uv executable is unavailable")

portable_env = Path("/tmp/diffusion-smc-repro-venv")
uv_cache = Path("/tmp/diffusion-smc-uv-cache")
env = os.environ.copy()
env.update(
    {
        # PyTorch requires this to be present before the notebook kernel starts
        # when deterministic CuBLAS algorithms are enabled.
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "UV_PROJECT_ENVIRONMENT": str(portable_env),
        "UV_CACHE_DIR": str(uv_cache),
        "UV_LINK_MODE": "copy",
        "UV_PYTHON_DOWNLOADS": "automatic",
        "MPLBACKEND": "Agg",
    }
)
uv_version_result = run([uv_bin, "--version"], cwd=REPO_ROOT, env=env, timeout=60)
sync_result = run(
    [uv_bin, "sync", "--frozen", "--no-install-project"],
    cwd=REPO_ROOT,
    env=env,
    timeout=1200,
)

venv_python = portable_env / "bin/python"
if not venv_python.is_file():
    raise FileNotFoundError(venv_python)

DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
runtime_notebook = Path("/tmp/GMM.seed-42.numba-42.smc-rescue.ipynb")
patch_result = run(
    [
        str(venv_python),
        str(REPO_ROOT / "scripts/prepare_das_notebook.py"),
        str(REPO_ROOT / "upstream/das/notebooks/GMM.ipynb"),
        str(runtime_notebook),
        "--seed",
        "42",
        "--numba-seed",
        "42",
        "--smc-only",
        "--manifest",
        str(DIAGNOSTICS / "runtime_patch.json"),
    ],
    cwd=REPO_ROOT,
    env=env,
    timeout=300,
)

notebook_output = DIAGNOSTICS / "notebook_output"
notebook_result = run(
    [
        str(venv_python),
        str(REPO_ROOT / "scripts/run_notebook.py"),
        str(runtime_notebook),
        "--output-dir",
        str(notebook_output),
        "--upstream-root",
        str(REPO_ROOT / "upstream/das"),
        "--upstream-commit",
        UPSTREAM_COMMIT,
        "--pythonpath",
        str(REPO_ROOT / "upstream/das"),
        "--timeout",
        "7000",
    ],
    cwd=REPO_ROOT,
    env=env,
    timeout=7100,
)

metrics_path = notebook_output / "das_metrics.json"
manifest_path = notebook_output / "run_manifest.json"
if not metrics_path.is_file() or not manifest_path.is_file():
    raise FileNotFoundError(f"missing DAS outputs: metrics={metrics_path} manifest={manifest_path}")
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if metrics.get("numba_seed") != 42 or metrics.get("seed") != 42:
    raise RuntimeError("DAS rescue must persist matching Python/Torch and Numba seed 42")
smc = metrics["smc_das"]
target = metrics["target"]

environment_probe = run(
    [
        str(venv_python),
        "-c",
        (
            "import json,platform,torch; "
            "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
            "'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
            "'gpu_count':torch.cuda.device_count(),'gpu_names':[torch.cuda.get_device_name(i) "
            "for i in range(torch.cuda.device_count())]}))"
        ),
    ],
    cwd=REPO_ROOT,
    env=env,
    timeout=120,
)
environment_summary = json.loads(str(environment_probe["stdout_tail"]).splitlines()[-1])

summary = {
    "schema_version": 1,
    "status": "PASS",
    "paper": "DAS",
    "experiment": "official GMM pretrained plus SMC cells with explicit Numba RNG seed",
    "seed": 42,
    "numba_seed": 42,
    "upstream_commit": UPSTREAM_COMMIT,
    "execution_note": "Official notebook uses one CUDA device; the Kaggle allocation exposes two T4 GPUs.",
    "uv_version": str(uv_version_result["stdout_tail"]).strip(),
    "uv_sync_elapsed_s": sync_result["elapsed_s"],
    "environment": environment_summary,
    "run_manifest": manifest,
    "primary_metrics": {
        "target_reward_mean": target["reward"]["mean"],
        "smc_reward_mean": smc["reward"]["mean"],
        "smc_reward_gap_abs_to_target": smc["reward_gap_abs_to_target"],
        "paper_style_smc_emd": metrics["paper_style_smc_emd"],
        "smc_emd_mean": smc["emd_mean"],
        "smc_emd_std": smc["emd_std"],
        "smc_target_mode_coverage": smc["target_mode_coverage"],
        "target_mode_count": smc["target_mode_count"],
    },
    "step_elapsed_s": {
        "prepare_notebook": patch_result["elapsed_s"],
        "execute_notebook": notebook_result["elapsed_s"],
    },
}
(DIAGNOSTICS / "das_kaggle_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("DAS_KAGGLE_SUMMARY " + json.dumps(summary, sort_keys=True))
