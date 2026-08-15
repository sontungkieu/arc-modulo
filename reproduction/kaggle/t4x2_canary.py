"""Kaggle T4 x2 and frozen-uv environment canary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REPO_ROOT = Path("/kaggle/working/repo")
KJO_ROOT = Path("/kaggle/working/kaggle_job_ops")
run_candidates = sorted(
    (path for path in KJO_ROOT.iterdir() if path.is_dir()),
    key=lambda path: path.stat().st_mtime,
) if KJO_ROOT.is_dir() else []
RUN_ID = os.environ.get("DIFFUSION_SMC_RUN_ID") or (run_candidates[-1].name if run_candidates else "t4x2-canary")
DIAGNOSTICS = Path("/kaggle/working/kaggle_job_ops") / RUN_ID / "diagnostics"
UV_VERSION = "0.10.2"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, timeout: int = 1200) -> dict:
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
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    print("T4X2_COMMAND_SUMMARY", json.dumps(result, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with return code {completed.returncode}: {command}")
    return result


if not REPO_ROOT.is_dir():
    raise FileNotFoundError(f"repo dataset was not copied to {REPO_ROOT}")

lock_path = REPO_ROOT / "uv.lock"
project_path = REPO_ROOT / "pyproject.toml"
for required in (lock_path, project_path):
    if not required.is_file():
        raise FileNotFoundError(required)

uv_bin = shutil.which("uv")
install_result = None
if uv_bin is None:
    install_result = run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", f"uv=={UV_VERSION}"],
        timeout=300,
    )
    uv_bin = shutil.which("uv")
if uv_bin is None:
    candidate = Path(sys.executable).parent / "uv"
    uv_bin = str(candidate) if candidate.is_file() else None
if uv_bin is None:
    raise RuntimeError("uv executable not found after installation")

portable_env = REPO_ROOT / ".venv"
uv_cache = Path("/kaggle/working/.uv-cache")
env = os.environ.copy()
env.update(
    {
        "UV_PROJECT_ENVIRONMENT": str(portable_env),
        "UV_CACHE_DIR": str(uv_cache),
        "UV_LINK_MODE": "copy",
        "UV_PYTHON_DOWNLOADS": "automatic",
    }
)
uv_version_result = run([uv_bin, "--version"], cwd=REPO_ROOT, env=env, timeout=60)
sync_result = run(
    [uv_bin, "sync", "--frozen", "--no-install-project"],
    cwd=REPO_ROOT,
    env=env,
    timeout=1200,
)

venv_python = portable_env / "bin" / "python"
if not venv_python.is_file():
    raise FileNotFoundError(venv_python)

probe_code = r"""
import json
import platform
import torch
import numpy
import scipy

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(count)]
if not torch.cuda.is_available():
    raise RuntimeError("locked environment has no CUDA")
if count < 2:
    raise RuntimeError(f"expected at least two GPUs, saw {count}")
if not all("T4" in name.upper() for name in names[:2]):
    raise RuntimeError(f"expected two T4 GPUs, saw {names}")

checks = []
for index in range(2):
    device = torch.device(f"cuda:{index}")
    left = torch.arange(256 * 256, device=device, dtype=torch.float32).reshape(256, 256)
    right = torch.eye(256, device=device)
    value = float((left @ right).sum().item())
    checks.append({"device": str(device), "sum": value})

payload = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "cuda_available": torch.cuda.is_available(),
    "gpu_count": count,
    "gpu_names": names,
    "per_device_checks": checks,
}
print("T4X2_LOCKED_ENV_SUMMARY", json.dumps(payload, sort_keys=True))
"""
probe_result = run([str(venv_python), "-c", probe_code], cwd=REPO_ROOT, env=env, timeout=300)
summary_line = next(
    line for line in probe_result["stdout_tail"].splitlines() if line.startswith("T4X2_LOCKED_ENV_SUMMARY ")
)
locked_environment = json.loads(summary_line.split(" ", 1)[1])

summary = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "status": "PASS",
    "repo_root": str(REPO_ROOT),
    "uv_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
    "uv_version": uv_version_result["stdout_tail"].strip(),
    "uv_install_performed": install_result is not None,
    "uv_sync_elapsed_s": sync_result["elapsed_s"],
    "locked_environment": locked_environment,
}
DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
summary_path = DIAGNOSTICS / "t4x2_canary_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("T4X2_CANARY_SUMMARY", json.dumps(summary, sort_keys=True))
