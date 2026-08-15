#!/usr/bin/env python3
"""Isolated Modal runner for the FK Correctors Table A1 reproduction.

Keeping this entrypoint separate prevents Modal from validating unrelated L40S
functions declared by the multi-paper runner on workspaces without L40S access.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/opt/repro")
UV_VERSION = "0.10.2"
UPSTREAM_COMMIT = "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6"

app = modal.App("fk-correctors-table-a1-reproduction")
artifacts = modal.Volume.from_name("diffusion-smc-repro-artifacts", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .uv_pip_install(f"uv=={UV_VERSION}", uv_version=UV_VERSION)
    .add_local_file(ROOT / "pyproject.toml", str(REMOTE_ROOT / "pyproject.toml"), copy=True)
    .add_local_file(ROOT / "uv.lock", str(REMOTE_ROOT / "uv.lock"), copy=True)
    .run_commands(
        "cd /opt/repro && UV_PROJECT_ENVIRONMENT=/opt/repro/.venv uv sync --frozen --no-install-project",
    )
    .add_local_file(
        ROOT / "scripts/prepare_fkc_table_a1.py",
        str(REMOTE_ROOT / "scripts/prepare_fkc_table_a1.py"),
        copy=True,
    )
    .add_local_file(
        ROOT / "scripts/run_notebook.py",
        str(REMOTE_ROOT / "scripts/run_notebook.py"),
        copy=True,
    )
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


@app.function(
    image=image,
    gpu="L4",
    cpu=4.0,
    memory=32768,
    volumes={"/vol": artifacts},
    timeout=5200,
)
def run_table_a1(run_id: str, mode: str = "full") -> dict[str, object]:
    if mode not in {"canary", "full"}:
        raise ValueError(f"Unsupported mode: {mode}")
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
        UPSTREAM_COMMIT,
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
        raise FileNotFoundError(f"Missing metrics artifact: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    if metrics.get("mode") != mode or len(metrics.get("methods", {})) != expected_method_count:
        raise RuntimeError("Table A1 artifact has the wrong mode or method count")
    if any(method["n_runs"] != configuration["num_runs"] for method in metrics["methods"].values()):
        raise RuntimeError("At least one Table A1 method has an incomplete run count")

    payload = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_payload(),
        "paper": "fk-correctors",
        "status": "COMPLETED",
        "reproduction_scope": (
            "exact Table A1 six-method protocol" if mode == "full" else "BDC runtime canary"
        ),
        "mode": mode,
        **configuration,
        "method_count": expected_method_count,
        "runtime_repair": "forward reset_transition_per_index at the BDC sampler call site",
        "runner_isolation": "L4-only entrypoint; no unrelated GPU function declarations",
    }
    (run_dir / "modal_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    artifacts.commit()
    print(f"DIFFUSION_SMC_ENV_JSON={json.dumps(payload['environment'], sort_keys=True)}")
    print(f"DIFFUSION_SMC_SUMMARY_JSON={json.dumps(payload, sort_keys=True)}")
    return payload
