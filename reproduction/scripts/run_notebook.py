#!/usr/bin/env python3
"""Execute an upstream notebook copy and record a machine-readable run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--upstream-commit")
    parser.add_argument("--pythonpath", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.notebook.resolve()
    output_dir = args.output_dir.resolve()
    upstream_root = args.upstream_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)

    copied = output_dir / source.name
    executed = output_dir / f"{source.stem}.executed.ipynb"
    manifest_path = output_dir / "run_manifest.json"
    shutil.copy2(source, copied)

    pythonpath = [str(Path(value).resolve()) for value in args.pythonpath]
    prior_pythonpath = os.environ.get("PYTHONPATH")
    if prior_pythonpath:
        pythonpath.append(prior_pythonpath)
    if pythonpath:
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    os.environ.setdefault("MPLBACKEND", "Agg")

    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    status = "RUNNING"
    error: str | None = None

    manifest = {
        "schema_version": 1,
        "status": status,
        "source_notebook": str(source),
        "source_sha256": sha256(source),
        "copied_notebook": str(copied),
        "executed_notebook": str(executed),
        "upstream_root": str(upstream_root),
        "upstream_commit": args.upstream_commit or git_commit(upstream_root),
        "started_at_utc": started.isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "pythonpath": pythonpath,
        "timeout_s": args.timeout,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        notebook = nbformat.read(copied, as_version=4)
        client = NotebookClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(output_dir)}},
            allow_errors=False,
        )
        client.execute()
        nbformat.write(notebook, executed)
        status = "COMPLETED"
    except Exception as exc:  # preserve the partial notebook and exact failure
        status = "FAILED"
        error = f"{type(exc).__name__}: {exc}"
        try:
            nbformat.write(notebook, executed)
        except Exception:
            pass
    finally:
        finished = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": status,
                "finished_at_utc": finished.isoformat(),
                "elapsed_s": round(time.monotonic() - started_monotonic, 6),
                "error": error,
                "executed_sha256": sha256(executed) if executed.exists() else None,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"RUN_MANIFEST_JSON={json.dumps(manifest, sort_keys=True)}")
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
