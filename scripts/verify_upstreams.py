#!/usr/bin/env python3
"""Verify that cloned upstream repositories match UPSTREAMS.toml."""

from __future__ import annotations

import json
import subprocess
import sys
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 in the frozen portable environment.
    import tomli as tomllib
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    manifest = tomllib.loads((root / "UPSTREAMS.toml").read_text(encoding="utf-8"))
    results = []
    ok = True
    for item in manifest["repository"]:
        repo = root / item["path"]
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
        )
        actual = commit.stdout.strip() if commit.returncode == 0 else None
        entry_ok = actual == item["commit"] and dirty.returncode == 0 and not dirty.stdout.strip()
        ok &= entry_ok
        results.append(
            {
                "id": item["id"],
                "path": str(repo),
                "expected_commit": item["commit"],
                "actual_commit": actual,
                "clean": dirty.returncode == 0 and not dirty.stdout.strip(),
                "ok": entry_ok,
            }
        )
    payload = {"ok": ok, "repositories": results}
    print(json.dumps(payload, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
