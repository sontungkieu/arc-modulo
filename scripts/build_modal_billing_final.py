#!/usr/bin/env python3
"""Join provider billing rows to the exact canonical Null-TTA Modal app ids."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


EXPECTED_PROFILES = {"kieusontung6", "kieusontung8", "phamvanvuhoan"}
EXPECTED_CANONICAL_APPS = {"kieusontung6": 10, "kieusontung8": 8, "phamvanvuhoan": 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument(
        "--snapshot",
        action="append",
        required=True,
        help="PROFILE=/absolute/path/to/raw_billing_report.json (repeat three times)",
    )
    parser.add_argument(
        "--archive-dir",
        required=True,
        type=Path,
        help="Repository-owned directory in which to preserve the exact raw provider reports",
    )
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_snapshot_arg(value: str) -> tuple[str, Path]:
    profile, separator, raw_path = value.partition("=")
    if not separator or not profile or not raw_path:
        raise ValueError(f"Invalid --snapshot value: {value!r}")
    return profile, Path(raw_path).resolve()


def report_timestamp(path: Path) -> str:
    stamp = path.stem.rsplit("_", 1)[-1]
    parsed = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return parsed.isoformat()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run_ids = {str(row["run_id"]) for row in manifest["runs"]}
    if len(run_ids) != 25:
        raise RuntimeError("Billing join requires the validated 25-artifact Null-TTA manifest")

    snapshots = dict(parse_snapshot_arg(value) for value in args.snapshot)
    if set(snapshots) != EXPECTED_PROFILES:
        raise RuntimeError(f"Expected snapshots for exactly {sorted(EXPECTED_PROFILES)}")

    ledger_rows = [json.loads(line) for line in args.ledger.read_text(encoding="utf-8").splitlines() if line]
    submitted: dict[str, dict[str, Any]] = {}
    completed: set[str] = set()
    for row in ledger_rows:
        run_id = row.get("run_id")
        if run_id not in run_ids:
            continue
        if row.get("status") == "submitted" and row.get("app_id"):
            submitted[str(run_id)] = row
        if row.get("status") in {"completed", "succeeded"}:
            completed.add(str(run_id))
    if set(submitted) != run_ids or completed != run_ids:
        raise RuntimeError("Every canonical run must have submitted app-id and completed ledger evidence")

    by_profile: dict[str, set[str]] = {profile: set() for profile in EXPECTED_PROFILES}
    for run_id, row in submitted.items():
        profile = str(row["profile"])
        if profile not in by_profile:
            raise RuntimeError(f"Unexpected Modal profile for {run_id}: {profile}")
        by_profile[profile].add(str(row["app_id"]))
    recovery_sources = [row.get("recovery_source") for row in manifest["runs"] if row.get("recovery_source")]
    if len(recovery_sources) != 1:
        raise RuntimeError("Expected exactly one recovered source-generation app")
    source = recovery_sources[0]
    source_profile = str(source["source_profile"])
    source_app_id = str(source["source_app_id"])
    if source_profile not in by_profile:
        raise RuntimeError(f"Unexpected recovery source profile: {source_profile}")
    by_profile[source_profile].add(source_app_id)
    if any(len(by_profile[profile]) != count for profile, count in EXPECTED_CANONICAL_APPS.items()):
        raise RuntimeError(f"Unexpected canonical app-id split: {by_profile}")

    output_snapshots: dict[str, Any] = {}
    for profile, path in snapshots.items():
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise RuntimeError(f"Raw Modal billing report is not a row list: {path}")
        total = sum((Decimal(str(row["cost"])) for row in rows), Decimal("0"))
        billed_app_ids = {
            str(row.get("object_id")) for row in rows if str(row.get("object_id")) in by_profile[profile]
        }
        if billed_app_ids != by_profile[profile]:
            missing = sorted(by_profile[profile] - billed_app_ids)
            raise RuntimeError(f"Provider billing snapshot has not exposed all canonical apps for {profile}: {missing}")
        canonical = sum(
            (Decimal(str(row["cost"])) for row in rows if str(row.get("object_id")) in by_profile[profile]),
            Decimal("0"),
        )
        null_total = sum(
            (
                Decimal(str(row["cost"]))
                for row in rows
                if str(row.get("description", "")).startswith("null-tta-reproduction")
            ),
            Decimal("0"),
        )
        archived_path = args.archive_dir / profile / path.name
        archived_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, archived_path)
        if sha256(archived_path) != sha256(path):
            raise RuntimeError(f"Archived billing snapshot hash mismatch for {profile}")
        output_snapshots[profile] = {
            "final_reported_usage": float(total),
            "canonical_run_cost": float(canonical),
            "all_null_tta_attempts_cost": float(null_total),
            "canonical_app_count": len(by_profile[profile]),
            "provider_billed_canonical_app_count": len(billed_app_ids),
            "canonical_app_ids": sorted(by_profile[profile]),
            "reported_at_utc": report_timestamp(path),
            "raw_report_path": str(archived_path),
            "raw_report_sha256": sha256(archived_path),
        }

    hard_limit = 29.0
    if any(row["final_reported_usage"] >= hard_limit for row in output_snapshots.values()):
        raise RuntimeError(f"Modal hard guard exceeded: {output_snapshots}")
    output = {
        "schema_version": 2,
        "billing_period": ["2026-08-01", "2026-09-01"],
        "currency": "USD",
        "guard": {
            "budget_per_workspace": 30.0,
            "reserve_per_workspace": 1.0,
            "hard_limit_per_workspace": hard_limit,
        },
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "ledger_path": str(args.ledger.resolve()),
        "ledger_sha256": sha256(args.ledger),
        "snapshots": output_snapshots,
        "all_under_hard_limit": True,
        "qualification": (
            "Provider-reported billing may lag. canonical_run_cost joins rows by the exact 26 scientific "
            "app ids behind 25 canonical artifact records, including the canceled source-generation app "
            "whose four committed prompt pairs were hash-validated and recovered. all_null_tta_attempts_cost "
            "also includes canaries and excluded canceled/failed attempts."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(args.out), "profiles": sorted(output_snapshots)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
