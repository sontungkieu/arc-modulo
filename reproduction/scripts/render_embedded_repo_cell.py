#!/usr/bin/env python3
"""Render a deterministic, hash-checked repo archive as a Kaggle source cell."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import textwrap
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    files = sorted(path for path in repo.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"No files found under {repo}")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(repo).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    data = stream.getvalue()
    archive_sha = hashlib.sha256(data).hexdigest()
    encoded = base64.b85encode(data).decode("ascii")
    chunks = "\n".join(f"    {chunk!r}" for chunk in textwrap.wrap(encoded, width=100))
    source = f'''# Deterministic embedded repo package (generated; source files remain readable in the repo dataset manifest)
import base64
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

KJO_EMBEDDED_REPO_ARCHIVE_SHA256 = {archive_sha!r}
KJO_EMBEDDED_REPO_FILE_COUNT = {len(files)}
KJO_EMBEDDED_REPO_B85 = (\n{chunks}\n)
archive_bytes = base64.b85decode(KJO_EMBEDDED_REPO_B85.encode("ascii"))
observed_sha256 = hashlib.sha256(archive_bytes).hexdigest()
if observed_sha256 != KJO_EMBEDDED_REPO_ARCHIVE_SHA256:
    raise RuntimeError(
        f"embedded repo archive hash mismatch: expected={{KJO_EMBEDDED_REPO_ARCHIVE_SHA256}} observed={{observed_sha256}}"
    )
working_dir = Path("/kaggle/working/repo")
if working_dir.exists():
    shutil.rmtree(working_dir)
working_dir.mkdir(parents=True)
with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
    names = archive.namelist()
    if len(names) != KJO_EMBEDDED_REPO_FILE_COUNT:
        raise RuntimeError(f"embedded repo file count mismatch: expected={{KJO_EMBEDDED_REPO_FILE_COUNT}} observed={{len(names)}}")
    archive.extractall(working_dir)
os.chdir(working_dir)
print("KJO_EMBEDDED_REPO_SUMMARY " + json.dumps({{
    "archive_sha256": observed_sha256,
    "file_count": len(names),
    "working_dir": str(working_dir),
}}, sort_keys=True))
'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding="utf-8")
    print(
        f"EMBEDDED_REPO_CELL out={args.out} files={len(files)} "
        f"archive_bytes={len(data)} archive_sha256={archive_sha}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
