#!/usr/bin/env python3
"""Install the tokenizer asset omitted from the HPSv2 1.2.0 wheel.

The asset is the OpenAI CLIP BPE vocabulary distributed by Meta's MMF public
artifact host.  The pinned digest makes the image build fail closed if the
download changes; this is a packaging repair and does not alter HPSv2 scoring.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
import urllib.request
from pathlib import Path


ASSET_URL = "https://dl.fbaipublicfiles.com/mmf/clip/bpe_simple_vocab_16e6.txt.gz"
ASSET_SHA256 = "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
ASSET_NAME = "bpe_simple_vocab_16e6.txt.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    spec = importlib.util.find_spec("hpsv2")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("hpsv2 is not installed in the target environment")

    package_root = Path(next(iter(spec.submodule_search_locations)))
    target = package_root / "src" / "open_clip" / ASSET_NAME
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing_digest = sha256(target)
        if existing_digest != ASSET_SHA256:
            raise RuntimeError(
                f"Existing HPSv2 BPE asset has unexpected SHA-256: {existing_digest}"
            )
        print(f"HPSV2_BPE_READY={target} SHA256={existing_digest}")
        return 0

    request = urllib.request.Request(ASSET_URL, headers={"User-Agent": "null-tta-reproduction/1.0"})
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        with urllib.request.urlopen(request, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)

    try:
        downloaded_digest = sha256(temporary)
        if downloaded_digest != ASSET_SHA256:
            raise RuntimeError(
                f"Downloaded HPSv2 BPE asset has unexpected SHA-256: {downloaded_digest}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"HPSV2_BPE_READY={target} SHA256={downloaded_digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
