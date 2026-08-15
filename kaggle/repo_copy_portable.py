"""Copy the attached source bundle across both Kaggle input layouts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


KJO_REPO_DATASET_SOURCE = "codemaivanngu/diffusion-smc-das-uv-src-20260811"
KJO_REPO_DATASET_SLUG = "diffusion-smc-das-uv-src-20260811"
KJO_REPO_DIR_NAME = "repo"
KJO_REPO_WORKING_DIR = Path("/kaggle/working/repo")

owner, slug = KJO_REPO_DATASET_SOURCE.split("/", 1)
input_root = Path(os.environ.get("KJO_KAGGLE_INPUT_ROOT", "/kaggle/input"))
dataset_candidates = [
    input_root / slug,
    input_root / "datasets" / owner / slug,
]
dataset_root = next((candidate for candidate in dataset_candidates if candidate.is_dir()), None)
source_repo = None
if dataset_root is not None:
    nested_repo = dataset_root / KJO_REPO_DIR_NAME
    if nested_repo.is_dir():
        source_repo = nested_repo
    elif (dataset_root / "pyproject.toml").is_file() and (dataset_root / "uv.lock").is_file():
        # Kaggle expands the repo.zip payload at the dataset root for private
        # datasets mounted through /kaggle/input/datasets/<owner>/<slug>.
        source_repo = dataset_root
if source_repo is None:
    available = sorted(str(path.relative_to(input_root)) for path in input_root.glob("**/*") if path.is_dir())
    raise FileNotFoundError(
        "Kaggle repo dataset is not mounted in a supported layout. "
        f"dataset_source={KJO_REPO_DATASET_SOURCE} "
        f"searched={[str(path) for path in dataset_candidates]} available={available[:100]}"
    )

if KJO_REPO_WORKING_DIR.exists():
    shutil.rmtree(KJO_REPO_WORKING_DIR)
shutil.copytree(source_repo, KJO_REPO_WORKING_DIR, symlinks=False)
os.chdir(KJO_REPO_WORKING_DIR)

summary = {
    "dataset_source": KJO_REPO_DATASET_SOURCE,
    "dataset_root": str(dataset_root),
    "source_repo": str(source_repo),
    "working_dir": str(KJO_REPO_WORKING_DIR),
    "file_count": sum(1 for path in KJO_REPO_WORKING_DIR.rglob("*") if path.is_file()),
}
print("KJO_REPO_DATASET_COPY_SUMMARY " + json.dumps(summary, sort_keys=True))
