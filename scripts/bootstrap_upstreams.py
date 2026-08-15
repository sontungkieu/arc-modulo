#!/usr/bin/env python3
"""Materialize the four pinned paper sources as linked Git worktrees."""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


PAPERS = (
    ("null-tta", "paper/null-tta", "337bf73037e9f24e9f844974d3287384abb610bf"),
    ("das", "paper/das", "2f4b2239f29ee59f80359bdfa5ed747b6d855a1e"),
    ("fk-steering", "paper/fk-steering", "9413005dee1e79f80fb4561a4a7ac8eec704281b"),
    ("fk-correctors", "paper/fk-correctors", "aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6"),
)


def run(*args: str, cwd: Path) -> str:
    completed = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout.strip()


def fetch_branch(root: Path, branch: str, retries: int) -> None:
    """Fetch one paper history with bounded retries and HTTP/1.1 stability."""
    refspec = f"refs/heads/{branch}:refs/remotes/origin/{branch}"
    command = (
        "git",
        "-c",
        "http.version=HTTP/1.1",
        "fetch",
        "--no-tags",
        "origin",
        refspec,
    )
    for attempt in range(1, retries + 1):
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode == 0:
            return
        if attempt == retries:
            raise SystemExit(
                f"Fetch failed for {branch} after {retries} attempts "
                f"(last exit code {completed.returncode})"
            )
        delay = min(5 * attempt, 15)
        print(
            f"RETRY {branch}: attempt {attempt}/{retries} failed; "
            f"waiting {delay}s",
            flush=True,
        )
        time.sleep(delay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-fk-steering-submodule",
        action="store_true",
        help="also initialize the optional discrete_diffusion/mdlm submodule",
    )
    parser.add_argument(
        "--fetch-retries",
        type=int,
        default=3,
        help="bounded retries per paper branch (default: 3)",
    )
    args = parser.parse_args()
    if args.fetch_retries < 1:
        parser.error("--fetch-retries must be at least 1")

    root = Path(__file__).resolve().parents[1]
    git_root = Path(run("git", "rev-parse", "--show-toplevel", cwd=root))
    if git_root != root:
        raise SystemExit(f"Run from a fresh main worktree; expected {root}, got {git_root}")

    upstream = root / "upstream"
    upstream.mkdir(exist_ok=True)
    for paper, branch, commit in PAPERS:
        fetch_branch(root, branch, args.fetch_retries)
        destination = upstream / paper
        if destination.exists() and any(destination.iterdir()):
            actual = run("git", "rev-parse", "HEAD", cwd=destination)
            if actual != commit:
                raise SystemExit(
                    f"{destination} exists at {actual}, expected {commit}; not overwriting"
                )
            print(f"OK existing {paper}: {actual}")
            continue

        run(
            "git",
            "merge-base",
            "--is-ancestor",
            commit,
            f"origin/{branch}",
            cwd=root,
        )
        run("git", "worktree", "add", "--detach", str(destination), commit, cwd=root)
        print(f"CREATED {paper}: {commit}")

    if args.with_fk_steering_submodule:
        run(
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
            cwd=upstream / "fk-steering",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
