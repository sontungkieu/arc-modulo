# Diffusion SMC reproduction suite

This suite keeps four official repositories unmodified and adds a portable
`uv`-locked execution layer around them:

- Null-TTA: `upstream/null-tta`
- DAS: `upstream/das`
- Feynman--Kac Steering: `upstream/fk-steering`
- Feynman--Kac Correctors: `upstream/fk-correctors`

Exact upstream URLs and commits are recorded in `UPSTREAMS.toml`.

## Portable environment

Python 3.10 and every Python dependency are managed by `uv`. The Windows/WSL
workstation keeps all three physical environments and the shared cache under
`D:\dev\codex`; the `.venv` entries in the C:-hosted source tree are only
symlinks.  This prevents a plain `uv run` from silently recreating a heavy
environment on C:.

The three locked environments are:

- toy DAS/FK Correctors/report tooling: `/mnt/d/dev/codex/diffusion_smc_repro/toy-venv`
- Null-TTA: `/mnt/d/dev/codex/diffusion_smc_repro/null-tta-venv`
- FK Steering: `/mnt/d/dev/codex/diffusion_smc_repro/fk-steering-venv`

Create or refresh them explicitly:

```bash
export UV_PROJECT_ENVIRONMENT=/mnt/d/dev/codex/diffusion_smc_repro/toy-venv
export UV_CACHE_DIR=/mnt/d/dev/codex/diffusion_smc_repro/uv-cache
uv sync --frozen
uv run python scripts/verify_upstreams.py

cd envs/null-tta
UV_PROJECT_ENVIRONMENT=/mnt/d/dev/codex/diffusion_smc_repro/null-tta-venv \
UV_CACHE_DIR=/mnt/d/dev/codex/diffusion_smc_repro/uv-cache \
uv sync --frozen --no-install-project

cd ../fk-steering
UV_PROJECT_ENVIRONMENT=/mnt/d/dev/codex/diffusion_smc_repro/fk-steering-venv \
UV_CACHE_DIR=/mnt/d/dev/codex/diffusion_smc_repro/uv-cache \
uv sync --frozen --no-install-project
```

The same lockfile is synced independently on Talapas, Modal, or Kaggle; a
virtual environment itself is never copied between machines. Current runs use
`scripts/modal_runner.py`, whose Modal images execute `uv sync --frozen` inside
each remote workspace. Kaggle source cells under `kaggle/` unpack a
deterministic, hash-checked source archive into `/kaggle/working/repo` and sync
the same frozen lockfile into ephemeral remote storage. This avoids relying on
a second dataset mount that Kaggle may omit.

## Official notebook runners

The runner copies an official notebook into a run directory, executes the copy,
and writes `run_manifest.json`. Upstream worktrees remain clean. Scientific
commands are launched by the Modal/Kaggle wrappers; the examples below describe
commands inside a remote image, not local GPU experiments.

DAS Figure 1, first 2-D GMM reward:

```bash
uv run python scripts/run_notebook.py \
  upstream/das/notebooks/GMM.ipynb \
  --output-dir runs/das_gmm_seed42 \
  --upstream-root upstream/das \
  --pythonpath upstream/das \
  --timeout 7200
```

FK Correctors Table A1, 40-Gaussian annealing:

```bash
uv run python scripts/run_notebook.py \
  upstream/fk-correctors/applications/temperature_annealing/runner/notebooks/gmm_temp_annealed_birth_death.ipynb \
  --output-dir runs/fkc_gmm_table_a1 \
  --upstream-root upstream/fk-correctors \
  --pythonpath upstream/fk-correctors/applications/temperature_annealing/runner \
  --pythonpath upstream/fk-correctors/applications/temperature_annealing/fab \
  --timeout 7200
```

The FK Steering image experiment uses `envs/fk-steering/uv.lock` because its
official Diffusers and reward-model pins conflict with the toy environment.

Null-TTA has a separate frozen environment. Its 24 GB L4 path keeps PickScore
resident during generation and evaluates the official saved uint8 PNG pairs
sequentially with the other three official scorers. PNG is lossless after the
example's tensor-to-uint8 conversion; that quantization caveat applies to the
three secondary metrics, not the direct PickScore primary:

```bash
MODAL_PROFILE=your-modal-profile NULL_TTA_MODAL_GPU=L4 \
  uvx --from modal modal run scripts/modal_null_tta.py::run_null_tta \
  --run-id unique-run-id --seed 44 --prompt-start 0 --prompt-end 25 \
  --min-inner-steps 5 --max-inner-steps 55 --sequential-scorers
```

## Audited remote reproductions

The 2026-08-11 runs used three distinct Modal workspaces in parallel; the
multi-seed DAS batch was spread across all three after the initial one-paper-per-
workspace runs. No scientific experiment was executed locally. See `REPORT.md` and the structured
JSON files under `results/` for exact scope, metrics, provenance, and billing.
The canonical audited manuscript is
`report/diffusion_smc_reproduction_report.tex`; its verified PDF is
`output/pdf/diffusion_smc_reproduction_report.pdf`. DAS primary evidence uses
matched Python/Torch/Numba seeds 42, 43, and 44. The older five-seed batch is
forensic-only because the Numba resampling RNG was not seeded. Null-TTA
aggregation fails closed unless all 150 seed--prompt records and all four real
scorers are present. Primary PASS additionally requires both a PickScore point
match within 10% symmetric relative difference and a strictly positive 95%
hierarchical-bootstrap CI for paired improvement over the reproduced baseline.
`scripts/validate_null_tta_artifacts.py` additionally verifies the exact
25-artifact remote matrix, observed/requested GPU, scoring path, hashes, config,
and non-overlapping prompt coverage before the report can build.
The guarded final seed-44 matrix contains seven complete L4 artifacts, one
hash-validated four-prompt prefix generated on L4 and post-scored on L40S, and
one three-prompt L40S replacement. All use target-only generation plus
sequential post-scoring. The unfinished suffix and the provider-disabled retry
are excluded.
`scripts/build_modal_billing_final.py` joins provider billing rows to the exact
26 scientific app IDs behind the 25 artifact records through the local run
ledger, rather than estimating cost from wall time, and archives the exact raw
provider reports under `remote_artifacts/modal_billing/`.

The Kaggle T4x2 portability extension uses two run pages (owner-authenticated
while the kernels remain private):

- [T4x2 canary](https://www.kaggle.com/code/codemaivanngu/diffusion-smc-t4x2-canary-r4-20260811)
- [DAS GMM seed 42 matched-RNG rescue](https://www.kaggle.com/code/codemaivanngu/diffusion-smc-das-seed42-numba-rescue-t4x2-r6)

The canary verifies work on both visible T4 devices. The official DAS notebook
itself uses one CUDA device, so the scientific run demonstrates deployment on a
T4x2 allocation rather than multi-GPU scaling. Its matched-RNG metrics are
summarized in `results/das_kaggle_t4x2_rescue.json`. This duplicate seed is
intentionally excluded from the three-seed Modal aggregate.

FK Correctors now has both the earlier scaled diagnostic and an exact Table A1
runner. Exact mode is locked to 10,000 samples, 1,000 integration steps, 5
sequential stochastic runs, and all six no-FKC/BDC/systematic methods.
`scripts/prepare_fkc_table_a1.py` records source/runtime hashes and the one-site
BDC flag-forwarding repair; `scripts/modal_fkc_table_a1.py` is an isolated
L4-only Modal entrypoint. The official upstream notebook is never modified.
Machine-readable cell comparisons, raw run rows, variance diagnostics, remote
identity, VRAM, runtime, and provider cost are in
`results/fk_correctors_table_a1_exact.json`.

Example remote invocation:

```bash
export UV_CACHE_DIR=/path/to/large-disk/uv-cache
export MODAL_PROFILE=your-modal-profile
uvx --from modal==1.5.3 modal run \
  scripts/modal_fkc_table_a1.py::run_table_a1 \
  --run-id fkc-table-a1-unique-run-id \
  --mode full
```

Before a new run, use the Modal billing guard described by the local
`modal-gpu-ops` skill. Never reuse a run ID: the runner refuses to overwrite an
existing remote artifact directory.
