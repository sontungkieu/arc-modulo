# Kaggle T4 x2 portability

This directory contains readable notebook source cells for Kaggle deployment.
Scientific dependencies remain defined by the project-level `pyproject.toml`
and `uv.lock`. Kaggle Job Ops stages instrumented notebook copies. The current
rescue embeds a deterministic, SHA-256-checked archive of the required repo
files because the backend omitted a second attached dataset even when it was
present in submitted metadata.

The first gate is t4x2_canary.py. It requires the Kaggle shape
NvidiaTeslaT4, synchronizes the frozen uv environment in Kaggle working
storage, imports the locked scientific stack, and performs explicit work on
both visible T4 devices.

After the canary passes, `das_seed42.py` executes the official DAS GMM notebook
with matched Python/Torch and Numba seed 42 and writes metrics, the executed
notebook, runtime patch, and environment summary into KJO diagnostics.
`CUBLAS_WORKSPACE_CONFIG` is set before the Papermill child kernel starts so
strict deterministic CUDA execution is valid. The upstream notebook selects
one CUDA device; T4x2 therefore describes the verified allocation, not
scientific multi-GPU utilization.
