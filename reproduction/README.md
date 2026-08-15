# Feynman--Kac Correctors reproduction overlay

Branch này giữ nguyên lịch sử upstream tới commit
`aa6f5ed4a0ebb91329d4cd5823cc7e77c5e196e6`. Reproduction exact Table A1 dùng
10,000 samples, 1,000 integration steps, 5 sequential stochastic runs và đủ 6
methods.

```bash
uv sync --project reproduction --frozen --no-install-project
uv run --project reproduction python reproduction/scripts/run_notebook.py \
  applications/temperature_annealing/runner/notebooks/gmm_temp_annealed_birth_death.ipynb \
  --output-dir runs/fkc_table_a1 \
  --upstream-root . \
  --pythonpath applications/temperature_annealing/runner \
  --pythonpath applications/temperature_annealing/fab \
  --timeout 7200
```

Runner exact Modal và patch generator nằm trong `scripts/`; dùng branch `main`
để giữ nguyên layout nhiều-paper của wrapper. `results/` chứa 30 cell means,
raw sequential runs, variance, provenance và runtime/cost.

Verdict: **PASS_WITH_VARIANCE**. 30/30 means nằm trong paper mean ± 2 reported
SD; chỉ số vòng lặp upstream không reseed nên được gọi là sequential runs, không
gán nhãn seed giả.
