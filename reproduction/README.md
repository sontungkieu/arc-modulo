# DAS reproduction overlay

Branch này giữ nguyên lịch sử upstream tới commit
`2f4b2239f29ee59f80359bdfa5ed747b6d855a1e`. Lockfile và runner dưới
`reproduction/` tái lập Figure 1 toy 2-D GMM với Python/Torch/Numba seed được
ghim đồng nhất.

```bash
uv sync --project reproduction --frozen --no-install-project
uv run --project reproduction python reproduction/scripts/run_notebook.py \
  notebooks/GMM.ipynb \
  --output-dir runs/das_gmm_seed42 \
  --upstream-root . --pythonpath . --timeout 7200
```

`kaggle/` chứa source cell cho T4x2 canary và seed-42 rescue. Allocation T4x2
đã PASS, nhưng notebook khoa học upstream chỉ dùng một GPU và số Kaggle seed 42
FAIL numeric; không được đổi nhãn đó thành reproduction pass.

Verdict Modal: **PASS_WITH_VARIANCE**. Seed 43 là bad seed. PDF giải thích
EMD/Wasserstein-1, figure và JSON đầy đủ nằm trong `docs/` và `results/`.
