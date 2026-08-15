# Null-TTA reproduction overlay

Branch này giữ nguyên lịch sử upstream tới commit
`337bf73037e9f24e9f844974d3287384abb610bf`. Thư mục `reproduction/` là lớp
portable đã dùng để tái lập Table 1 trên SD-v1.5: `nmax=55`, seeds 42/43/44,
50 prompts/seed và bốn scorer thật.

```bash
uv sync --project reproduction --frozen --no-install-project

# Canary một prompt; vẫn cần GPU và tải model/reward weights.
uv run --project reproduction python reproduction/scripts/run_null_tta.py \
  --upstream-root . --output-dir runs/null_tta_s42_p0_1 \
  --seed 42 --prompt-start 0 --prompt-end 1 \
  --min-inner-steps 5 --max-inner-steps 55 --require-all-scorers
```

Kết quả aggregate ở `results/`; báo cáo/figure ở `docs/`. Wrapper Modal exact
dùng layout nhiều-paper, vì vậy chạy wrapper đó từ branch `main` của
`sontungkieu/arc-modulo`, nơi `upstream/null-tta` tồn tại đúng như lúc audit.

Verdict: **PASS**. PickScore Null-TTA tái lập là 0.315612 so với paper 0.315;
đây là hàng Null-TTA, không phải hàng backbone SD-v1.5.
