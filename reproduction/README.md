# FK Steering reproduction overlay

Branch này giữ nguyên lịch sử upstream tới commit
`9413005dee1e79f80fb4561a4a7ac8eec704281b`. Môi trường ảnh được khóa riêng vì
pin Diffusers/ImageReward xung đột với toy stack.

```bash
uv sync --project reproduction --frozen --no-install-project
cd text_to_image
uv run --project ../reproduction python launch_eval_runs.py \
  --use_smc --model_name='stabilityai/stable-diffusion-xl-base-1.0' \
  --lmbda=10.0 --resample_frequency=20 --resample_t_start=20 \
  --resample_t_end=80 --num_particles=4 --potential_type=max
```

Lệnh trên là launcher upstream, không phải đúng workload nhỏ đã audit. Workload
đã audit dùng một prompt, seeds 42/43/44, SD 2.1, `k=4`, lambda 2.0 và
ImageReward; config/result nằm trong `configs/` và `results/`. Wrapper Modal
exact nằm ở branch `main` vì cần layout `upstream/fk-steering`.

Verdict: **PARTIAL** — chỉ là reproduction định tính/một prompt, không phải
GenEval table và không đủ để tuyên bố phương pháp thắng baseline.

Nhánh `experiment/fk-steering-megakernel` bổ sung một experiment độc lập tại
[`megakernel/`](megakernel/): benchmark launch-overhead, compiled sampling,
CUDA Graph, XLA scan và fallback sharding. Các artifact phân biệt rõ true
megakernel với mega-launch approximation; chưa trộn benchmark systems này vào
verdict reproduction khoa học ở trên.

Phần discrete diffusion có submodule `discrete_diffusion/mdlm`; lấy bằng
`git submodule update --init --recursive` nếu cần. Phạm vi ảnh đã chạy không
dùng submodule này.
