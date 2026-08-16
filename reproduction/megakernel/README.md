# Diffusion sampling mega-launch experiment

Branch này kiểm tra ý tưởng giảm overhead sampling của FK Steering trên Kaggle
P100, T4 x2 và TPU. Tên branch giữ từ khóa `megakernel`, nhưng artifact luôn ghi
`true_megakernel=false`: các mode hiện tại là `torch.compile`, CUDA Graph và
XLA `jit(scan)`, không phải một persistent kernel chứa toàn bộ model.

Đây là ranh giới quan trọng. HazyResearch Megakernels hiện chỉ cung cấp demo
LLM cho H100/B200, nên không thể bê nguyên implementation đó sang P100/T4 hay
TPU. CUDA Graph gom nhiều launch thành một replay; XLA hạ sampling loop thành
compiled control flow; cả hai vẫn khác true megakernel.

## Hai tầng benchmark

1. `bench_loop.py` dùng cùng một synthetic latent denoiser trên Torch và JAX để
   đo riêng Python/driver dispatch overhead:
   - Torch eager;
   - `torch.compile(step)`;
   - `torch.compile(full_loop)`;
   - CUDA Graph trên từng denoising step;
   - JAX eager, `jit(step)` và `jit(lax.scan)` trên TPU.
2. `bench_fk_pipeline.py` chạy model thật qua pipeline FK Steering, nhưng tắt
   reward/resampling để cô lập denoising path:
   - eager;
   - compiled UNet;
   - CUDA-Graph UNet;
   - `device_map=balanced` để thử fit model qua hai T4.

Benchmark SD2.1 mặc định dùng mirror public
`sd2-community/stable-diffusion-2-1`, giống runtime substitution của
reproduction hiện có, vì model ID upstream không còn tải ổn định.

Không pool kết quả hai tầng. Synthetic loop chỉ trả lời “launch overhead có đủ
lớn để đáng tối ưu không”; model thật mới trả lời speedup end-to-end của
denoising. FK + ImageReward/PIL sẽ là gate sau nếu denoising optimization thắng.

## Acceptance gate

`compare_results.py` chỉ đề nghị dùng một mode khi:

- median latency nhanh hơn eager ít nhất 5% sau warmup;
- probe output lệch tối đa không quá `5e-3`;
- optimization không âm thầm fallback;
- peak memory vẫn nằm trong hardware budget.

Compile time được báo riêng và không giấu. Nếu chỉ sinh một ảnh rồi thoát,
compile có thể làm tổng thời gian tệ hơn dù steady-state nhanh hơn.

## P100, T4 và TPU

- Kaggle P100 mặc định hiện dùng image PyTorch CUDA 12.8 không chứa `sm_60`.
  Job phải dùng UV lock của `reproduction/` với Python 3.10.19 để cài Torch
  2.4.0 tương thích Pascal; không được dùng nhầm notebook interpreter.
- T4 x2 chạy được Inductor/CUDA Graph. Với model vừa một GPU, data parallel cho
  prompt/particle độc lập thường hợp lý hơn model sharding. Với model không vừa,
  `device_map=balanced` là fallback để fit; PCIe transfer có thể làm latency
  chậm hơn và được báo như trade-off, không gọi là speedup.
- TPU không chạy CUDA megakernel. `tpu_env/uv.lock` khóa JAX và đường tối ưu là
  `jax.jit(jax.lax.scan)`. Đây mới là phép so backend tương đương về sampling
  loop; model FK PyTorch chưa được tuyên bố là đã port sang TPU.

## Chạy local smoke test

```bash
uv run --project reproduction --frozen \
  python -m unittest discover -s reproduction/megakernel/tests -v
```

Ví dụ benchmark loop Torch:

```bash
uv run --project reproduction --frozen \
  python reproduction/megakernel/bench_loop.py \
  --backend torch --device cuda \
  --modes eager,compile-step,compile-loop,cuda-graph-step \
  --output outputs/loop-torch.json
```

Ví dụ model thật:

```bash
uv run --project reproduction --frozen \
  python reproduction/megakernel/bench_fk_pipeline.py \
  --mode compile-unet --steps 20 --particles 4 \
  --output outputs/fk-compile.json
```

Nguồn kỹ thuật chính:

- HazyResearch Megakernels: <https://github.com/HazyResearch/Megakernels>
- PyTorch `torch.compile`: <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- Diffusers model/device sharding: <https://huggingface.co/docs/diffusers/main/tutorials/inference_with_big_models>
- JAX `lax.scan`: <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>
