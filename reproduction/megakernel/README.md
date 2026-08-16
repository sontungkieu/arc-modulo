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
   - compiled UNet with compiler CUDA Graphs disabled so recurrent outputs are
     not silently reused;
   - CUDA-Graph UNet;
   - pipeline `device_map=balanced` và layer-level `unet-sharded` để phân biệt
     component placement với sharding bên trong denoiser.

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
  thử pipeline `device_map=balanced` trước; `unet-sharded` dùng Accelerate hooks
  để chia layer của chính denoiser khi component lớn nhất vẫn không vừa một GPU.
  PCIe transfer có thể làm latency chậm hơn và được báo như trade-off, không gọi
  là speedup.
- TPU không chạy CUDA megakernel. `tpu_env/uv.lock` khóa JAX và đường tối ưu là
  `jax.jit(jax.lax.scan)`. Đây mới là phép so backend tương đương về sampling
  loop; model FK PyTorch chưa được tuyên bố là đã port sang TPU.

## Kết quả Kaggle hiện tại

Các số dưới đây là median steady-state sau warmup. `speedup > 1` nghĩa là nhanh
hơn eager; một mode chỉ được chấp nhận khi đồng thời qua correctness gate.

| Hardware / workload | Mode | Eager (s) | Optimized (s) | Speedup | Gate |
| --- | --- | ---: | ---: | ---: | --- |
| P100 synthetic loop | CUDA Graph step | 0.03686 | 0.02308 | 1.60x | pass |
| P100 SD2.1, 2 particles x 20 steps | CUDA Graph UNet | 8.75924 | 8.73461 | 1.003x | reject: <5% |
| T4 synthetic loop | `compile(step)` | 0.04145 | 0.00699 | 5.93x | pass; warmup 7.87 s |
| T4 synthetic loop | `compile(full_loop)` | 0.04145 | 0.01234 | 3.36x | pass; warmup 60.37 s |
| T4 synthetic loop | CUDA Graph step | 0.04145 | 0.02439 | 1.70x | pass |
| T4 SD2.1, 2 particles x 20 steps | compiled UNet | 4.49768 | 3.74812 | 1.20x | reject: latent max error 1.082 |
| T4 SDXL, forced two-GPU UNet shard | layer dispatch | 7.65612 | 8.60627 | 0.890x | correct, but 12.4% slower |

Evidence:

- [P100 r3](https://www.kaggle.com/code/bangchi/arc-megakernel-p100-r3-20260816)
  dùng commit `16a6ab3`; Pascal không chạy Inductor theo policy, còn CUDA Graph
  đúng số học nhưng gần như không cải thiện model thật.
- [T4 x2 r4](https://www.kaggle.com/code/bangchi/arc-megakernel-t4x2-r4-20260816)
  tắt compiler CUDA Graph vẫn cho cùng sai lệch latent ở compiled SD2.1. Vì
  vậy sai lệch không chỉ là output aliasing; mode này chưa đủ an toàn để bật.
- [T4 x2 r5](https://www.kaggle.com/code/bangchi/arc-megakernel-t4x2-r5-20260816)
  dùng commit `427b771`; sửa output lifetime giúp synthetic compile đạt sai số
  tối đa `4.77e-7`. Forced SDXL shard thực sự đặt layer lên cả hai T4, nhưng
  peak allocation đo được vẫn lệch (`6.999 GB` trên GPU 0, `0.547 GB` trên GPU
  1) và chậm hơn single-GPU. Đây là feasibility evidence khi áp per-GPU cap,
  chưa phải bằng chứng một model lớn hơn 16 GB đã chạy thành công.
- [TPU r4](https://www.kaggle.com/code/johnntlhudson/arc-megakernel-tpu-r4-20260816)
  dùng commit `427b771`, KJO runtime 0.12.2 và locked JAX environment; job đang
  chờ terminal evidence nên chưa có tuyên bố speedup TPU.

Kết luận tạm thời: tối ưu launch overhead cải thiện mạnh microbenchmark, nhưng
không tự động chuyển thành tốc độ end-to-end khi UNet compute chiếm ưu thế.
Trên T4, `torch.compile` là hướng đáng tiếp tục nếu tìm được correctness issue;
trên P100 nên giữ eager. Sharding chỉ nên là fallback để fit model và phải đo
PCIe overhead, không phải mặc định để tăng tốc.

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
