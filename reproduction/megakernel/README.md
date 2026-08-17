# Diffusion sampling mega-launch experiment

Branch này kiểm tra ý tưởng giảm overhead sampling của FK Steering trên Kaggle
P100, T4 x2 và TPU. Tên branch giữ từ khóa `megakernel`, nhưng artifact luôn ghi
`true_megakernel=false`: các mode hiện tại là `torch.compile`, CUDA Graph và
XLA `jit(scan)`, không phải một persistent kernel chứa toàn bộ model.

Đây là ranh giới quan trọng. HazyResearch Megakernels hiện chỉ cung cấp demo
LLM cho H100/B200, nên không thể bê nguyên implementation đó sang P100/T4 hay
TPU. CUDA Graph gom nhiều launch thành một replay; XLA hạ sampling loop thành
compiled control flow; cả hai vẫn khác true megakernel.

## Ba tầng benchmark

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
3. `bench_fk_paired.py` là gate quyết định cho model thật. Eager và candidate
   dùng chung đúng một model đã load, prompt batch, initial latent tường minh,
   seed, scheduler và layout bộ nhớ. Script đo riêng:
   - denoising-only để cô lập UNet/scheduler;
   - full FK Steering theo cấu hình paper (`4` particles, `100` steps,
     ImageReward, DDIM `eta=1`, resampling từ step `20` đến `80` mỗi `20`
     steps);
   - first call, steady-state median, compile break-even, peak memory;
   - sai số final output và latent sau từng step bằng contract
     `atol=rtol=5e-3`.

Trace correctness chạy ngoài các mẫu latency để phép copy latent về CPU không
làm chậm số benchmark. Với `torch.compile`, baseline và candidate đều giữ
layout contiguous nguyên bản của repo; như vậy không trộn ảnh hưởng của
channels-last vào sai số hay speedup.

Benchmark SD2.1 mặc định dùng mirror public
`sd2-community/stable-diffusion-2-1`, giống runtime substitution của
reproduction hiện có, vì model ID upstream không còn tải ổn định.

Không pool kết quả ba tầng. Synthetic loop chỉ trả lời “launch overhead có đủ
lớn để đáng tối ưu không”; model thật mới trả lời speedup end-to-end. Paired
gate hiện đo cả denoising riêng và pipeline hoàn chỉnh với FK + ImageReward/PIL.

## Acceptance gate

`compare_results.py` và paired harness chỉ đề nghị dùng một mode khi:

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
| TPU synthetic loop | `jit(step)` | 0.52526 | 0.01955 | 26.87x | pass; first call 0.804 s |
| TPU synthetic loop | `jit(lax.scan)` | 0.52526 | 0.00174 | 301.35x | pass; first call 1.332 s |

Paired gate ở commit `6550efe` dùng cùng model đã load, prompt, initial latent,
seed và scheduler cho eager/candidate. Tất cả dòng dưới dùng SD2.1, `4`
particles, `100` steps; full FK còn dùng ImageReward, DDIM `eta=1` và SMC theo
config paper. T4 runtime nhìn thấy hai GPU nhưng workload single-device chỉ dùng
GPU 0; đây là latency một T4, không phải data-parallel throughput hai T4.

| Hardware / workload | Mode | Eager (s) | Optimized (s) | Speedup | Numerical gate | Decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| P100 denoising | CUDA Graph UNet | 82.912 | 82.907 | 1.00006x | exact, 100/100 steps | reject: <5% |
| P100 full FK | CUDA Graph UNet | 89.789 | 89.771 | 1.00019x | exact, 100/100 steps and pixels | reject: <5% |
| T4 denoising | compiled UNet | 52.621 | 45.606 | 1.1538x | fail from step index 7; final max/mean error 4.568/0.140 | reject: incorrect |
| T4 full FK | compiled UNet | 61.439 | 54.497 | 1.1274x | fail from step index 6; final pixel max/mean error 255/63.016 | reject: incorrect |
| T4 denoising | CUDA Graph UNet | 52.576 | 52.651 | 0.9986x | exact, 100/100 steps | reject: slower |
| T4 full FK | CUDA Graph UNet | 61.492 | 61.854 | 0.9942x | exact, 100/100 steps and pixels | reject: slower |

`torch.compile` deterministic trong từng mode nhưng khác eager: sai số nhỏ tích
lũy từ trước mốc SMC resampling đầu tiên ở step index `20`, rồi khuếch đại qua
100 denoising steps. Vì vậy đây không phải randomness từ multinomial/ImageReward
và cũng không phải output aliasing. Denoising compile first call trên T4 là
`144.47 s`, steady-state `45.61 s`, break-even khoảng `14.1` batch calls; full FK
là `104.10 s`, `54.50 s`, break-even khoảng `7.1`. Các break-even này chỉ là
diagnostic vì correctness gate đã fail.

CUDA Graph thực sự capture một lần, replay hàng trăm lần và không fallback,
nhưng không cải thiện end-to-end. Trên P100, full FK tăng eager latency khoảng
`6.88 s` so với denoising-only; trên T4 tăng khoảng `8.82 s`. Một T4 nhanh hơn
P100 khoảng `1.58x` ở denoising và `1.46x` ở full FK trong contract này.

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
  dùng commit `427b771`, KJO runtime 0.12.2, Python 3.11.6, UV 0.11.13 và locked
  JAX 0.6.2 environment. Cả ba mode cho output khớp eager tuyệt đối trong phép
  đo này. `jit(scan)` nhanh hơn `jit(step)` 11.21x, tách được lợi ích compile cả
  loop khỏi lợi ích compile riêng denoiser step. Tám TPU core được nhìn thấy,
  nhưng benchmark không khai báo sharding nên chỉ là single-default-device
  launch-overhead proxy, không phải multi-core hay full diffusion-model result.
- [P100 paired real pipeline](https://www.kaggle.com/code/johnntlhudson/arc-fk-real-p100-r1-20260817)
  dùng exact commit `6550efe`, KJO runtime 0.12.2 SHA256
  `3344255e6d6e563b389caf346b2bb78bed87984875468050c57ddc97dc39acfd`,
  Python 3.10.19, UV 0.11.13 và unchanged lock. Cả denoising lẫn full FK khớp
  bit-for-bit nhưng CUDA Graph không đạt speed gate.
- [T4 x2 paired real pipeline](https://www.kaggle.com/code/johnntlhudson/arc-fk-real-t4x2-r1-20260817)
  dùng cùng commit/runtime/locked environment. Cả bốn paired steps hoàn tất;
  compiled UNet nhanh nhưng sai, còn CUDA Graph đúng nhưng không nhanh.

Kết luận hiện tại: chưa có mode nào vừa đúng vừa nhanh trên pipeline FK thật.
Giữ eager cho P100/T4. `torch.compile` trên T4 chỉ đáng tiếp tục như một nhánh
điều tra numerical lowering/precision, không được bật cho generation. CUDA
Graph không có lợi end-to-end ở batch/workload này. Sharding chỉ nên là fallback
để fit model và phải đo PCIe overhead, không phải mặc định để tăng tốc. TPU
`jit(lax.scan)` vẫn chỉ pass proxy gate; repo FK hiện là PyTorch/CUDA và chưa có
model/reward pipeline JAX tương đương, nên không được gọi số `301x` là speedup
của repo thật.

## SDXL component placement trên T4 x2

`bench_fk_components.py` kiểm tra fallback cho SDXL khi toàn bộ full-FK workload
không vừa một GPU. Đây là component placement, không phải tensor/model sharding:

- GPU 0 giữ UNet, scheduler state và latent trong toàn bộ denoising loop;
- GPU 1 giữ hai text encoder, VAE và ImageReward;
- prompt embedding chỉ chuyển GPU 1 -> GPU 0 một lần trước sampling;
- ở mỗi checkpoint FK, `x0_pred` chuyển GPU 0 -> GPU 1 để decode/chấm reward,
  rồi chỉ vector reward chuyển ngược về GPU 0 để resample.

Đặt riêng text encoder trên GPU 1 là hợp lệ nhưng thường chưa tận dụng được GPU
thứ hai vì encoder chỉ chạy một lần mỗi prompt. VAE và ImageReward mới là các
component được gọi lặp tại checkpoint FK, nên canary đặt cả cụm auxiliary này
trên GPU 1. Script báo riêng placement, parameter bytes, peak allocated/reserved
memory và latency của từng layout.

Gate có hai tầng. Cấu hình nhỏ phải cho output pixel single/split khớp đúng; sau
đó còn phải khớp reward, importance weight, ESS và resampling indices tại từng
checkpoint. Cấu hình paper-sized một prompt mới được dùng để xác nhận split
layout có fit T4 16 GB hay không. Một prompt pass vẫn chỉ là feasibility canary,
không phải reproduction GenEval hay bằng chứng cho metric tổng hợp của paper.

```bash
uv run --project reproduction --frozen \
  python reproduction/megakernel/bench_fk_components.py \
  --layout split --particles 4 --steps 100 --height 1024 --width 1024 \
  --resampling-t-start 20 --resampling-t-end 80 --resample-frequency 20 \
  --output outputs/fk-sdxl-split.json \
  --array-output outputs/fk-sdxl-split.npy
```

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

Gate paired denoising và full FK thật:

```bash
uv run --project reproduction --frozen \
  python reproduction/megakernel/bench_fk_paired.py \
  --candidate compile-unet --workload denoise \
  --particles 4 --steps 100 --warmups 1 --repeats 3 \
  --output outputs/fk-paired-denoise-compile.json

uv run --project reproduction --frozen \
  python reproduction/megakernel/bench_fk_paired.py \
  --candidate cuda-graph-unet --workload full-fk \
  --particles 4 --steps 100 --warmups 1 --repeats 3 \
  --output outputs/fk-paired-full-cudagraph.json
```

Nguồn kỹ thuật chính:

- HazyResearch Megakernels: <https://github.com/HazyResearch/Megakernels>
- PyTorch `torch.compile`: <https://docs.pytorch.org/docs/stable/generated/torch.compile.html>
- Diffusers model/device sharding: <https://huggingface.co/docs/diffusers/main/tutorials/inference_with_big_models>
- JAX `lax.scan`: <https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>
