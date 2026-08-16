#!/usr/bin/env python3
"""Cross-backend diffusion-loop launch-overhead benchmark.

The denoiser is intentionally synthetic and weight-compatible across Torch and
JAX.  It isolates repeated sampling-loop dispatch overhead; it is not evidence
that a full Stable Diffusion model has the same speedup.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

TRUE_MEGAKERNEL = False


@dataclass
class Result:
    backend: str
    mode: str
    status: str
    median_s: float | None
    min_s: float | None
    compile_warmup_s: float | None
    particle_steps_per_s: float | None
    speedup_vs_eager: float | None
    max_abs_error_vs_eager: float | None
    mean_abs_error_vs_eager: float | None
    peak_memory_bytes: int | None
    reason: str | None = None


def _weights(seed: int, channels: int, blocks: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    scale = 1.0 / math.sqrt(channels)
    return {
        "input": (rng.standard_normal((channels, channels)) * scale).astype("float32"),
        "blocks": (rng.standard_normal((blocks, channels, channels)) * scale).astype(
            "float32"
        ),
        "biases": (rng.standard_normal((blocks, channels)) * 0.01).astype("float32"),
        "output": (rng.standard_normal((channels, channels)) * scale).astype("float32"),
    }


def _emit(result: Result, *, particles: int, steps: int) -> None:
    payload = asdict(result)
    payload.update(
        {
            "bs": particles,
            "p": 0,
            "g": steps,
            "prefill_tok_s": 0.0,
            "generate_tok_s": result.particle_steps_per_s or 0.0,
            "throughput_unit": "particle_steps_per_second",
            "true_megakernel": TRUE_MEGAKERNEL,
        }
    )
    print("KJO_BENCHMARK_THROUGHPUT_ROW " + json.dumps(payload, sort_keys=True))


def _torch_run(args: argparse.Namespace) -> tuple[list[Result], dict[str, Any]]:
    import torch
    from cuda_graph import CudaGraphCallable

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    raw = _weights(args.seed, args.channels, args.blocks)
    weights = {key: torch.from_numpy(value).to(device) for key, value in raw.items()}
    initial = torch.from_numpy(
        np.random.default_rng(args.seed + 1)
        .standard_normal((args.particles, args.height, args.width, args.channels))
        .astype("float32")
    ).to(device)
    alphas = torch.linspace(0.002, 0.02, args.steps, device=device)
    timesteps = torch.linspace(1.0, 0.0, args.steps, device=device)

    def denoise(x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(x @ weights["input"] + timestep)
        for block, bias in zip(weights["blocks"], weights["biases"], strict=True):
            spatial = (
                torch.roll(h, 1, 1)
                + torch.roll(h, -1, 1)
                + torch.roll(h, 1, 2)
                + torch.roll(h, -1, 2)
            ) * 0.125
            h = h + 0.1 * torch.tanh(h @ block + bias + spatial)
        return h @ weights["output"]

    def step(
        x: torch.Tensor, alpha: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        return x - alpha * denoise(x, timestep)

    def loop(
        step_function: Callable[..., torch.Tensor], x: torch.Tensor
    ) -> torch.Tensor:
        for index in range(args.steps):
            x = step_function(x, alphas[index], timesteps[index])
        return x

    def full_loop(x: torch.Tensor) -> torch.Tensor:
        return loop(step, x)

    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None
    with torch.inference_mode():
        reference = loop(step, initial.clone())
        synchronize()

    requested_modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    results: list[Result] = []
    eager_median: float | None = None
    cuda_capability = (
        torch.cuda.get_device_capability() if device.type == "cuda" else None
    )

    for mode in requested_modes:
        if mode not in {"eager", "compile-step", "compile-loop", "cuda-graph-step"}:
            results.append(
                Result(
                    "torch",
                    mode,
                    "skipped",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "unknown mode",
                )
            )
            continue
        if mode != "eager" and device.type != "cuda":
            results.append(
                Result(
                    "torch",
                    mode,
                    "skipped",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "optimization requires CUDA",
                )
            )
            continue
        if mode.startswith("compile") and cuda_capability and cuda_capability[0] < 7:
            results.append(
                Result(
                    "torch",
                    mode,
                    "skipped",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    f"Inductor/Triton unsupported policy for compute capability {cuda_capability}",
                )
            )
            continue

        graph_wrapper: CudaGraphCallable | None = None
        if mode == "eager":

            def run_once() -> torch.Tensor:
                return loop(step, initial.clone())

        elif mode == "compile-step":
            compiled_step = torch.compile(step, mode="reduce-overhead", fullgraph=True)

            def run_compiled_steps(
                compiled_step: Callable[..., torch.Tensor] = compiled_step,
            ) -> torch.Tensor:
                x = initial.clone()
                for index in range(args.steps):
                    # reduce-overhead uses CUDAGraph Trees. Marking the logical
                    # iteration prevents a later replay from overwriting an
                    # output that the diffusion recurrence still owns.
                    torch.compiler.cudagraph_mark_step_begin()
                    x = compiled_step(x, alphas[index], timesteps[index]).clone()
                return x

            run_once = run_compiled_steps
        elif mode == "compile-loop":
            compiled_loop = torch.compile(
                full_loop, mode="reduce-overhead", fullgraph=True
            )

            def run_once(
                compiled_loop: Callable[[torch.Tensor], torch.Tensor] = compiled_loop,
            ) -> torch.Tensor:
                return compiled_loop(initial.clone())

        else:
            graph_wrapper = CudaGraphCallable(step)

            def run_once(
                graph_wrapper: CudaGraphCallable = graph_wrapper,
            ) -> torch.Tensor:
                return loop(graph_wrapper, initial.clone())

        try:
            synchronize()
            first_start = time.perf_counter()
            candidate = run_once()
            synchronize()
            compile_warmup_s = time.perf_counter() - first_start
            for _ in range(max(0, args.warmups - 1)):
                candidate = run_once()
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(args.repeats):
                synchronize()
                started = time.perf_counter()
                candidate = run_once()
                synchronize()
                timings.append(time.perf_counter() - started)
            difference = (candidate.float() - reference.float()).abs()
            median_s = statistics.median(timings)
            if mode == "eager":
                eager_median = median_s
            status = "ok"
            reason = None
            if graph_wrapper and graph_wrapper.stats.disabled_reason:
                status = "fallback"
                reason = graph_wrapper.stats.disabled_reason
            result = Result(
                backend="torch",
                mode=mode,
                status=status,
                median_s=median_s,
                min_s=min(timings),
                compile_warmup_s=compile_warmup_s,
                particle_steps_per_s=args.particles * args.steps / median_s,
                speedup_vs_eager=(eager_median / median_s) if eager_median else None,
                max_abs_error_vs_eager=float(difference.max().item()),
                mean_abs_error_vs_eager=float(difference.mean().item()),
                peak_memory_bytes=(
                    torch.cuda.max_memory_allocated() if device.type == "cuda" else None
                ),
                reason=reason,
            )
        except Exception as exc:  # noqa: BLE001 - preserve benchmark failure evidence
            result = Result(
                "torch",
                mode,
                "error",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        _emit(result, particles=args.particles, steps=args.steps)

    metadata = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_capability": cuda_capability,
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "gpu_count": torch.cuda.device_count() if device.type == "cuda" else 0,
    }
    return results, metadata


def _jax_run(args: argparse.Namespace) -> tuple[list[Result], dict[str, Any]]:
    import jax
    import jax.numpy as jnp

    raw = _weights(args.seed, args.channels, args.blocks)
    weights = {key: jnp.asarray(value) for key, value in raw.items()}
    initial = jnp.asarray(
        np.random.default_rng(args.seed + 1)
        .standard_normal((args.particles, args.height, args.width, args.channels))
        .astype("float32")
    )
    alphas = jnp.linspace(0.002, 0.02, args.steps)
    timesteps = jnp.linspace(1.0, 0.0, args.steps)

    def denoise(x: Any, timestep: Any) -> Any:
        h = jnp.tanh(x @ weights["input"] + timestep)
        for index in range(args.blocks):
            spatial = (
                jnp.roll(h, 1, 1)
                + jnp.roll(h, -1, 1)
                + jnp.roll(h, 1, 2)
                + jnp.roll(h, -1, 2)
            ) * 0.125
            h = h + 0.1 * jnp.tanh(
                h @ weights["blocks"][index] + weights["biases"][index] + spatial
            )
        return h @ weights["output"]

    def step(x: Any, alpha: Any, timestep: Any) -> Any:
        return x - alpha * denoise(x, timestep)

    def eager_loop(x: Any, step_fn: Callable[..., Any]) -> Any:
        for index in range(args.steps):
            x = step_fn(x, alphas[index], timesteps[index])
        return x

    def scan_loop(x: Any) -> Any:
        def body(carry: Any, inputs: tuple[Any, Any]) -> tuple[Any, None]:
            alpha, timestep = inputs
            return step(carry, alpha, timestep), None

        return jax.lax.scan(body, x, (alphas, timesteps))[0]

    reference = eager_loop(initial, step)
    reference.block_until_ready()
    modes = [part.strip() for part in args.modes.split(",") if part.strip()]
    results: list[Result] = []
    eager_median: float | None = None
    for mode in modes:
        if mode == "eager":

            def run_once() -> Any:
                return eager_loop(initial, step)

        elif mode in {"jit-step", "compile-step"}:
            compiled_step = jax.jit(step)

            def run_once(compiled_step: Callable[..., Any] = compiled_step) -> Any:
                return eager_loop(initial, compiled_step)

        elif mode in {"jit-scan", "compile-loop"}:
            compiled_scan = jax.jit(scan_loop)

            def run_once(compiled_scan: Callable[..., Any] = compiled_scan) -> Any:
                return compiled_scan(initial)

        else:
            result = Result(
                "jax",
                mode,
                "skipped",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "unknown mode",
            )
            results.append(result)
            continue
        try:
            started = time.perf_counter()
            candidate = run_once()
            candidate.block_until_ready()
            compile_warmup_s = time.perf_counter() - started
            for _ in range(max(0, args.warmups - 1)):
                run_once().block_until_ready()
            timings = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                candidate = run_once()
                candidate.block_until_ready()
                timings.append(time.perf_counter() - started)
            difference = np.abs(
                np.asarray(candidate, dtype=np.float32)
                - np.asarray(reference, dtype=np.float32)
            )
            median_s = statistics.median(timings)
            if mode == "eager":
                eager_median = median_s
            result = Result(
                "jax",
                mode,
                "ok",
                median_s,
                min(timings),
                compile_warmup_s,
                args.particles * args.steps / median_s,
                (eager_median / median_s) if eager_median else None,
                float(difference.max()),
                float(difference.mean()),
                None,
            )
        except Exception as exc:  # noqa: BLE001 - preserve benchmark failure evidence
            result = Result(
                "jax",
                mode,
                "error",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
        _emit(result, particles=args.particles, steps=args.steps)
    metadata = {
        "jax_version": jax.__version__,
        "default_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_count": jax.device_count(),
    }
    return results, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "jax"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--modes", default="eager,compile-step,compile-loop,cuda-graph-step"
    )
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        min(
            args.particles,
            args.steps,
            args.height,
            args.width,
            args.channels,
            args.blocks,
            args.warmups,
            args.repeats,
        )
        <= 0
    ):
        parser.error("all numeric workload arguments must be positive")
    if args.backend == "torch":
        results, metadata = _torch_run(args)
    else:
        results, metadata = _jax_run(args)
    payload = {
        "schema_version": 1,
        "benchmark": "synthetic_diffusion_sampling_loop",
        "scope": "launch-overhead proxy; not full-model scientific evidence",
        "true_megakernel": TRUE_MEGAKERNEL,
        "config": {
            key: value for key, value in vars(args).items() if key not in {"output"}
        },
        "metadata": metadata,
        "results": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("DIFFUSION_LOOP_SUMMARY " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
