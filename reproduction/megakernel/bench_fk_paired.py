#!/usr/bin/env python3
"""Paired correctness and latency benchmark for the real FK pipeline.

The eager baseline and optimized candidate share one loaded model, one explicit
initial latent, one prompt batch, and reset RNG state.  Numerical traces are
captured in separate unmeasured runs so device-to-host copies do not pollute the
steady-state latency samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_TO_IMAGE = REPO_ROOT / "text_to_image"
sys.path.insert(0, str(TEXT_TO_IMAGE))
sys.path.insert(0, str(TEXT_TO_IMAGE / "fkd_diffusers"))


def tensor_payload(value: Any) -> tuple[np.ndarray, str]:
    """Materialize a pipeline output as a float32 array plus a stable hash."""
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    elif isinstance(value, list) and value and hasattr(value[0], "convert"):
        array = np.stack(
            [np.asarray(image.convert("RGB"), dtype=np.float32) for image in value]
        )
    else:
        array = np.asarray(value, dtype=np.float32)
    contiguous = np.ascontiguousarray(array)
    return contiguous, hashlib.sha256(contiguous.tobytes()).hexdigest()


def difference(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Return allclose-style error evidence without retaining full tensors."""
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "allclose": False,
            "violation_fraction": 1.0,
        }
    reference64 = reference.astype(np.float64, copy=False)
    candidate64 = candidate.astype(np.float64, copy=False)
    absolute = np.abs(candidate64 - reference64)
    tolerance = atol + rtol * np.abs(reference64)
    violations = absolute > tolerance
    denominator = np.maximum(np.abs(reference64), 1e-12)
    relative = absolute / denominator
    return {
        "shape_match": True,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "allclose": not bool(np.any(violations)),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(absolute.max(initial=0.0)),
        "mean_abs": float(absolute.mean()) if absolute.size else 0.0,
        "rmse": float(np.sqrt(np.mean(absolute**2))) if absolute.size else 0.0,
        "max_rel": float(relative.max(initial=0.0)),
        "violation_fraction": float(violations.mean()) if violations.size else 0.0,
    }


def trace_difference(
    reference: list[np.ndarray],
    candidate: list[np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for index, (reference_step, candidate_step) in enumerate(
        zip(reference, candidate, strict=False)
    ):
        rows.append(
            {
                "step_index": index,
                **difference(reference_step, candidate_step, atol=atol, rtol=rtol),
            }
        )
    first_failed = next(
        (row["step_index"] for row in rows if not row["allclose"]), None
    )
    return {
        "reference_steps": len(reference),
        "candidate_steps": len(candidate),
        "step_count_match": len(reference) == len(candidate),
        "all_steps_allclose": len(reference) == len(candidate)
        and all(row["allclose"] for row in rows),
        "first_failed_step": first_failed,
        "max_abs_over_steps": max((row.get("max_abs", 0.0) for row in rows), default=0.0),
        "max_violation_fraction_over_steps": max(
            (row.get("violation_fraction", 0.0) for row in rows), default=0.0
        ),
        "steps": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", choices=("compile-unet", "cuda-graph-unet"), required=True
    )
    parser.add_argument("--workload", choices=("denoise", "full-fk"), required=True)
    parser.add_argument("--model-id", default="sd2-community/stable-diffusion-2-1")
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="a photo of a brown knife and a blue donut")
    parser.add_argument("--potential-type", choices=("diff", "max", "add", "rt"), default="diff")
    parser.add_argument("--lmbda", type=float, default=2.0)
    parser.add_argument("--resample-frequency", type=int, default=20)
    parser.add_argument("--resampling-t-start", type=int, default=20)
    parser.add_argument("--resampling-t-end", type=int, default=80)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--min-speedup", type=float, default=1.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.particles < 1 or args.steps < 1 or args.repeats < 1 or args.warmups < 0:
        raise ValueError("particles, steps, and repeats must be positive; warmups nonnegative")
    if args.workload == "full-fk" and args.resampling_t_end >= args.steps:
        raise ValueError("full-fk requires resampling_t_end < steps")

    import torch
    from cuda_graph import install_unet_cudagraph
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion

    if not torch.cuda.is_available():
        raise RuntimeError("The paired real-pipeline benchmark requires CUDA")
    if args.candidate == "compile-unet" and torch.cuda.get_device_capability(0)[0] < 7:
        raise RuntimeError("compile-unet requires compute capability 7.0 or newer")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    load_started = time.perf_counter()
    pipe = FKDStableDiffusion.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    eager_unet = pipe.unet
    load_s = time.perf_counter() - load_started

    prompts = [args.prompt] * args.particles
    latent_height = args.height // pipe.vae_scale_factor
    latent_width = args.width // pipe.vae_scale_factor
    latent_generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    initial_latents = torch.randn(
        (
            args.particles,
            pipe.unet.config.in_channels,
            latent_height,
            latent_width,
        ),
        generator=latent_generator,
        device="cuda:0",
        dtype=pipe.unet.dtype,
    )
    initial_array, initial_hash = tensor_payload(initial_latents)

    reward_preload_s = None
    if args.workload == "full-fk":
        from PIL import Image
        # The pipeline imports this module as top-level ``rewards``.  Reuse
        # that exact module namespace so the measured run sees the warm cache.
        from rewards import get_reward_function

        reward_started = time.perf_counter()
        get_reward_function(
            "ImageReward",
            images=[Image.new("RGB", (224, 224))] * args.particles,
            prompts=prompts,
        )
        torch.cuda.synchronize()
        reward_preload_s = time.perf_counter() - reward_started

    eta = args.eta if args.eta is not None else (1.0 if args.workload == "full-fk" else 0.0)
    fkd_args: dict[str, Any] | None = None
    output_type = "latent"
    if args.workload == "full-fk":
        output_type = "pil"
        fkd_args = {
            "potential_type": args.potential_type,
            "lmbda": args.lmbda,
            "num_particles": args.particles,
            "adaptive_resampling": True,
            "resample_frequency": args.resample_frequency,
            "resampling_t_start": args.resampling_t_start,
            "resampling_t_end": args.resampling_t_end,
            "time_steps": args.steps,
            "guidance_reward_fn": "ImageReward",
            "metric_to_chase": None,
            "use_smc": True,
        }

    def reset_rng() -> torch.Generator:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        return torch.Generator(device="cuda:0").manual_seed(args.seed)

    def run_once(*, capture_trace: bool = False) -> tuple[Any, list[np.ndarray]]:
        trace: list[np.ndarray] = []

        def callback(
            _pipeline: Any,
            _index: int,
            _timestep: Any,
            values: dict[str, Any],
        ) -> dict[str, Any]:
            trace.append(values["latents"].detach().float().cpu().numpy().copy())
            return values

        result = pipe(
            prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            eta=eta,
            generator=reset_rng(),
            latents=initial_latents.clone(),
            output_type=output_type,
            fkd_args=fkd_args,
            callback_on_step_end=callback if capture_trace else None,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        return result.images, trace

    def benchmark_mode() -> dict[str, Any]:
        torch.cuda.synchronize()
        first_started = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - first_started
        for _ in range(max(0, args.warmups - 1)):
            run_once()
        torch.cuda.synchronize()

        for device_index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_index)
        timings: list[float] = []
        hashes: list[str] = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output, _ = run_once()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            _, digest = tensor_payload(output)
            hashes.append(digest)
        median_s = statistics.median(timings)
        output_array, output_hash = tensor_payload(output)
        return {
            "first_call_s": first_call_s,
            "timings_s": timings,
            "median_s": median_s,
            "particle_steps_per_s": args.particles * args.steps / median_s,
            "images_per_s": args.particles / median_s,
            "output_hashes": hashes,
            "deterministic_within_mode": len(set(hashes)) == 1,
            "output_hash": output_hash,
            "output_summary": {
                "mean": float(output_array.mean()),
                "std": float(output_array.std()),
                "min": float(output_array.min()),
                "max": float(output_array.max()),
            },
            "device_memory": {
                str(index): {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
                }
                for index in range(torch.cuda.device_count())
            },
        }

    pipe.unet = eager_unet
    eager_metrics = benchmark_mode()
    eager_trace_output, eager_trace = run_once(capture_trace=True)
    eager_output_array, eager_output_hash = tensor_payload(eager_trace_output)

    graph_wrapper = None
    compile_options = None
    if args.candidate == "compile-unet":
        compile_options = {"triton.cudagraphs": False}
        pipe.unet = torch.compile(eager_unet, fullgraph=True, options=compile_options)
    else:
        pipe.unet = eager_unet
        graph_wrapper = install_unet_cudagraph(pipe.unet)

    candidate_metrics = benchmark_mode()
    candidate_trace_output, candidate_trace = run_once(capture_trace=True)
    candidate_output_array, candidate_output_hash = tensor_payload(candidate_trace_output)

    final_difference = difference(
        eager_output_array, candidate_output_array, atol=args.atol, rtol=args.rtol
    )
    per_step = trace_difference(
        eager_trace, candidate_trace, atol=args.atol, rtol=args.rtol
    )
    speedup = eager_metrics["median_s"] / candidate_metrics["median_s"]
    graph_stats = vars(graph_wrapper.stats) if graph_wrapper is not None else None
    no_fallback = not graph_stats or not graph_stats.get("disabled_reason")
    numerical_match = (
        eager_metrics["deterministic_within_mode"]
        and candidate_metrics["deterministic_within_mode"]
        and final_difference["allclose"]
        and per_step["all_steps_allclose"]
    )
    accepted = numerical_match and no_fallback and speedup >= args.min_speedup

    payload = {
        "schema_version": 2,
        "benchmark": "fk_steering_real_pipeline_paired",
        "scope": (
            "real FK Steering SD pipeline with ImageReward and SMC"
            if args.workload == "full-fk"
            else "real FK Steering SD pipeline denoising with reward/SMC disabled"
        ),
        "true_megakernel": False,
        "workload": args.workload,
        "candidate": args.candidate,
        "model_id": args.model_id,
        "prompt": args.prompt,
        "particles": args.particles,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "eta": eta,
        "fkd_args": fkd_args,
        "seed": args.seed,
        "initial_latent_sha256": initial_hash,
        "initial_latent_summary": {
            "shape": list(initial_array.shape),
            "mean": float(initial_array.mean()),
            "std": float(initial_array.std()),
        },
        "load_s": load_s,
        "reward_preload_s": reward_preload_s,
        "eager": eager_metrics,
        "optimized": candidate_metrics,
        "eager_trace_output_hash": eager_output_hash,
        "optimized_trace_output_hash": candidate_output_hash,
        "speedup_steady_state": speedup,
        "break_even_repeats": (
            max(
                0.0,
                (candidate_metrics["first_call_s"] - candidate_metrics["median_s"])
                / max(eager_metrics["median_s"] - candidate_metrics["median_s"], 1e-12),
            )
            if speedup > 1.0
            else None
        ),
        "numerics": {
            "contract": (
                "same loaded weights, prompt batch, explicit initial latent, seed, "
                "scheduler, and trace point"
            ),
            "final_output": final_difference,
            "per_step_latents": per_step,
            "numerical_match": numerical_match,
        },
        "optimization": {
            "kind": (
                "torch.compile UNet; original contiguous model layout; compiler "
                "CUDA Graphs disabled"
                if args.candidate == "compile-unet"
                else "multi-kernel CUDA Graph replay of UNet"
            ),
            "compile_options": compile_options,
            "cuda_graph_stats": graph_stats,
            "no_fallback": no_fallback,
        },
        "acceptance": {
            "min_speedup": args.min_speedup,
            "atol": args.atol,
            "rtol": args.rtol,
            "passed": accepted,
            "speed_gate": speedup >= args.min_speedup,
            "numerical_gate": numerical_match,
            "fallback_gate": no_fallback,
        },
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "determinism_settings": {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    row = {
        "bs": args.particles,
        "p": 0,
        "g": args.steps,
        "prefill_tok_s": 0.0,
        "generate_tok_s": candidate_metrics["particle_steps_per_s"],
        "throughput_unit": "particle_steps_per_second",
        "mode": args.candidate,
        "workload": args.workload,
        "speedup": speedup,
        "numerical_match": numerical_match,
        "accepted": accepted,
        "true_megakernel": False,
    }
    print("KJO_BENCHMARK_THROUGHPUT_ROW " + json.dumps(row, sort_keys=True))
    print("FK_PAIRED_BENCHMARK_SUMMARY " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
