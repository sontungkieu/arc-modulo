#!/usr/bin/env python3
"""Benchmark the real FK Steering diffusion pipeline without reward resampling.

The workload uses the upstream FK-capable pipeline with ``fkd_args=None`` to
isolate denoising-loop latency.  Reward-guided SMC is intentionally a later,
separate gate because PIL preprocessing and ImageReward would otherwise hide
whether launch-overhead optimization helped the sampler itself.
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


def _tensor_payload(value: Any) -> tuple[np.ndarray, str]:
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().float().cpu().numpy()
    elif isinstance(value, list) and value and hasattr(value[0], "convert"):
        array = np.stack(
            [np.asarray(image.convert("RGB"), dtype=np.float32) for image in value]
        )
    else:
        array = np.asarray(value, dtype=np.float32)
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return array, digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("eager", "compile-unet", "cuda-graph-unet"), required=True
    )
    parser.add_argument("--model-id", default="sd2-community/stable-diffusion-2-1")
    parser.add_argument("--pipeline", choices=("sd", "sdxl"), default="sd")
    parser.add_argument(
        "--device-map",
        choices=("single", "balanced", "unet-sharded"),
        default="single",
    )
    parser.add_argument("--max-memory-gib", type=float)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from cuda_graph import install_unet_cudagraph
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sd import FKDStableDiffusion
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL

    if not torch.cuda.is_available():
        raise RuntimeError("The real FK pipeline benchmark requires CUDA")
    if args.device_map != "single" and torch.cuda.device_count() < 2:
        raise RuntimeError(f"{args.device_map} requires at least two visible GPUs")
    if args.device_map != "single" and args.mode != "eager":
        raise ValueError(
            "compile/CUDA Graph and cross-device placement are separate benchmark groups"
        )

    pipeline_class = (
        FKDStableDiffusionXL if args.pipeline == "sdxl" else FKDStableDiffusion
    )
    load_kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "use_safetensors": True,
    }
    if args.pipeline == "sd":
        load_kwargs.update({"safety_checker": None, "requires_safety_checker": False})
    if args.device_map == "balanced":
        load_kwargs["device_map"] = "balanced"
        if args.max_memory_gib is not None:
            load_kwargs["max_memory"] = {
                index: f"{args.max_memory_gib:g}GiB"
                for index in range(torch.cuda.device_count())
            }

    load_started = time.perf_counter()
    pipe = pipeline_class.from_pretrained(args.model_id, **load_kwargs)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    unet_device_map = None
    if args.device_map == "single":
        pipe = pipe.to("cuda")
    elif args.device_map == "unet-sharded":
        from accelerate import dispatch_model, infer_auto_device_map

        per_gpu_limit = args.max_memory_gib or 0.8 * min(
            torch.cuda.get_device_properties(index).total_memory / (1024**3)
            for index in range(torch.cuda.device_count())
        )
        max_memory = {
            index: f"{per_gpu_limit:g}GiB" for index in range(torch.cuda.device_count())
        }
        no_split_modules = getattr(pipe.unet, "_no_split_modules", None) or []
        unet_device_map = infer_auto_device_map(
            pipe.unet,
            max_memory=max_memory,
            no_split_module_classes=no_split_modules,
            dtype=torch.float16,
        )
        gpu_devices = {
            int(device.split(":", 1)[1])
            if isinstance(device, str) and device.startswith("cuda:")
            else device
            for device in unet_device_map.values()
            if isinstance(device, int)
            or (isinstance(device, str) and device.startswith("cuda:"))
        }
        if len(gpu_devices) < 2:
            raise RuntimeError(
                "unet-sharded did not place UNet layers on at least two GPUs; "
                f"lower --max-memory-gib (map={unet_device_map})"
            )
        pipe.unet = dispatch_model(pipe.unet, device_map=unet_device_map)
        # Prompt encoding and latent scheduler state originate on cuda:0. Keep
        # the smaller components there; Accelerate hooks transfer activations
        # only at UNet shard boundaries.
        for component_name in ("vae", "text_encoder", "text_encoder_2"):
            component = getattr(pipe, component_name, None)
            if isinstance(component, torch.nn.Module):
                component.to("cuda:0")
    pipe.set_progress_bar_config(disable=True)
    load_s = time.perf_counter() - load_started

    graph_wrapper = None
    if args.mode == "compile-unet":
        capability = torch.cuda.get_device_capability(0)
        if capability[0] < 7:
            raise RuntimeError(
                f"compile-unet is disabled for compute capability {capability}"
            )
        pipe.unet.to(memory_format=torch.channels_last)
        pipe.unet = torch.compile(
            pipe.unet,
            fullgraph=True,
            options={"triton.cudagraphs": False},
        )
    elif args.mode == "cuda-graph-unet":
        graph_wrapper = install_unet_cudagraph(pipe.unet)

    prompts = ["a photo of a brown knife and a blue donut"] * args.particles
    output_type = "pil" if args.decode else "latent"

    def run_once(seed: int) -> Any:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device="cuda:0").manual_seed(seed)
        result = pipe(
            prompts,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            output_type=output_type,
            fkd_args=None,
        )
        return result.images

    torch.cuda.synchronize()
    first_started = time.perf_counter()
    candidate = run_once(args.seed)
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - first_started
    for warmup_index in range(1, args.warmups):
        candidate = run_once(args.seed + warmup_index)
    torch.cuda.synchronize()

    for device_index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_index)
    timings: list[float] = []
    hashes: list[str] = []
    summaries: list[dict[str, float]] = []
    probes: list[list[float]] = []
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        candidate = run_once(args.seed)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        array, digest = _tensor_payload(candidate)
        hashes.append(digest)
        probes.append(
            np.ascontiguousarray(array).reshape(-1)[:256].astype(float).tolist()
        )
        summaries.append(
            {
                "mean": float(array.mean()),
                "std": float(array.std()),
                "min": float(array.min()),
                "max": float(array.max()),
            }
        )

    median_s = statistics.median(timings)
    particle_steps_per_s = args.particles * args.steps / median_s
    device_memory = {
        str(index): {
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
        for index in range(torch.cuda.device_count())
    }
    graph_stats = vars(graph_wrapper.stats) if graph_wrapper is not None else None
    status = "ok"
    if graph_stats and graph_stats["disabled_reason"]:
        status = "fallback"
    payload = {
        "schema_version": 1,
        "benchmark": "fk_steering_real_pipeline_denoising",
        "scope": "real model and scheduler; FK reward/resampling disabled",
        "true_megakernel": False,
        "optimization_kind": {
            "eager": "none",
            "compile-unet": "operator fusion with compiler CUDA Graphs disabled for output ownership",
            "cuda-graph-unet": "multi-kernel CUDA Graph mega-launch",
        }[args.mode],
        "status": status,
        "mode": args.mode,
        "model_id": args.model_id,
        "pipeline": args.pipeline,
        "device_map_requested": args.device_map,
        "hf_device_map": getattr(pipe, "hf_device_map", None),
        "unet_device_map": unet_device_map,
        "max_memory_gib": args.max_memory_gib,
        "particles": args.particles,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "decode": args.decode,
        "load_s": load_s,
        "compile_or_capture_first_call_s": first_call_s,
        "timings_s": timings,
        "median_s": median_s,
        "particle_steps_per_s": particle_steps_per_s,
        "deterministic_output_hashes": hashes,
        "deterministic_within_run": len(set(hashes)) == 1,
        "output_summaries": summaries,
        "output_probes_first_256": probes,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_memory": device_memory,
        "cuda_graph_stats": graph_stats,
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
        "generate_tok_s": particle_steps_per_s,
        "throughput_unit": "particle_steps_per_second",
        "mode": args.mode,
        "model_id": args.model_id,
        "true_megakernel": False,
        "compile_s": first_call_s,
    }
    print("KJO_BENCHMARK_THROUGHPUT_ROW " + json.dumps(row, sort_keys=True))
    print("FK_PIPELINE_BENCHMARK_SUMMARY " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
