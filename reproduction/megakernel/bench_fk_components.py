#!/usr/bin/env python3
"""Measure full SDXL FK Steering with explicit one- or two-GPU placement.

The split layout keeps the denoising state and UNet on ``cuda:0`` while
prompt encoders, VAE decoding, and ImageReward run on ``cuda:1``.  Prompt
embeddings are computed explicitly before sampling so both layouts exercise
the same pipeline call contract and can be compared numerically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TEXT_TO_IMAGE = REPO_ROOT / "text_to_image"
sys.path.insert(0, str(TEXT_TO_IMAGE))
sys.path.insert(0, str(TEXT_TO_IMAGE / "fkd_diffusers"))


def image_payload(images: list[Any]) -> tuple[np.ndarray, str]:
    array = np.stack(
        [np.asarray(image.convert("RGB"), dtype=np.uint8) for image in images]
    )
    contiguous = np.ascontiguousarray(array)
    return contiguous, hashlib.sha256(contiguous.tobytes()).hexdigest()


def module_summary(module: Any) -> dict[str, Any] | None:
    if module is None:
        return None
    parameters = list(module.parameters())
    buffers = list(module.buffers())
    tensors = [*parameters, *buffers]
    device = str(tensors[0].device) if tensors else "unknown"
    bytes_ = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return {
        "device": device,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_and_buffer_bytes": bytes_,
        "dtype": str(parameters[0].dtype) if parameters else "unknown",
    }


def synchronize_devices(torch: Any, device_indices: set[int]) -> None:
    for index in sorted(device_indices):
        # Torch 2.4 may reject memory/synchronization calls for a secondary
        # device before its CUDA context has been initialized. Entering the
        # already-placed component's context avoids probing unused GPUs.
        with torch.cuda.device(index):
            torch.cuda.synchronize()


def memory_summary(torch: Any) -> dict[str, Any]:
    return {
        str(index): {
            "name": torch.cuda.get_device_name(index),
            "total_bytes": torch.cuda.get_device_properties(index).total_memory,
            "current_allocated_bytes": torch.cuda.memory_allocated(index),
            "current_reserved_bytes": torch.cuda.memory_reserved(index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
        for index in range(torch.cuda.device_count())
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=("single", "split"), required=True)
    parser.add_argument(
        "--model-id", default="stabilityai/stable-diffusion-xl-base-1.0"
    )
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt", default="a photo of a brown knife and a blue donut"
    )
    parser.add_argument(
        "--potential-type", choices=("diff", "max", "add", "rt"), default="max"
    )
    parser.add_argument("--lmbda", type=float, default=10.0)
    parser.add_argument("--adaptive-resampling", action="store_true")
    parser.add_argument("--resample-frequency", type=int, default=20)
    parser.add_argument("--resampling-t-start", type=int, default=20)
    parser.add_argument("--resampling-t-end", type=int, default=80)
    parser.add_argument(
        "--vae-decode-batch-size",
        type=int,
        default=0,
        help="VAE decode microbatch size; 0 keeps the full particle batch.",
    )
    parser.add_argument(
        "--reward-batch-size",
        type=int,
        default=0,
        help="ImageReward microbatch size; 0 keeps the full particle batch.",
    )
    parser.add_argument(
        "--empty-cache-between-auxiliary-phases",
        action="store_true",
        help="Release cached VAE blocks before ImageReward inference.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--array-output", type=Path)
    args = parser.parse_args()

    if args.particles < 2 or args.steps < 2:
        raise ValueError("full FK canary requires at least two particles and steps")
    if args.resampling_t_end >= args.steps:
        raise ValueError("resampling_t_end must be smaller than steps")
    if args.vae_decode_batch_size < 0 or args.reward_batch_size < 0:
        raise ValueError("microbatch sizes must be nonnegative")

    import torch
    from PIL import Image
    from diffusers import DDIMScheduler
    from fkd_diffusers.fkd_pipeline_sdxl import FKDStableDiffusionXL
    import rewards as reward_module

    if not torch.cuda.is_available():
        raise RuntimeError("The component-placement canary requires CUDA")
    if args.layout == "split" and torch.cuda.device_count() < 2:
        raise RuntimeError("The split layout requires two visible CUDA devices")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    denoise_device = torch.device("cuda:0")
    auxiliary_device = (
        torch.device("cuda:1") if args.layout == "split" else denoise_device
    )
    active_device_indices = {
        int(denoise_device.index or 0),
        int(auxiliary_device.index or 0),
    }
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "fk_steering_sdxl_component_placement",
        "scope": "full SDXL FK Steering with ImageReward",
        "layout": args.layout,
        "model_id": args.model_id,
        "particles": args.particles,
        "steps": args.steps,
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "eta": args.eta,
        "seed": args.seed,
        "prompt": args.prompt,
        "vae_decode_batch_size": args.vae_decode_batch_size,
        "reward_batch_size": args.reward_batch_size,
        "empty_cache_between_auxiliary_phases": (
            args.empty_cache_between_auxiliary_phases
        ),
        "denoise_device": str(denoise_device),
        "auxiliary_device": str(auxiliary_device),
        "status": "starting",
    }

    try:
        load_started = time.perf_counter()
        pipe = FKDStableDiffusionXL.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
        )
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        payload["model_load_s"] = time.perf_counter() - load_started

        placement_started = time.perf_counter()
        pipe.unet.to(denoise_device)
        pipe.vae.to(auxiliary_device)
        pipe.text_encoder.to(auxiliary_device)
        pipe.text_encoder_2.to(auxiliary_device)
        synchronize_devices(torch, active_device_indices)
        for index in sorted(active_device_indices):
            with torch.cuda.device(index):
                torch.cuda.reset_peak_memory_stats()
        payload["placement_s"] = time.perf_counter() - placement_started

        prompts = [args.prompt] * args.particles
        encode_started = time.perf_counter()
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt=prompts,
                prompt_2=None,
                device=auxiliary_device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=args.guidance_scale > 1.0,
                negative_prompt=None,
                negative_prompt_2=None,
            )
        synchronize_devices(torch, active_device_indices)
        payload["text_encode_s"] = time.perf_counter() - encode_started

        reward_started = time.perf_counter()
        reward_module.REWARDS_DICT["ImageReward"] = reward_module.rm_load(
            "ImageReward-v1.0", device=auxiliary_device
        )
        reward_preload_batch_size = args.reward_batch_size or args.particles
        preload_images = [Image.new("RGB", (224, 224))] * args.particles
        for start in range(0, args.particles, reward_preload_batch_size):
            stop = min(start + reward_preload_batch_size, args.particles)
            reward_module.get_reward_function(
                "ImageReward",
                images=preload_images[start:stop],
                prompts=prompts[start:stop],
            )
        synchronize_devices(torch, active_device_indices)
        payload["reward_preload_s"] = time.perf_counter() - reward_started
        if args.empty_cache_between_auxiliary_phases:
            with torch.cuda.device(auxiliary_device):
                torch.cuda.empty_cache()

        latent_generator = torch.Generator(device=denoise_device).manual_seed(args.seed)
        initial_latents = torch.randn(
            (
                args.particles,
                pipe.unet.config.in_channels,
                args.height // pipe.vae_scale_factor,
                args.width // pipe.vae_scale_factor,
            ),
            generator=latent_generator,
            device=denoise_device,
            dtype=pipe.unet.dtype,
        )
        initial_latents_array = initial_latents.detach().float().cpu().numpy()
        payload["initial_latent_sha256"] = hashlib.sha256(
            np.ascontiguousarray(initial_latents_array).tobytes()
        ).hexdigest()

        fkd_args = {
            "potential_type": args.potential_type,
            "lmbda": args.lmbda,
            "num_particles": args.particles,
            "adaptive_resampling": args.adaptive_resampling,
            "resample_frequency": args.resample_frequency,
            "resampling_t_start": args.resampling_t_start,
            "resampling_t_end": args.resampling_t_end,
            "time_steps": args.steps,
            "guidance_reward_fn": "ImageReward",
            "metric_to_chase": None,
            "use_smc": True,
            "execution_device": str(denoise_device),
            "reward_prompts": prompts,
            "record_trace": True,
            "vae_decode_batch_size": args.vae_decode_batch_size,
            "reward_batch_size": args.reward_batch_size,
            "empty_cache_between_auxiliary_phases": (
                args.empty_cache_between_auxiliary_phases
            ),
        }
        payload["fkd_args"] = {
            key: value for key, value in fkd_args.items() if key != "reward_prompts"
        }

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        sample_generator = torch.Generator(device=denoise_device).manual_seed(args.seed)
        synchronize_devices(torch, active_device_indices)
        sample_started = time.perf_counter()
        result = pipe(
            prompt=None,
            prompt_2=None,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            eta=args.eta,
            generator=sample_generator,
            latents=initial_latents.clone(),
            output_type="pil",
            fkd_args=fkd_args,
        )
        synchronize_devices(torch, active_device_indices)
        payload["sample_s"] = time.perf_counter() - sample_started
        payload["inference_s"] = payload["text_encode_s"] + payload["sample_s"]

        output_array, output_hash = image_payload(result.images)
        payload["output_sha256"] = output_hash
        payload["output_summary"] = {
            "shape": list(output_array.shape),
            "mean": float(output_array.mean()),
            "std": float(output_array.std()),
            "min": int(output_array.min()),
            "max": int(output_array.max()),
        }
        payload["fkd_trace"] = pipe._fkd_trace
        if args.array_output is not None:
            args.array_output.parent.mkdir(parents=True, exist_ok=True)
            np.save(args.array_output, output_array, allow_pickle=False)
            payload["array_output"] = str(args.array_output)

        payload["status"] = "ok"
        payload["component_placement"] = {
            name: module_summary(getattr(pipe, name, None))
            for name in ("unet", "vae", "text_encoder", "text_encoder_2")
        }
    except (torch.cuda.OutOfMemoryError, MemoryError) as error:
        payload["status"] = "oom"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)[-2000:]
        torch.cuda.empty_cache()
    except RuntimeError as error:
        # Some CUDA kernels surface allocation failures as a plain
        # ``RuntimeError`` rather than ``torch.cuda.OutOfMemoryError``.  Keep
        # expected capacity failures machine-readable, but never turn an
        # unrelated runtime error into an apparent OOM result.
        if "out of memory" not in str(error).lower():
            payload["status"] = "error"
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)[-2000:]
            payload["traceback_tail"] = traceback.format_exc()[-5000:]
            raise
        payload["status"] = "oom"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)[-2000:]
        torch.cuda.empty_cache()
    except Exception as error:
        payload["status"] = "error"
        payload["error_type"] = type(error).__name__
        payload["error"] = str(error)[-2000:]
        payload["traceback_tail"] = traceback.format_exc()[-5000:]
        payload["elapsed_s"] = time.perf_counter() - started
        payload["device_memory"] = memory_summary(torch)
        write_payload(args.output, payload)
        raise
    finally:
        payload["elapsed_s"] = time.perf_counter() - started
        payload["device_memory"] = memory_summary(torch)
        payload["torch_version"] = torch.__version__
        payload["torch_cuda"] = torch.version.cuda
        payload["gpu_count"] = torch.cuda.device_count()
        write_payload(args.output, payload)

    generate_rate = (
        args.particles * args.steps / payload["sample_s"]
        if payload["status"] == "ok"
        else 0.0
    )
    row = {
        "bs": args.particles,
        "p": 0,
        "g": args.steps,
        "prefill_tok_s": 0.0,
        "generate_tok_s": generate_rate,
        "throughput_unit": "particle_steps_per_second",
        "layout": args.layout,
        "status": payload["status"],
        "model_id": args.model_id,
    }
    print("KJO_BENCHMARK_THROUGHPUT_ROW " + json.dumps(row, sort_keys=True))
    print("FK_COMPONENT_BENCHMARK_SUMMARY " + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
