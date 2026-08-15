#!/usr/bin/env python3
"""Run a provenance-checked slice of the official Null-TTA SD-v1.5 example."""

from __future__ import annotations

import argparse
import builtins
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable


OFFICIAL_COMMIT = "337bf73037e9f24e9f844974d3287384abb610bf"
PAPER_PROMPT_COUNT = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--prompt-start", type=int, default=0)
    parser.add_argument("--prompt-end", type=int, default=PAPER_PROMPT_COUNT)
    parser.add_argument("--target-reward", choices=["pickscore", "aesthetic", "hpsv2"], default="pickscore")
    parser.add_argument("--min-inner-steps", type=int, default=5)
    parser.add_argument("--max-inner-steps", type=int, default=55)
    parser.add_argument("--num-particles", type=int, default=3)
    parser.add_argument("--num-inference-steps", type=int, default=100)
    parser.add_argument("--lambda-alpha", type=float, default=100.0)
    parser.add_argument("--lambda-reg", type=float, default=0.002)
    parser.add_argument("--phi-variance", type=float, default=0.01)
    parser.add_argument("--lr-uncond", type=float, default=0.01)
    parser.add_argument("--tampering-coef", type=float, default=0.008)
    parser.add_argument("--require-all-scorers", action="store_true")
    parser.add_argument(
        "--target-only-generation",
        action="store_true",
        help=(
            "Keep only the optimization target scorer resident during generation. "
            "The saved base/optimized images must subsequently be evaluated with "
            "score_null_tta_images.py before Table 1 aggregation."
        ),
    )
    return parser.parse_args()


def load_official_module(example_path: Path):
    spec = importlib.util.spec_from_file_location("null_tta_official_sd", example_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official example: {example_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def wrap_reward_loader(
    registry: dict[str, dict[str, Any]],
    name: str,
    loader: Callable[[], Any],
    *,
    required: bool,
) -> Callable[[], Any]:
    def wrapped() -> Any:
        scorer = loader()
        scorer_type = type(scorer).__name__
        scorer_module = type(scorer).__module__
        is_mock = scorer_type.lower().startswith("mock")
        is_disabled = scorer_type == "DisabledEvaluationScorer"
        registry[name] = {
            "type": scorer_type,
            "module": scorer_module,
            "is_mock": is_mock,
            "is_disabled": is_disabled,
        }
        if required and is_mock:
            raise RuntimeError(f"Required target scorer {name} fell back to {scorer_type}")
        return scorer

    return wrapped


class DisabledEvaluationScorer:
    """Zero-parameter placeholder for metrics intentionally deferred to post-scoring."""

    def __call__(self, images: Any, prompts: Any) -> Any:
        import torch

        return torch.full(
            (images.shape[0],),
            float("nan"),
            device=images.device,
            dtype=torch.float32,
        )

    def eval(self) -> "DisabledEvaluationScorer":
        return self


def finite_float(value: str, *, field: str, row_index: int) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {field} at CSV row {row_index}: {value!r}")
    return parsed


def main() -> int:
    args = parse_args()
    if args.target_only_generation and args.require_all_scorers:
        raise ValueError("--target-only-generation and --require-all-scorers are mutually exclusive")
    if args.target_only_generation and args.target_reward != "pickscore":
        raise ValueError("The memory-reduced generation path is validated only for PickScore targeting")
    upstream = args.upstream_root.resolve()
    example_path = upstream / "examples/null_tta_sd.py"
    if not example_path.is_file():
        raise FileNotFoundError(example_path)
    if not (0 <= args.prompt_start < args.prompt_end <= PAPER_PROMPT_COUNT):
        raise ValueError(
            f"Invalid prompt slice [{args.prompt_start}, {args.prompt_end}); expected within [0, {PAPER_PROMPT_COUNT})"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    os.chdir(output_dir)
    sys.path.insert(0, str(upstream))

    started = time.perf_counter()
    module = load_official_module(example_path)
    if len(module.prompt_list) != PAPER_PROMPT_COUNT:
        raise RuntimeError(
            f"Official prompt list changed: expected {PAPER_PROMPT_COUNT}, found {len(module.prompt_list)}"
        )
    selected_prompts = list(module.prompt_list[args.prompt_start : args.prompt_end])
    module.prompt_list = selected_prompts

    scorer_registry: dict[str, dict[str, Any]] = {}
    module.get_reward_fn = wrap_reward_loader(
        scorer_registry,
        "pickscore",
        module.get_reward_fn,
        required=args.require_all_scorers or args.target_reward == "pickscore",
    )
    if args.target_only_generation:
        module.get_aesthetic_fn = DisabledEvaluationScorer
        module.get_hps_fn = DisabledEvaluationScorer
        module.get_imagereward_fn = DisabledEvaluationScorer
    module.get_aesthetic_fn = wrap_reward_loader(
        scorer_registry,
        "aesthetic",
        module.get_aesthetic_fn,
        required=args.require_all_scorers or args.target_reward == "aesthetic",
    )
    module.get_hps_fn = wrap_reward_loader(
        scorer_registry,
        "hpsv2",
        module.get_hps_fn,
        required=args.require_all_scorers or args.target_reward == "hpsv2",
    )
    module.get_imagereward_fn = wrap_reward_loader(
        scorer_registry,
        "imagereward",
        module.get_imagereward_fn,
        required=args.require_all_scorers,
    )

    # The official inner loop catches broad exceptions and otherwise continues
    # after CUDA OOM, which can produce a result CSV even though reward-guided
    # updates were skipped.  Preserve the original message, then fail closed so
    # an operationally completed job cannot be accepted as scientific output.
    def fail_closed_print(*values: object, **kwargs: Any) -> None:
        builtins.print(*values, **kwargs)
        rendered = " ".join(str(value) for value in values)
        if "CUDA out of memory" in rendered or "ERROR decoding or getting reward in inner loop" in rendered:
            raise RuntimeError(f"Null-TTA inner-loop failure detected: {rendered}")

    module.print = fail_closed_print

    sys.argv = [
        str(example_path),
        "--target_reward",
        args.target_reward,
        "--lambda_alpha",
        str(args.lambda_alpha),
        "--lambda_reg",
        str(args.lambda_reg),
        "--phi_variance",
        str(args.phi_variance),
        "--seed",
        str(args.seed),
        "--min_inner_steps",
        str(args.min_inner_steps),
        "--max_inner_steps",
        str(args.max_inner_steps),
        "--num_particles",
        str(args.num_particles),
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--lr_uncond",
        str(args.lr_uncond),
        "--tampering_coef",
        str(args.tampering_coef),
    ]
    module.main()

    csv_candidates = sorted(output_dir.rglob(f"results_{args.target_reward}.csv"))
    if len(csv_candidates) != 1:
        raise RuntimeError(f"Expected one result CSV, found {len(csv_candidates)}: {csv_candidates}")
    result_csv = csv_candidates[0]
    with result_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    prompt_rows = [row for row in rows if row.get("prompt_idx") != "avg"]
    if len(prompt_rows) != len(selected_prompts):
        raise RuntimeError(
            f"Expected {len(selected_prompts)} prompt rows, found {len(prompt_rows)} in {result_csv}"
        )

    metric_columns = {
        "pickscore": ("baseline_pick", "optimized_pick"),
        "aesthetic": ("baseline_aes", "optimized_aes"),
        "hpsv2": ("baseline_hps", "optimized_hps"),
        "imagereward": ("baseline_ir", "optimized_ir"),
    }
    baseline_column, optimized_column = metric_columns[args.target_reward]
    per_prompt = []
    for row_index, row in enumerate(prompt_rows):
        baseline = finite_float(row[baseline_column], field=baseline_column, row_index=row_index)
        optimized = finite_float(row[optimized_column], field=optimized_column, row_index=row_index)
        expected_prompt = selected_prompts[row_index]
        if row["prompt"] != expected_prompt:
            raise RuntimeError(f"Prompt mismatch at slice row {row_index}")
        row_metrics: dict[str, dict[str, float] | None] = {}
        for metric_name, (base_field, opt_field) in metric_columns.items():
            if scorer_registry[metric_name]["is_mock"] or scorer_registry[metric_name]["is_disabled"]:
                row_metrics[metric_name] = None
                continue
            metric_base = finite_float(row[base_field], field=base_field, row_index=row_index)
            metric_opt = finite_float(row[opt_field], field=opt_field, row_index=row_index)
            row_metrics[metric_name] = {
                "baseline": metric_base,
                "optimized": metric_opt,
                "improvement": metric_opt - metric_base,
            }
        per_prompt.append(
            {
                "global_prompt_index": args.prompt_start + row_index,
                "prompt": expected_prompt,
                "baseline": baseline,
                "optimized": optimized,
                "improvement": optimized - baseline,
                "metrics": row_metrics,
            }
        )

    baseline_values = [row["baseline"] for row in per_prompt]
    optimized_values = [row["optimized"] for row in per_prompt]
    improvement_values = [row["improvement"] for row in per_prompt]

    def stats(values: list[float]) -> dict[str, float | int]:
        n = len(values)
        sample_std = stdev(values) if n > 1 else 0.0
        return {
            "mean": mean(values),
            "std": sample_std,
            "n": n,
            "se": sample_std / math.sqrt(n) if n else float("nan"),
        }

    all_metrics: dict[str, dict[str, dict[str, float | int]] | None] = {}
    for metric_name in metric_columns:
        metric_rows = [row["metrics"][metric_name] for row in per_prompt]
        if any(row is None for row in metric_rows):
            all_metrics[metric_name] = None
            continue
        typed_rows = [row for row in metric_rows if row is not None]
        all_metrics[metric_name] = {
            "baseline": stats([row["baseline"] for row in typed_rows]),
            "optimized": stats([row["optimized"] for row in typed_rows]),
            "improvement": stats([row["improvement"] for row in typed_rows]),
        }

    import torch

    payload = {
        "schema_version": 1,
        "paper": "Null-TTA",
        "paper_venue": "CVPR 2026",
        "upstream_commit": OFFICIAL_COMMIT,
        "official_example_sha256": sha256(example_path),
        "scope": "official SD-v1.5 example on a contiguous slice of the official 50-prompt list",
        "seed": args.seed,
        "prompt_slice": [args.prompt_start, args.prompt_end],
        "prompt_count": len(selected_prompts),
        "target_reward": args.target_reward,
        "config": {
            "min_inner_steps": args.min_inner_steps,
            "max_inner_steps": args.max_inner_steps,
            "num_particles": args.num_particles,
            "num_inference_steps": args.num_inference_steps,
            "lambda_alpha": args.lambda_alpha,
            "lambda_reg": args.lambda_reg,
            "phi_variance": args.phi_variance,
            "lr_uncond": args.lr_uncond,
            "tampering_coef": args.tampering_coef,
        },
        "scorers": scorer_registry,
        "require_all_scorers": args.require_all_scorers,
        "target_only_generation": args.target_only_generation,
        "all_scorers_are_real": all(
            not item["is_mock"] and not item["is_disabled"] for item in scorer_registry.values()
        ),
        "target_scorer_is_real": not scorer_registry[args.target_reward]["is_mock"],
        "metrics": all_metrics,
        "target_metric": {
            "baseline": stats(baseline_values),
            "optimized": stats(optimized_values),
            "improvement": stats(improvement_values),
        },
        "per_prompt": per_prompt,
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else None
        ),
        "result_csv": str(result_csv.relative_to(output_dir)),
    }
    metrics_path = output_dir / "null_tta_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("NULL_TTA_REPRO_JSON=" + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
