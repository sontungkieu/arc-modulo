"""CUDA Graph helpers for static-shape diffusion denoiser calls.

This module deliberately calls the optimization a *mega-launch*, not a true
megakernel.  A CUDA Graph replay removes Python/driver launch overhead but the
captured graph still contains many CUDA kernels.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


def _tree_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            str(value.device),
            str(value.layout),
            bool(value.requires_grad),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_tree_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_tree_signature(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _tree_signature(value[key])) for key in sorted(value)),
        )
    return ("constant", type(value).__qualname__, repr(value))


def _tree_clone_static(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_tree_clone_static(item) for item in value)
    if isinstance(value, list):
        return [_tree_clone_static(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_clone_static(item) for key, item in value.items()}
    return value


def _tree_copy_(destination: Any, source: Any) -> None:
    if isinstance(source, torch.Tensor):
        destination.copy_(source)
        return
    if isinstance(source, (tuple, list)):
        for dst_item, src_item in zip(destination, source, strict=True):
            _tree_copy_(dst_item, src_item)
        return
    if isinstance(source, dict):
        for key in source:
            _tree_copy_(destination[key], source[key])


@dataclass
class CudaGraphStats:
    capture_count: int = 0
    replay_count: int = 0
    fallback_count: int = 0
    disabled_reason: str | None = None


class CudaGraphCallable:
    """Capture one static-shape callable and replay it on subsequent calls.

    Tensor values are copied into stable input buffers before replay.  A new
    tensor shape/dtype/non-tensor signature triggers one new capture.  Capture
    failure is fail-open: the original callable remains usable and the reason
    is exposed through :attr:`stats` for benchmark evidence.
    """

    def __init__(self, function: Callable[..., Any], warmup_calls: int = 2) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CudaGraphCallable requires a CUDA runtime")
        self.function = function
        self.warmup_calls = max(1, int(warmup_calls))
        self.stats = CudaGraphStats()
        self._signature: Any = None
        self._static_args: tuple[Any, ...] | None = None
        self._static_kwargs: dict[str, Any] | None = None
        self._static_output: Any = None
        self._graph: torch.cuda.CUDAGraph | None = None

    def _capture(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        self._static_args = _tree_clone_static(args)
        self._static_kwargs = _tree_clone_static(kwargs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(self.warmup_calls):
                self.function(*self._static_args, **self._static_kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            static_output = self.function(*self._static_args, **self._static_kwargs)
        self._graph = graph
        self._static_output = static_output
        self._signature = (_tree_signature(args), _tree_signature(kwargs))
        self.stats.capture_count += 1

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.stats.disabled_reason is not None:
            self.stats.fallback_count += 1
            return self.function(*args, **kwargs)

        signature = (_tree_signature(args), _tree_signature(kwargs))
        try:
            if self._graph is None or signature != self._signature:
                self._capture(args, kwargs)
                return self._static_output
            assert self._static_args is not None
            assert self._static_kwargs is not None
            _tree_copy_(self._static_args, args)
            _tree_copy_(self._static_kwargs, kwargs)
            self._graph.replay()
            self.stats.replay_count += 1
            return self._static_output
        except Exception as exc:  # noqa: BLE001 - capture support differs by model/GPU
            self.stats.disabled_reason = f"{type(exc).__name__}: {exc}"
            self.stats.fallback_count += 1
            return self.function(*args, **kwargs)


def install_unet_cudagraph(unet: torch.nn.Module) -> CudaGraphCallable:
    """Replace ``unet.forward`` with a fail-open CUDA Graph replay wrapper."""

    original_forward = unet.forward
    wrapper = CudaGraphCallable(original_forward)
    unet.forward = wrapper  # type: ignore[method-assign]
    return wrapper
