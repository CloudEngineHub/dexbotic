"""Reusable fixed-shape CUDA Graph runner."""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping, TypeAlias, Union

import torch


@lru_cache(maxsize=1)
def ensure_cuda_graph_available() -> None:
    """Validate CUDA Graph support once for the lifetime of this process."""

    if not torch.cuda.is_available() or not hasattr(torch.cuda, "CUDAGraph"):
        raise RuntimeError("CUDA Graph requires an available CUDA runtime.")


TensorTree: TypeAlias = Union[
    torch.Tensor,
    tuple["TensorTree", ...],
    list["TensorTree"],
    dict[str, "TensorTree"],
]


def _map_tree(
    value: TensorTree, fn: Callable[[torch.Tensor], torch.Tensor]
) -> TensorTree:
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_map_tree(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_tree(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_tree(item, fn) for key, item in value.items()}
    output_type = type(value).__name__
    raise TypeError(f"CUDA Graph output contains unsupported {output_type}.")


class CudaGraphRunner:
    """Capture and replay a tensor-only callable with address-stable inputs."""

    def __init__(
        self,
        graph,
        static_inputs,
        static_outputs,
        *,
        clone_outputs: bool = True,
    ) -> None:
        self._graph = graph
        self._static_inputs = static_inputs
        self._static_outputs = static_outputs
        self._clone_outputs = bool(clone_outputs)

    @property
    def static_inputs(self) -> Mapping[str, torch.Tensor]:
        """Expose owned staging tensors for zero-copy profile preparation."""

        return self._static_inputs

    @classmethod
    def capture(
        cls,
        function: Callable[..., TensorTree],
        capture_inputs: Mapping[str, torch.Tensor],
        *,
        warmup_steps: int = 2,
        clone_outputs: bool = True,
    ) -> "CudaGraphRunner":
        ensure_cuda_graph_available()
        if not capture_inputs:
            raise ValueError("CUDA Graph capture requires at least one input tensor.")
        devices = {tensor.device for tensor in capture_inputs.values()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("CUDA Graph inputs must share one CUDA device.")
        device = next(iter(devices))
        static_inputs = {
            name: tensor.detach().clone() for name, tensor in capture_inputs.items()
        }
        for tensor in static_inputs.values():
            torch._dynamo.mark_static_address(tensor)
        torch.cuda.synchronize(device)
        with torch.inference_mode():
            for _ in range(warmup_steps):
                function(**static_inputs)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            static_outputs = function(**static_inputs)
        torch.cuda.synchronize(device)
        _map_tree(static_outputs, lambda tensor: tensor)
        return cls(
            graph,
            static_inputs,
            static_outputs,
            clone_outputs=clone_outputs,
        )

    def replay(self) -> TensorTree:
        """Replay already-staged buffers; the caller must serialize access."""

        self._graph.replay()
        if self._clone_outputs:
            return _map_tree(self._static_outputs, torch.Tensor.clone)
        return self._static_outputs
