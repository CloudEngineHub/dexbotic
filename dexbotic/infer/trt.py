"""Model-independent TensorRT build, metadata, and execution helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "EngineManifest",
    "TensorManifest",
    "TensorRTRunner",
    "build_fp16_engine_from_onnx",
    "engine_shape",
    "load_tensorrt",
    "resolve_io_names",
]


@dataclass(frozen=True)
class TensorManifest:
    """Describe one input or output tensor in a TensorRT engine."""

    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class EngineManifest:
    """Serializable metadata describing a generated TensorRT engine."""

    model_family: str
    model_revision: str
    component: str
    precision: str
    inputs: tuple[TensorManifest, ...]
    outputs: tuple[TensorManifest, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def write(self, path: str | Path) -> None:
        """Write this manifest as formatted JSON."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> "EngineManifest":
        """Load an engine manifest from JSON."""

        raw = json.loads(Path(path).read_text())
        raw["inputs"] = tuple(TensorManifest(**item) for item in raw["inputs"])
        raw["outputs"] = tuple(TensorManifest(**item) for item in raw["outputs"])
        return cls(**raw)


def load_tensorrt(context: str):
    """Import TensorRT or raise an error identifying the requesting component."""

    try:
        import tensorrt as trt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(f"{context} requires the tensorrt Python package.") from exc
    return trt


def resolve_cuda_device(device: torch.device | str, context: str) -> torch.device:
    requested = torch.device(device)
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"{context} requires an available CUDA device.")
    if requested.index is None:
        requested = torch.device("cuda", torch.cuda.current_device())
    return requested


def deserialize_engine(trt: Any, engine_path: str | Path, context: str):
    engine_path = Path(engine_path)
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to deserialize {context} engine: {engine_path}")
    return logger, engine


def has_tensor_api(engine: Any) -> bool:
    return hasattr(engine, "num_io_tensors")


def resolve_io_names(engine: Any, trt: Any) -> tuple[list[str], list[str]]:
    """Return input and output names for a TensorRT engine."""

    inputs: list[str] = []
    outputs: list[str] = []
    if has_tensor_api(engine):
        for index in range(int(engine.num_io_tensors)):
            name = engine.get_tensor_name(index)
            mode = engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                inputs.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                outputs.append(name)
    else:
        for index in range(int(engine.num_bindings)):
            name = engine.get_binding_name(index)
            (inputs if engine.binding_is_input(index) else outputs).append(name)
    return inputs, outputs


def binding_index(engine: Any, name: str) -> int:
    if not hasattr(engine, "get_binding_index"):
        raise RuntimeError("TensorRT binding-index API is unavailable.")
    return int(engine.get_binding_index(name))


def engine_shape(engine: Any, name: str) -> tuple[int, ...]:
    """Return the declared shape of a named engine tensor or binding."""

    if has_tensor_api(engine):
        return tuple(int(dim) for dim in engine.get_tensor_shape(name))
    return tuple(
        int(dim) for dim in engine.get_binding_shape(binding_index(engine, name))
    )


def runtime_shape(engine: Any, context: Any, name: str) -> tuple[int, ...]:
    if has_tensor_api(engine):
        return tuple(int(dim) for dim in context.get_tensor_shape(name))
    return tuple(
        int(dim) for dim in context.get_binding_shape(binding_index(engine, name))
    )


def trt_dtype_to_torch(trt: Any, trt_dtype: Any) -> torch.dtype:
    mapping = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    if hasattr(trt, "int64"):
        mapping[trt.int64] = torch.int64
    if hasattr(trt, "bfloat16"):
        mapping[trt.bfloat16] = torch.bfloat16
    try:
        return mapping[trt_dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported TensorRT dtype: {trt_dtype}") from exc


def tensor_dtype(engine: Any, trt: Any, name: str) -> torch.dtype:
    if has_tensor_api(engine):
        value = engine.get_tensor_dtype(name)
    else:
        value = engine.get_binding_dtype(binding_index(engine, name))
    return trt_dtype_to_torch(trt, value)


def set_input_shape(
    engine: Any, context: Any, name: str, shape: tuple[int, ...]
) -> None:
    if has_tensor_api(engine):
        if any(dim < 0 for dim in engine_shape(engine, name)):
            context.set_input_shape(name, shape)
        return
    index = binding_index(engine, name)
    if any(int(dim) < 0 for dim in engine.get_binding_shape(index)):
        context.set_binding_shape(index, shape)


def execute_async(
    *,
    engine: Any,
    context: Any,
    device: torch.device,
    inputs: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
) -> bool:
    stream = torch.cuda.current_stream(device)
    if has_tensor_api(engine):
        for name, tensor in {**inputs, **outputs}.items():
            context.set_tensor_address(name, int(tensor.data_ptr()))
        return bool(context.execute_async_v3(stream_handle=stream.cuda_stream))
    bindings = [0] * int(engine.num_bindings)
    for name, tensor in {**inputs, **outputs}.items():
        bindings[binding_index(engine, name)] = int(tensor.data_ptr())
    return bool(
        context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
    )


def build_fp16_engine_from_onnx(
    *,
    onnx_path: str | Path,
    engine_path: str | Path,
    workspace_gb: float,
    context: str,
) -> None:
    """Build and save an FP16 TensorRT engine from an ONNX model."""

    trt = load_tensorrt(f"{context} TensorRT export")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    else:
        network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    onnx_path = Path(onnx_path)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError(
            f"Failed to parse {context} ONNX for TensorRT:\n" + "\n".join(errors)
        )
    config = builder.create_builder_config()
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    workspace_bytes = int(float(workspace_gb) * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        serialized = None if engine is None else engine.serialize()
    if serialized is None:
        raise RuntimeError(f"Failed to build {context} TensorRT engine.")
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))


class TensorRTRunner:
    """Model-independent owner of one TensorRT engine and execution context."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | str,
        context: str,
    ) -> None:
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"{context} engine not found: {self.engine_path}")
        self.device = resolve_cuda_device(device, context)
        self.trt = load_tensorrt(context)
        self.logger, self.engine = deserialize_engine(
            self.trt, self.engine_path, context
        )
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create {context} execution context.")
        self.input_names, self.output_names = resolve_io_names(self.engine, self.trt)
        self.input_dtypes = {
            name: tensor_dtype(self.engine, self.trt, name) for name in self.input_names
        }
        self.output_dtypes = {
            name: tensor_dtype(self.engine, self.trt, name)
            for name in self.output_names
        }

    def prepare_input(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        """Move an input to the engine device and required dtype."""

        tensor = tensor.to(device=self.device, dtype=self.input_dtypes[name])
        return tensor if tensor.is_contiguous() else tensor.contiguous()

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> None:
        """Set a dynamic input shape when the engine requires it."""

        set_input_shape(self.engine, self.context, name, shape)

    def output_shape(self, name: str) -> tuple[int, ...]:
        """Return the resolved runtime shape of an output tensor."""

        return runtime_shape(self.engine, self.context, name)

    def execute(
        self,
        *,
        inputs: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> None:
        """Execute the engine asynchronously on the current CUDA stream."""

        if not execute_async(
            engine=self.engine,
            context=self.context,
            device=self.device,
            inputs=inputs,
            outputs=outputs,
        ):
            raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")
