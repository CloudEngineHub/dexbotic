"""Optional FP8 prefix MLP execution with persistent per-bucket TRT contexts."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dexbotic.infer.artifacts import engine_directory_lock, read_verified_file
from dexbotic.infer.trt import (
    assign_device_memory,
    create_execution_context,
    current_build_metadata,
    device_memory_size,
    load_tensorrt,
    resolve_cuda_device,
    resolve_io_names,
    validate_build_metadata,
)
from dexbotic.model.dm05.infer.fast.prefix_mlp_manifest import (
    MANIFEST_NAME,
    prefix_mlp_model_identity,
    prefix_source_model_sha256,
    validate_calibration,
    validate_prefix_buckets,
    validate_prefix_mlp_manifest,
)


class PrefixMLPFP8Backend:
    """Own one engine per layer and one immutable context per layer/bucket.

    All contexts share scratch memory because the owning fast runtime serializes
    requests on CUDA, including requests submitted from different host threads.
    Shape, tensor addresses, engines and contexts remain alive for every captured
    graph. No context is reshaped after capture.
    """

    def __init__(self, vlm, engine_dir, buckets, *, device, vision_engine_identity):
        language_model = vlm.model.language_model
        self.device = resolve_cuda_device(device, "FP8 prefix MLP")
        self.engine_dir = Path(engine_dir).expanduser()
        self.buckets = validate_prefix_buckets(buckets)
        self.hidden_size = int(language_model.config.hidden_size)
        self.layer_count = len(language_model.layers)
        self.engines = {}
        self.contexts = {bucket: {} for bucket in self.buckets}
        self.io = {}
        self.trt = load_tensorrt("FP8 prefix MLP inference")
        self.logger = self.trt.Logger(self.trt.Logger.WARNING)
        self.runtime = self.trt.Runtime(self.logger)
        with torch.cuda.device(self.device), engine_directory_lock(self.engine_dir):
            manifest_path = self.engine_dir / MANIFEST_NAME
            model_identity = prefix_mlp_model_identity(language_model)
            source_sha256 = prefix_source_model_sha256(vlm)
            self.manifest = validate_prefix_mlp_manifest(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                model_identity,
                seq_lens=self.buckets,
                prefix_source_sha256=source_sha256,
            )
            validate_build_metadata(
                self.manifest["build_metadata"], current_build_metadata(self.device)
            )
            calibration_record = self.manifest["calibration"]
            calibration = json.loads(
                read_verified_file(
                    self.engine_dir / calibration_record["file"], calibration_record
                )
            )
            validate_calibration(
                calibration,
                model_identity,
                prefix_source_sha256=source_sha256,
            )
            expected_vision = calibration["vision_engine"]
            if any(
                vision_engine_identity.get(key) != expected_vision[key]
                for key in ("bytes", "sha256")
            ):
                raise ValueError("Loaded vision engine differs from FP8 calibration")
            for index in range(self.layer_count):
                record = self.manifest["engines"][str(index)]
                data = read_verified_file(self.engine_dir / record["engine"], record)
                engine = self.runtime.deserialize_cuda_engine(data)
                if engine is None:
                    raise RuntimeError(f"Cannot deserialize FP8 MLP layer {index}")
                self._validate_engine(engine, index)
                self.engines[index] = engine

            for bucket in self.buckets:
                shape = (1, bucket, self.hidden_size)
                source = torch.empty(shape, dtype=torch.float16, device=self.device)
                output = torch.empty_like(source)
                self.io[bucket] = (source, output)
                for index, engine in self.engines.items():
                    context = create_execution_context(
                        engine, "FP8 prefix MLP", use_external_device_memory=True
                    )
                    if not context.set_input_shape("hidden_states", shape):
                        raise RuntimeError(
                            f"FP8 layer {index} rejected input shape {shape}"
                        )
                    if tuple(context.get_tensor_shape("layer_out")) != shape:
                        raise RuntimeError(
                            f"FP8 layer {index} output shape differs from {shape}"
                        )
                    for name, value in (
                        ("hidden_states", source),
                        ("layer_out", output),
                    ):
                        if not context.set_tensor_address(name, value.data_ptr()):
                            raise RuntimeError(f"FP8 layer {index} cannot bind {name}")
                    self.contexts[bucket][index] = context

            scratch_bytes = max(device_memory_size(e) for e in self.engines.values())
            self.device_memory = torch.empty(
                max(256, scratch_bytes), dtype=torch.uint8, device=self.device
            )
            for contexts in self.contexts.values():
                for index, context in contexts.items():
                    assign_device_memory(
                        context=context,
                        device_memory=self.device_memory,
                        device=self.device,
                        required_size=device_memory_size(self.engines[index]),
                        label=f"FP8 prefix MLP layer {index}",
                    )

        for tensor in [
            self.device_memory,
            *(v for pair in self.io.values() for v in pair),
        ]:
            torch._dynamo.mark_static_address(tensor)

    def _validate_engine(self, engine, index):
        if not hasattr(engine, "get_tensor_profile_shape"):
            raise RuntimeError("FP8 prefix MLP requires the TensorRT 10 tensor API")
        inputs, outputs = resolve_io_names(engine, self.trt)
        if inputs != ["hidden_states"] or outputs != ["layer_out"]:
            raise RuntimeError(f"FP8 layer {index} has incompatible input/output names")
        if engine.num_optimization_profiles != 1:
            raise RuntimeError(f"FP8 layer {index} must have one optimization profile")
        profile = self.manifest["profile"]
        expected = tuple(
            (1, int(profile[f"{name}_seq_len"]), self.hidden_size)
            for name in ("min", "opt", "max")
        )
        allowed_shapes = {(1, -1, self.hidden_size)}
        # TensorRT may specialize the sequence dimension for a single bucket.
        if expected[0] == expected[1] == expected[2]:
            allowed_shapes.add(expected[0])
        for name in inputs + outputs:
            if engine.get_tensor_dtype(name) != self.trt.float16:
                raise RuntimeError(f"FP8 layer {index} must have FP16 I/O: {name}")
            if tuple(engine.get_tensor_shape(name)) not in allowed_shapes:
                raise RuntimeError(
                    f"FP8 layer {index} has an incompatible {name} shape"
                )
            if engine.get_tensor_location(name) != self.trt.TensorLocation.DEVICE:
                raise RuntimeError(f"FP8 layer {index} requires device I/O: {name}")
            if engine.get_tensor_format(name) != self.trt.TensorFormat.LINEAR:
                raise RuntimeError(f"FP8 layer {index} requires linear I/O: {name}")
        actual = tuple(
            tuple(s) for s in engine.get_tensor_profile_shape("hidden_states", 0)
        )
        if actual != expected:
            raise RuntimeError(
                f"FP8 layer {index} profile differs from manifest: "
                f"{actual} != {expected}"
            )

    def forward_into(self, layer_idx, mlp_input, mlp_output):
        bucket = int(mlp_input.shape[1])
        source, output = self.io[bucket]
        if (
            tuple(mlp_input.shape) != tuple(source.shape)
            or tuple(mlp_output.shape) != tuple(output.shape)
            or mlp_input.device != self.device
            or mlp_output.device != self.device
            or mlp_input.dtype != torch.bfloat16
            or mlp_output.dtype != torch.bfloat16
        ):
            raise ValueError(
                "FP8 prefix MLP requires matching static CUDA BF16 buffers"
            )
        source.copy_(mlp_input)
        stream = torch.cuda.current_stream(self.device)
        if not self.contexts[bucket][layer_idx].execute_async_v3(
            stream_handle=stream.cuda_stream
        ):
            raise RuntimeError(f"FP8 prefix MLP layer {layer_idx} execution failed")
        mlp_output.copy_(output)
