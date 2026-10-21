#!/usr/bin/env python3
"""Build, publish and discover checkpoint-local vision and FP8 Prefix MLP engines.

By default, prepare all engines:
    python -m dexbotic.model.dm05.infer.fast.build_prefix_mlp_trt
        --checkpoint /path/to/model --num-images 3
        --calibration-inputs /path/to/calibration_inputs.pt

Use the build-prefix-mlp subcommand to build only MLP engines against an existing
calibration file. Each GPU stage runs in its own process. FP8 changes model
numerics; validate task success before deploying these engines.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dexbotic.infer.artifacts import (
    cleanup_engine_directory,
    engine_directory_lock,
    file_record,
    publish_engine_generation,
    read_verified_file,
    write_json_atomic,
)
from dexbotic.model.dm05.infer.fast.prefix_mlp_manifest import (
    MANIFEST_NAME,
    PREFIX_MLP_FORMAT_VERSION,
    SPEC,
    prefix_mlp_model_identity,
    prefix_source_model_sha256,
    read_activation_amax,
    validate_calibration,
    validate_prefix_buckets,
    validate_prefix_mlp_manifest,
)

logger = logging.getLogger(__name__)
PREPARED_FP8_FORMAT_VERSION = 4


def prepared_fp8_engines_root(checkpoint: str | Path) -> Path:
    return Path(checkpoint).expanduser() / ".dexbotic" / "fast_fp8"


def find_prepared_fp8_engines(checkpoint) -> tuple[Path, Path] | None:
    """Return verified vision/MLP paths, or ``None`` for missing or damaged files.

    Shape, model and environment compatibility are checked when loading engines.
    """
    root = prepared_fp8_engines_root(checkpoint)
    metadata_path = root / "prepared.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        if not isinstance(metadata, dict):
            raise ValueError("prepared metadata must be an object")
        if metadata.get("format_version") != PREPARED_FP8_FORMAT_VERSION:
            raise ValueError("unsupported prepared engine metadata version")
        generation = metadata["generation"]
        if (
            not isinstance(generation, str)
            or not generation.startswith("generation-")
            or Path(generation).name != generation
        ):
            raise ValueError("invalid prepared generation")
        generation_dir = root / generation
        if generation_dir.is_symlink():
            raise ValueError("prepared generation cannot be a symlink")
        vision_engine_path = generation_dir / "vision.engine"
        read_verified_file(vision_engine_path, metadata["vision"])
        prefix_engine_dir = generation_dir / "prefix_mlp"
        read_verified_file(
            prefix_engine_dir / MANIFEST_NAME,
            metadata["prefix_manifest"],
        )
        return vision_engine_path, prefix_engine_dir
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            "Ignoring invalid DM05 prepared FP8 engines at %s: %s",
            root,
            exc,
        )
        return None


def prepare(
    *,
    checkpoint: str | Path,
    num_images: int,
    prefix_buckets,
    calibration_inputs,
    chunk_size: int | None = None,
    diffusion_steps: int = 10,
    seed: int = 20260915,
) -> None:
    """Calibrate, build, and atomically publish a vision/MLP engine pair."""
    if type(num_images) is not int or num_images <= 0:
        raise ValueError("num_images must be a positive integer")
    buckets = validate_prefix_buckets(prefix_buckets)
    calibration_inputs = tuple(Path(path).expanduser() for path in calibration_inputs)
    if not calibration_inputs:
        raise ValueError("calibration_inputs must contain at least one tensor file")
    for path in calibration_inputs:
        if not path.is_file():
            raise FileNotFoundError(f"Calibration input file does not exist: {path}")
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    if type(diffusion_steps) is not int or diffusion_steps <= 0:
        raise ValueError("diffusion_steps must be a positive integer")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    checkpoint = Path(checkpoint).expanduser()
    root = prepared_fp8_engines_root(checkpoint)
    root.mkdir(parents=True, exist_ok=True)
    generation = "generation-" + uuid.uuid4().hex
    staging = root / (".building-" + generation)
    staging.mkdir()
    module = "dexbotic.model.dm05.infer.fast."

    def run(name, *arguments):
        subprocess.run(
            [sys.executable, "-m", module + name, *map(str, arguments)], check=True
        )

    try:
        run(
            "build_vision_trt",
            "--checkpoint",
            checkpoint,
            "--num-images",
            num_images,
            "--onnx-path",
            staging / "vision.onnx",
            "--engine-path",
            staging / "vision.engine",
        )
        run(
            "collect_prefix_mlp_calibration",
            "--checkpoint",
            checkpoint,
            "--vision-trt-engine-path",
            staging / "vision.engine",
            "--inputs",
            *calibration_inputs,
            "--output",
            staging / "calibration.json",
            "--num-images",
            num_images,
            "--bucket",
            max(buckets),
            "--diffusion-steps",
            diffusion_steps,
            "--seed",
            seed,
            *(("--chunk-size", chunk_size) if chunk_size is not None else ()),
        )
        run(
            "build_prefix_mlp_trt",
            "build-prefix-mlp",
            "--checkpoint",
            checkpoint,
            "--calibration-json",
            staging / "calibration.json",
            "--output-dir",
            staging / "prefix_mlp",
            "--min-seq-len",
            min(buckets),
            "--opt-seq-len",
            buckets[min(1, len(buckets) - 1)],
            "--max-seq-len",
            max(buckets),
        )
        ready = {
            "format_version": PREPARED_FP8_FORMAT_VERSION,
            "generation": generation,
            "vision": file_record(staging / "vision.engine"),
            "prefix_manifest": file_record(staging / "prefix_mlp" / MANIFEST_NAME),
        }
        # Immutable generations let current readers finish while a new one is
        # built. The atomic ready-file replacement is the only publication point.
        with engine_directory_lock(root, exclusive=True):
            staging.rename(root / generation)
            write_json_atomic(root / "prepared.json", ready)
        print(
            f"Prepared native DM05 fast engines: {root / 'prepared.json'}", flush=True
        )
    finally:
        cleanup_engine_directory(staging, purpose="incomplete FP8 preparation")


def build_layer_engine(
    *,
    engine_path: Path,
    mlp,
    hidden_size: int,
    input_amax: float,
    down_amax: float,
    min_seq_len: int,
    opt_seq_len: int,
    max_seq_len: int,
    workspace_gb: float,
) -> None:
    from dexbotic.infer.trt import (
        add_static_fp8_linear_weight_qdq,
        add_static_fp8_qdq,
        fp8_scale_from_amax,
        load_tensorrt,
    )

    trt = load_tensorrt("Building FP8 Prefix MLP TensorRT engines")
    if not hasattr(trt, "fp8"):
        raise RuntimeError("The installed TensorRT version does not support FP8.")
    if not hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        raise RuntimeError("Building FP8 Prefix MLP requires strongly typed TensorRT.")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    hidden_states = network.add_input(
        "hidden_states",
        trt.float16,
        (1, -1, hidden_size),
    )
    if hidden_states is None:
        raise RuntimeError("Failed to add Prefix MLP hidden_states input.")

    keepalive = []
    mlp_input = add_static_fp8_qdq(
        trt=trt,
        network=network,
        tensor=hidden_states,
        scale=fp8_scale_from_amax(input_amax),
        name="mlp_input",
        keepalive=keepalive,
    )
    gate_weight = add_static_fp8_linear_weight_qdq(
        trt=trt,
        network=network,
        weight=mlp.gate_proj.weight,
        name="gate_proj",
        keepalive=keepalive,
    )
    up_weight = add_static_fp8_linear_weight_qdq(
        trt=trt,
        network=network,
        weight=mlp.up_proj.weight,
        name="up_proj",
        keepalive=keepalive,
    )
    down_weight = add_static_fp8_linear_weight_qdq(
        trt=trt,
        network=network,
        weight=mlp.down_proj.weight,
        name="down_proj",
        keepalive=keepalive,
    )

    gate = network.add_matrix_multiply(
        mlp_input,
        trt.MatrixOperation.NONE,
        gate_weight,
        trt.MatrixOperation.NONE,
    )
    up = network.add_matrix_multiply(
        mlp_input,
        trt.MatrixOperation.NONE,
        up_weight,
        trt.MatrixOperation.NONE,
    )
    if gate is None or up is None:
        raise RuntimeError("Failed to add Prefix MLP gate/up GEMMs.")
    gate.name = "gate_proj"
    up.name = "up_proj"
    activation = network.add_activation(
        gate.get_output(0),
        trt.ActivationType.GELU_TANH,
    )
    if activation is None:
        raise RuntimeError("Failed to add Prefix MLP GELU activation.")
    activation.name = "gelu_tanh"
    product = network.add_elementwise(
        activation.get_output(0),
        up.get_output(0),
        trt.ElementWiseOperation.PROD,
    )
    if product is None:
        raise RuntimeError("Failed to add Prefix MLP gate/up product.")
    product.name = "gate_up_product"
    down_input = add_static_fp8_qdq(
        trt=trt,
        network=network,
        tensor=product.get_output(0),
        scale=fp8_scale_from_amax(down_amax),
        name="down_input",
        keepalive=keepalive,
    )
    down = network.add_matrix_multiply(
        down_input,
        trt.MatrixOperation.NONE,
        down_weight,
        trt.MatrixOperation.NONE,
    )
    if down is None:
        raise RuntimeError("Failed to add Prefix MLP down GEMM.")
    down.name = "down_proj"
    output = down.get_output(0)
    output.name = "layer_out"
    network.mark_output(output)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30)),
    )
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    profile = builder.create_optimization_profile()
    min_shape = (1, min_seq_len, hidden_size)
    opt_shape = (1, opt_seq_len, hidden_size)
    max_shape = (1, max_seq_len, hidden_size)
    profile.set_shape("hidden_states", min_shape, opt_shape, max_shape)
    actual_shapes = tuple(
        tuple(int(dim) for dim in shape) for shape in profile.get_shape("hidden_states")
    )
    if actual_shapes != (min_shape, opt_shape, max_shape):
        raise RuntimeError(
            "TensorRT rejected Prefix MLP profile "
            f"min={min_shape}, opt={opt_shape}, max={max_shape}."
        )
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Failed to add the Prefix MLP optimization profile.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build FP8 Prefix MLP TensorRT engine.")
    engine_path.write_bytes(bytes(serialized))


def build_prefix_mlp_engines(args: argparse.Namespace) -> Path:
    import torch

    from dexbotic.infer.trt import current_build_metadata

    with torch.inference_mode():
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Building FP8 Prefix MLP TensorRT engines requires CUDA."
            )
        lengths = (args.min_seq_len, args.opt_seq_len, args.max_seq_len)
        if lengths != tuple(sorted(lengths)) or args.min_seq_len <= 0:
            raise ValueError(f"Expected positive min <= opt <= max, got {lengths}.")
        if not math.isfinite(args.workspace_gb) or args.workspace_gb <= 0:
            raise ValueError("--workspace-gb must be finite and positive.")

        calibration_bytes = args.calibration_json.read_bytes()
        calibration = json.loads(calibration_bytes)
        if not isinstance(calibration, dict):
            raise ValueError("Calibration JSON must be an object.")
        effective_config = calibration.get("effective_config", {})
        overrides = (
            {"chunk_size": effective_config["chunk_size"]}
            if "chunk_size" in effective_config
            else None
        )
        from dexbotic.model.dm05.dm05_lora import (
            load_dm05_model_for_inference,
            unwrap_dm05_model,
        )

        model = load_dm05_model_for_inference(
            str(args.checkpoint),
            config_overrides=overrides,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        ).eval()
        vlm = unwrap_dm05_model(model).model.vlm
        language_model = vlm.model.language_model
        model_identity = prefix_mlp_model_identity(language_model)
        source_sha = prefix_source_model_sha256(vlm)
        validate_calibration(
            calibration, model_identity, prefix_source_sha256=source_sha
        )
        metadata = current_build_metadata()
        engine_dir = args.output_dir.absolute()
        if engine_dir.is_symlink():
            raise ValueError("--output-dir cannot be a symlink.")

        # Hold one lock across an entire generation. Readers hold the corresponding
        # shared lock until every engine has been verified and deserialized.
        with engine_directory_lock(engine_dir, exclusive=True):
            if engine_dir.exists():
                if not engine_dir.is_dir():
                    raise ValueError("--output-dir must be a directory.")
                if not args.force:
                    raise FileExistsError(
                        f"Engine directory exists: {engine_dir}; "
                        "use --force to rebuild."
                    )
                if (
                    any(engine_dir.iterdir())
                    and not (engine_dir / MANIFEST_NAME).is_file()
                ):
                    raise ValueError(
                        "Refusing to replace a directory without an MLP manifest."
                    )
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{engine_dir.name}.building-", dir=engine_dir.parent
                )
            )
            backup = (
                engine_dir.parent / f".{engine_dir.name}.previous-{uuid.uuid4().hex}"
            )
            try:
                calibration_path = staging / "calibration.json"
                calibration_path.write_bytes(calibration_bytes)
                records = {}
                for index, layer in enumerate(language_model.layers):
                    input_amax = read_activation_amax(calibration, index, "mlp_input")
                    down_amax = read_activation_amax(calibration, index, "down_input")
                    engine_path = staging / f"layer_{index:02d}.engine"
                    build_layer_engine(
                        engine_path=engine_path,
                        mlp=layer.mlp,
                        hidden_size=model_identity["hidden_size"],
                        input_amax=input_amax,
                        down_amax=down_amax,
                        min_seq_len=args.min_seq_len,
                        opt_seq_len=args.opt_seq_len,
                        max_seq_len=args.max_seq_len,
                        workspace_gb=args.workspace_gb,
                    )
                    records[str(index)] = {
                        **file_record(engine_path, filename_key="engine"),
                        "input_range": input_amax,
                        "down_range": down_amax,
                    }
                    print(
                        f"layer {index}: built {engine_path.name} "
                        f"({engine_path.stat().st_size / 2**20:.1f} MiB)",
                        flush=True,
                    )
                manifest = {
                    "format_version": PREFIX_MLP_FORMAT_VERSION,
                    "spec": SPEC,
                    "created_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                    "checkpoint": str(args.checkpoint),
                    "model_identity": model_identity,
                    "prefix_source_sha256": source_sha,
                    "profile": {
                        "batch_size": 1,
                        "hidden_size": model_identity["hidden_size"],
                        "min_seq_len": args.min_seq_len,
                        "opt_seq_len": args.opt_seq_len,
                        "max_seq_len": args.max_seq_len,
                        "layer_indices": list(range(model_identity["layer_count"])),
                    },
                    "calibration": file_record(calibration_path),
                    "build_metadata": metadata,
                    "engines": records,
                }
                validate_prefix_mlp_manifest(
                    manifest, model_identity, prefix_source_sha256=source_sha
                )
                write_json_atomic(staging / MANIFEST_NAME, manifest)
                publish_engine_generation(staging, engine_dir, backup)
            finally:
                cleanup_engine_directory(
                    staging, purpose="incomplete engine staging directory"
                )
        return engine_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    prefix_only = bool(argv and argv[0] == "build-prefix-mlp")
    parser = argparse.ArgumentParser(
        description=(
            "Build FP8 Prefix MLP engines from an existing calibration file."
            if prefix_only
            else __doc__
        ),
        prog=f"{Path(sys.argv[0]).name} build-prefix-mlp" if prefix_only else None,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    if prefix_only:
        parser.add_argument("--calibration-json", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--min-seq-len", type=int, default=576)
        parser.add_argument("--opt-seq-len", type=int, default=768)
        parser.add_argument("--max-seq-len", type=int, default=1024)
        parser.add_argument("--workspace-gb", type=float, default=8.0)
        parser.add_argument("--force", action="store_true")
    else:
        parser.add_argument("--num-images", type=int, required=True)
        parser.add_argument("--calibration-inputs", type=Path, nargs="+", required=True)
        parser.add_argument(
            "--prefix-buckets", type=int, nargs="+", default=[576, 704, 768, 896, 1024]
        )
        parser.add_argument("--chunk-size", type=int)
        parser.add_argument("--diffusion-steps", type=int, default=10)
        parser.add_argument("--seed", type=int, default=20260915)
    parser.set_defaults(command="build-prefix-mlp" if prefix_only else "prepare")
    return parser.parse_args(argv[1:] if prefix_only else argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build-prefix-mlp":
        print(
            f"Built FP8 Prefix MLP engines: {build_prefix_mlp_engines(args)}",
            flush=True,
        )
    else:
        prepare(
            checkpoint=args.checkpoint,
            num_images=args.num_images,
            prefix_buckets=args.prefix_buckets,
            calibration_inputs=args.calibration_inputs,
            chunk_size=args.chunk_size,
            diffusion_steps=args.diffusion_steps,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
