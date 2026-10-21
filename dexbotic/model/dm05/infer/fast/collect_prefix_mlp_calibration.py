#!/usr/bin/env python3
"""Collect static FP8 Prefix MLP ranges from transformed deployment inputs.

Each ``--inputs`` file is a ``torch.save`` dictionary, or a list of dictionaries,
containing ``pixel_values``, ``input_ids``, ``attention_mask`` and
``token_type_ids``. Inputs must come from the deployment preprocessing pipeline;
this command never synthesizes or truncates observations.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch

from dexbotic.infer.artifacts import file_record, write_json_atomic
from dexbotic.infer.calibration import MaxAbsObserver, load_tensor_record_files
from dexbotic.model.dm05.dm05_lora import (
    load_dm05_model_for_inference,
    unwrap_dm05_model,
)
from dexbotic.model.dm05.dm05_utils import mask_history_pad_tokens_in_attention
from dexbotic.model.dm05.infer.fast.prefix_mlp_manifest import (
    PREFIX_MLP_FORMAT_VERSION,
    SPEC,
    prefix_mlp_model_identity,
    prefix_source_model_sha256,
    validate_calibration,
)
from dexbotic.model.dm05.infer.fast.runtime import DM05FastRuntime

_REQUIRED_INPUTS = {"pixel_values", "input_ids", "attention_mask", "token_type_ids"}
_OPTIONAL_INPUTS = {"diffusion_input_noise", "action_mask"}


def _validate_sample(record: dict, *, bucket: int, num_images: int) -> None:
    ids = record["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= bucket:
        raise ValueError(
            f"Expected input_ids [1, 1..{bucket}], got {tuple(ids.shape)}; "
            "inputs are never truncated."
        )
    if ids.dtype not in (torch.int32, torch.int64) or bool((ids < 0).any()):
        raise ValueError("input_ids must contain nonnegative integer token IDs.")
    for key in ("attention_mask", "token_type_ids"):
        value = record[key]
        if value.shape != ids.shape or not bool(((value == 0) | (value == 1)).all()):
            raise ValueError(
                f"{key} must be binary and have the same shape as input_ids."
            )
    if not bool(record["attention_mask"].any()):
        raise ValueError("Calibration sample has no visible tokens.")
    pixels = record["pixel_values"]
    if pixels.ndim != 4 or pixels.shape[:2] != (num_images, 3):
        raise ValueError(
            f"Expected pixel_values [{num_images}, 3, H, W], "
            f"got {tuple(pixels.shape)}."
        )
    if not pixels.is_floating_point() or not bool(torch.isfinite(pixels).all()):
        raise ValueError("pixel_values must contain finite transformed floats.")


def _load_samples(paths: list[Path], *, bucket: int, num_images: int):
    def validate(record, _path, _index):
        _validate_sample(record, bucket=bucket, num_images=num_images)

    samples, provenance, distinct_count = load_tensor_record_files(
        paths,
        required_keys=_REQUIRED_INPUTS,
        optional_keys=_OPTIONAL_INPUTS,
        validate_record=validate,
    )
    if distinct_count < 2:
        raise ValueError(
            "Calibration requires at least two distinct real transformed input samples."
        )
    return samples, provenance, distinct_count


@torch.inference_mode()
def collect_calibration(args: argparse.Namespace) -> dict:
    if args.num_images <= 0 or not 0 < args.bucket <= 1024:
        raise ValueError("Expected positive --num-images and --bucket in 1..1024.")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.diffusion_steps <= 0:
        raise ValueError("--diffusion-steps must be positive.")
    samples, input_files, unique_count = _load_samples(
        args.inputs, bucket=args.bucket, num_images=args.num_images
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Collecting Prefix MLP calibration requires CUDA.")
    if not args.vision_trt_engine_path.is_file():
        raise FileNotFoundError(
            "Build the matching vision engine explicitly before calibration."
        )
    vision_record = file_record(args.vision_trt_engine_path)
    overrides = None if args.chunk_size is None else {"chunk_size": args.chunk_size}
    model = load_dm05_model_for_inference(
        str(args.checkpoint),
        config_overrides=overrides,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    dm05_model = unwrap_dm05_model(model)
    vlm = dm05_model.model.vlm
    language_model = vlm.model.language_model
    model_identity = prefix_mlp_model_identity(language_model)
    source_sha = prefix_source_model_sha256(vlm)
    dm05_model.set_attention_implementation(
        llm_attn_implementation="flex_attention",
        vision_attn_implementation="sdpa",
        action_attn_implementation="sdpa",
        bf16=True,
    )
    dm05_model.enable_gradient_checkpointing(
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
    )
    dm05_model = dm05_model.eval().to("cuda")
    runtime = DM05FastRuntime(
        dm05_model,
        args.vision_trt_engine_path,
        prefix_buckets=(args.bucket,),
        diffusion_steps=args.diffusion_steps,
        overflow_policy="error",
    )
    loaded_vision_identity = runtime.vision_runner.engine_identity
    if any(
        loaded_vision_identity.get(key) != vision_record[key]
        for key in ("bytes", "sha256")
    ):
        raise ValueError(
            "Vision engine changed while calibration was starting; "
            "retry with a stable artifact."
        )
    vision_record = {"file": args.vision_trt_engine_path.name, **loaded_vision_identity}
    observers = {
        str(index): {
            activation: MaxAbsObserver() for activation in ("mlp_input", "down_input")
        }
        for index in range(model_identity["layer_count"])
    }
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise_shape = (1, runtime.suffix_len, runtime.action_dim)
    for sample_index, sample in enumerate(samples):
        noise = sample.get("diffusion_input_noise")
        if noise is None:
            noise = torch.randn(noise_shape, generator=generator, dtype=torch.float32)
        action_mask = sample.get("action_mask")
        if action_mask is None:
            action_mask = torch.ones((1, 1, runtime.action_dim))
        if tuple(noise.shape) != noise_shape or not bool(torch.isfinite(noise).all()):
            raise ValueError(
                "diffusion_input_noise must be finite with exact shape "
                f"{noise_shape}; check --chunk-size."
            )
        expected_mask_shape = (1, 1, runtime.action_dim)
        if tuple(action_mask.shape) != expected_mask_shape or not bool(
            torch.isfinite(action_mask).all()
        ):
            raise ValueError(
                f"action_mask must be finite with exact shape {expected_mask_shape}."
            )
        runtime.infer(
            pixel_values=sample["pixel_values"],
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            token_type_ids=sample["token_type_ids"],
            diffusion_input_noise=noise,
            action_mask=action_mask,
        )
        _bucket, profile = runtime.profiles.select(int(sample["input_ids"].shape[1]))
        if profile is None or profile.graph_runner is None:
            raise RuntimeError("Calibration failed to initialize its graph profile.")
        static_inputs = profile.graph_runner.static_inputs
        visible_mask = mask_history_pad_tokens_in_attention(
            input_ids=static_inputs["input_ids"],
            attention_mask=static_inputs["attention_mask"],
        ).bool()
        handles, observed = [], set()

        def capture(
            index: int, activation: str, observed=observed, visible_mask=visible_mask
        ):
            def hook(_module, positional):
                key = (index, activation)
                if key in observed:
                    raise RuntimeError(
                        "Prefix MLP calibration hook ran more than once for one sample."
                    )
                observed.add(key)
                value = positional[0]
                if value.ndim != 3 or value.shape[:2] != visible_mask.shape:
                    raise ValueError(
                        "Observed MLP activation does not match the prefix "
                        "visible-row mask."
                    )
                observers[str(index)][activation].update(
                    value,
                    row_mask=visible_mask,
                )

            return hook

        for index, layer in enumerate(language_model.layers):
            handles.append(
                layer.mlp.register_forward_pre_hook(capture(index, "mlp_input"))
            )
            handles.append(
                layer.mlp.down_proj.register_forward_pre_hook(
                    capture(index, "down_input")
                )
            )
        try:
            # Replay populates the static inputs, but Python hooks only run on
            # this eager execution of the same bucketed deployment path.
            runtime._run_graph_profile(profile, static_inputs)
            torch.cuda.synchronize(runtime.device)
        finally:
            for handle in handles:
                handle.remove()
        if len(observed) != model_identity["layer_count"] * 2:
            raise RuntimeError(
                "Failed to observe every Prefix MLP activation; "
                "no calibration was written."
            )
        print(
            f"calibration sample {sample_index + 1}/{len(samples)}: "
            f"{int(visible_mask.sum().item())} visible tokens",
            flush=True,
        )
    layers = {
        index: {
            activation: {"visible_rows": observer.as_dict()}
            for activation, observer in layer_observers.items()
        }
        for index, layer_observers in observers.items()
    }
    result = {
        "format_version": PREFIX_MLP_FORMAT_VERSION,
        "spec": SPEC,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": str(args.checkpoint),
        "model_identity": model_identity,
        "prefix_source_sha256": source_sha,
        "vision_engine": vision_record,
        "batch_size": 1,
        "num_images": args.num_images,
        "bucket": args.bucket,
        "sample_count": len(samples),
        "distinct_sample_count": unique_count,
        "input_files": input_files,
        "actual_prefix_lengths": [
            int(sample["input_ids"].shape[1]) for sample in samples
        ],
        "effective_config": {
            "chunk_size": runtime.suffix_len,
            "action_dim": runtime.action_dim,
            "diffusion_steps": args.diffusion_steps,
            "seed": args.seed,
            "history": False,
        },
        "layers": layers,
    }
    validate_calibration(result, model_identity, prefix_source_sha256=source_sha)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-trt-engine-path", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--bucket", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260915)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = collect_calibration(args)
    write_json_atomic(args.output, result)
    print(f"Saved Prefix MLP calibration: {args.output}", flush=True)


if __name__ == "__main__":
    main()
