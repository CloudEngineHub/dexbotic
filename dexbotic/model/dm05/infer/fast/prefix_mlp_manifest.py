"""Model identity, engine profiles and validation for FP8 Prefix MLP manifests."""

from __future__ import annotations

import functools
import hashlib
import math

from dexbotic.infer.artifacts import canonical_json, validate_file_record

PREFIX_MLP_FORMAT_VERSION = 5
SPEC = "fp8_e4m3_per_tensor_max"
MANIFEST_NAME = "prefix_mlp_quant_manifest.json"


def _positive_int(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def validate_prefix_buckets(buckets) -> tuple[int, ...]:
    buckets = tuple(buckets)
    if (
        not buckets
        or any(type(b) is not int or b <= 0 for b in buckets)
        or buckets != tuple(sorted(set(buckets)))
    ):
        raise ValueError(
            "prefix_buckets must be nonempty, positive and strictly increasing"
        )
    return buckets


def prefix_mlp_model_identity(language_model) -> dict:
    """Hash the actual loaded MLP tensors, including their numerical dtype.

    The identity is computed during engine preparation and checked against the
    loaded model before enabling FP8 inference. Checkpoint path strings are
    provenance only: merged adapters and in-place weight changes alter identity.
    """
    import torch

    config = language_model.config
    hidden = _positive_int(getattr(config, "hidden_size", None), "hidden_size")
    intermediate = _positive_int(
        getattr(config, "intermediate_size", None), "intermediate_size"
    )
    count = _positive_int(getattr(config, "num_hidden_layers", None), "layer_count")
    if getattr(config, "model_type", None) != "gemma3_text":
        raise ValueError("FP8 Prefix MLP currently supports Gemma3 text MLPs only.")
    if getattr(config, "hidden_activation", None) != "gelu_pytorch_tanh":
        raise ValueError("FP8 Prefix MLP requires gelu_pytorch_tanh activation.")
    if len(language_model.layers) != count:
        raise ValueError("Prefix MLP layer count disagrees with the loaded config.")
    records = {}
    expected_shapes = {
        "gate_proj": (intermediate, hidden),
        "up_proj": (intermediate, hidden),
        "down_proj": (hidden, intermediate),
    }
    for index, layer in enumerate(language_model.layers):
        mlp = layer.mlp
        if (type(mlp).__module__, type(mlp).__name__) != (
            "transformers.models.gemma3.modeling_gemma3",
            "Gemma3MLP",
        ):
            raise ValueError(f"Unsupported Prefix MLP implementation at layer {index}.")
        activation = mlp.act_fn
        tanh_gelu = (type(activation).__module__, type(activation).__name__) == (
            "transformers.activations",
            "PytorchGELUTanh",
        )
        if (type(activation).__module__, type(activation).__name__) == (
            "transformers.activations",
            "GELUTanh",
        ):
            # Transformers 5.3 renamed the module; its default implementation
            # delegates to the same native tanh GELU through a partial.
            implementation = activation.act
            tanh_gelu = (
                isinstance(implementation, functools.partial)
                and implementation.func is torch.nn.functional.gelu
                and not implementation.args
                and implementation.keywords == {"approximate": "tanh"}
            )
        tanh_gelu |= (
            type(activation) is torch.nn.GELU and activation.approximate == "tanh"
        )
        tanh_gelu |= (
            isinstance(activation, functools.partial)
            and activation.func is torch.nn.functional.gelu
            and not activation.args
            and activation.keywords == {"approximate": "tanh"}
        )
        if not tanh_gelu:
            raise ValueError(f"Layer {index} does not execute tanh-approximated GELU.")
        weights = {}
        for name, shape in expected_shapes.items():
            projection = getattr(mlp, name)
            if type(projection) is not torch.nn.Linear or projection.bias is not None:
                raise ValueError(f"Layer {index} {name} must be a bias-free Linear.")
            value = projection.weight.detach()
            if tuple(value.shape) != shape or value.dtype != torch.bfloat16:
                raise ValueError(
                    f"Layer {index} {name} expected BF16 {shape}, "
                    f"got {value.dtype} {tuple(value.shape)}."
                )
            value = value.to(device="cpu").contiguous()
            if not torch.isfinite(value).all().item():
                raise ValueError(f"Layer {index} {name} contains non-finite weights.")
            weights[name] = {
                "shape": list(shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(
                    value.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        records[str(index)] = weights
    identity = {
        "model_type": "gemma3_text",
        "activation": "gelu_pytorch_tanh",
        "bias": False,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "layer_count": count,
        "weights": records,
    }
    identity["sha256"] = hashlib.sha256(canonical_json(identity)).hexdigest()
    return identity


def prefix_source_model_sha256(vlm) -> str:
    """Bind engines to all loaded upstream weights and semantic config.

    Attention implementation names, loader paths and dtype labels may change
    when attaching the fast runtime; tensor dtypes themselves remain hashed.
    """
    import torch

    def semantic_config(value):
        if isinstance(value, dict):
            return {
                str(key): semantic_config(item)
                for key, item in value.items()
                if not str(key).startswith("_")
                and key
                not in (
                    "transformers_version",
                    "dtype",
                    "torch_dtype",
                    "attn_implementation",
                )
            }
        if isinstance(value, (list, tuple)):
            return [semantic_config(item) for item in value]
        return value

    digest = hashlib.sha256(canonical_json(semantic_config(vlm.config.to_dict())))
    for name, tensor in sorted(vlm.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(canonical_json([name, list(value.shape), str(value.dtype)]))
        # memoryview avoids a second full-size allocation for large embeddings.
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def validate_model_identity(recorded: dict, actual: dict) -> None:
    if not isinstance(recorded, dict) or recorded != actual:
        raise ValueError(
            "FP8 Prefix MLP model identity mismatch: loaded weights, shapes, "
            "dtype or activation differ from the built model. Rebuild the engines."
        )


def read_activation_amax(calibration: dict, layer_idx: int, activation: str) -> float:
    try:
        stats = calibration["layers"][str(layer_idx)][activation]["visible_rows"]
        value = float(stats["max_abs"])
        _positive_int(stats["rows"], "calibration visible rows")
        _positive_int(stats["elements"], "calibration elements")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Missing valid calibration for layer {layer_idx} {activation}."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"Calibration layer {layer_idx} {activation} range must be "
            "finite and positive."
        )
    return value


def validate_calibration(
    calibration: dict, model_identity: dict, *, prefix_source_sha256: str
) -> None:
    if (
        not isinstance(calibration, dict)
        or calibration.get("format_version") != PREFIX_MLP_FORMAT_VERSION
    ):
        raise ValueError("Unsupported Prefix MLP calibration format version.")
    if calibration.get("spec") != SPEC or calibration.get("batch_size") != 1:
        raise ValueError(
            "Prefix MLP calibration requires this FP8 recipe and batch size one."
        )
    validate_model_identity(calibration.get("model_identity"), model_identity)
    validate_file_record(calibration.get("vision_engine"), filename_key="file")
    if (
        _positive_int(calibration.get("distinct_sample_count"), "distinct_sample_count")
        < 2
    ):
        raise ValueError(
            "Calibration requires at least two distinct real input samples."
        )
    if _positive_int(calibration.get("sample_count"), "sample_count") < 2:
        raise ValueError(
            "Calibration requires at least two real transformed input samples."
        )
    if calibration.get("prefix_source_sha256") != prefix_source_sha256:
        raise ValueError("Calibration upstream VLM weight/config identity mismatch.")
    count = model_identity["layer_count"]
    if set(calibration.get("layers", {})) != {str(i) for i in range(count)}:
        raise ValueError("Calibration layer set does not match the model.")
    for index in range(count):
        for activation in ("mlp_input", "down_input"):
            read_activation_amax(calibration, index, activation)


def validate_prefix_mlp_manifest(
    manifest: dict, model_identity: dict, seq_lens=None, *, prefix_source_sha256: str
) -> dict:
    """Validate a calibrated static-scale Prefix MLP manifest."""
    if (
        not isinstance(manifest, dict)
        or manifest.get("format_version") != PREFIX_MLP_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported Prefix MLP manifest version; rebuild with the current builder."
        )
    if manifest.get("spec") != SPEC:
        raise ValueError("Unsupported Prefix MLP quantization recipe.")
    validate_model_identity(manifest.get("model_identity"), model_identity)
    if manifest.get("prefix_source_sha256") != prefix_source_sha256:
        raise ValueError("FP8 Prefix MLP upstream VLM weight/config identity mismatch.")
    profile = manifest.get("profile")
    if not isinstance(profile, dict):
        raise ValueError("Missing Prefix MLP engine profile.")
    lengths = tuple(
        _positive_int(profile.get(f"{key}_seq_len"), f"{key}_seq_len")
        for key in ("min", "opt", "max")
    )
    if lengths != tuple(sorted(lengths)):
        raise ValueError(
            "Prefix MLP engine requires min <= opt <= max sequence length."
        )
    count = model_identity["layer_count"]
    if (
        profile.get("batch_size") != 1
        or profile.get("hidden_size") != model_identity["hidden_size"]
        or profile.get("layer_indices") != list(range(count))
    ):
        raise ValueError("Prefix MLP engine profile does not match the loaded model.")
    if seq_lens is not None:
        for length in validate_prefix_buckets(seq_lens):
            if not lengths[0] <= length <= lengths[2]:
                raise ValueError(
                    f"Prefix bucket {length} is outside engine profile {lengths}."
                )
    records = manifest.get("engines")
    if not isinstance(records, dict) or set(records) != {str(i) for i in range(count)}:
        raise ValueError(
            "Prefix MLP manifest must list every model layer exactly once."
        )
    filenames = set()
    for index in range(count):
        record = records[str(index)]
        validate_file_record(record, filename_key="engine")
        if record["engine"] in filenames:
            raise ValueError("Prefix MLP engine filenames must be unique.")
        filenames.add(record["engine"])
        for key in ("input_range", "down_range"):
            value = record.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"Prefix MLP {key} must be finite and positive.")
    validate_file_record(manifest.get("calibration"), filename_key="file")
    if not isinstance(manifest.get("build_metadata"), dict):
        raise ValueError("Missing FP8 build environment metadata.")
    return manifest
