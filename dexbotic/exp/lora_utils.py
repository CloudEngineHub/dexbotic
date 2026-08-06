"""Shared helpers for PEFT-backed LoRA experiment modules."""

import json
import os
from typing import Any, Iterable

import torch
from loguru import logger


def is_lora_checkpoint(model_name_or_path: str | None) -> bool:
    if not model_name_or_path:
        return False
    return os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))


def read_lora_base_model_path(adapter_path: str) -> str | None:
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        return None
    with open(adapter_config, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("base_model_name_or_path") or None


def resolve_lora_tokenizer_path(
    model_name_or_path: str,
    base_model_name_or_path: str | None = None,
) -> str:
    if base_model_name_or_path and is_lora_checkpoint(model_name_or_path):
        return base_model_name_or_path
    if os.path.exists(os.path.join(model_name_or_path, "tokenizer_config.json")):
        return model_name_or_path
    if is_lora_checkpoint(model_name_or_path):
        base_model_path = read_lora_base_model_path(model_name_or_path)
        if base_model_path:
            return base_model_path
    return model_name_or_path


def patch_peft_tied_weights_keys_compat(
    model: torch.nn.Module,
    model_label: str,
) -> None:
    root_tied_keys = getattr(model, "_tied_weights_keys", None)
    if not isinstance(root_tied_keys, dict):
        return

    patched_modules: list[str] = []
    for module_name, module in model.named_modules():
        if module is model:
            continue
        module_tied_keys = getattr(module, "_tied_weights_keys", None)
        if isinstance(module_tied_keys, list):
            module._tied_weights_keys = None
            patched_modules.append(module_name)

    if patched_modules:
        logger.info(
            "Patched PEFT tied-weight metadata for {} {} submodules before LoRA wrap: {}",
            len(patched_modules),
            model_label,
            patched_modules[:20],
        )


def cast_trainable_parameters_to_model_dtype(
    model: torch.nn.Module,
    model_label: str,
) -> None:
    target_dtype = next(
        (
            parameter.dtype
            for parameter in model.parameters()
            if not parameter.requires_grad and parameter.is_floating_point()
        ),
        None,
    )
    if target_dtype is None:
        return

    cast_names: list[str] = []
    for name, parameter in model.named_parameters():
        if (
            not parameter.requires_grad
            or not parameter.is_floating_point()
            or parameter.dtype == target_dtype
        ):
            continue
        parameter.data = parameter.data.to(dtype=target_dtype)
        if parameter.grad is not None:
            parameter.grad.data = parameter.grad.data.to(dtype=target_dtype)
        cast_names.append(name)

    if cast_names:
        logger.info(
            "Cast {} {} LoRA trainable parameters to model dtype {}: {}",
            len(cast_names),
            model_label,
            target_dtype,
            cast_names[:20],
        )


def apply_peft_lora(
    model: torch.nn.Module,
    lora_config,
    base_model_name_or_path: str | None,
    model_label: str,
) -> tuple[torch.nn.Module, Any]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            f"{model_label} LoRA requires `peft` to be installed in the training environment"
        ) from exc

    peft_kwargs = {
        "task_type": TaskType.CAUSAL_LM,
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "bias": lora_config.bias,
        "target_modules": lora_config.target_modules,
        "modules_to_save": lora_config.modules_to_save,
    }
    exclude_modules = getattr(lora_config, "exclude_modules", None)
    if exclude_modules is not None:
        peft_kwargs["exclude_modules"] = exclude_modules
    peft_config = LoraConfig(**peft_kwargs)
    peft_config.base_model_name_or_path = base_model_name_or_path or ""

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    patch_peft_tied_weights_keys_compat(model, model_label)
    model = get_peft_model(model, peft_config)
    if getattr(lora_config, "cast_trainable_to_model_dtype", False):
        cast_trainable_parameters_to_model_dtype(model, model_label)
    return model, peft_config


def _jsonable(value):
    if isinstance(value, set):
        return sorted(value)
    return value


def dump_lora_trainable_summary(
    model: torch.nn.Module,
    lora_config,
    peft_config,
    model_label: str,
    extra_summary: dict[str, Any] | None = None,
    extra_allowed_markers: Iterable[str] | None = None,
) -> None:
    trainable = []
    frozen_total = 0
    trainable_total = 0
    lora_modules = []
    unexpected_trainable = []
    allowed_markers = tuple(
        ["lora_", "modules_to_save"]
        + list(lora_config.modules_to_save)
        + list(extra_allowed_markers or [])
    )
    for name, param in model.named_parameters():
        count = int(param.numel())
        if param.requires_grad:
            trainable_total += count
            trainable.append({"name": name, "shape": list(param.shape), "numel": count})
            if not any(marker in name for marker in allowed_markers):
                unexpected_trainable.append(name)
        else:
            frozen_total += count
        if "lora_" in name:
            lora_modules.append(name)

    summary = {
        "target_modules": _jsonable(peft_config.target_modules),
        "modules_to_save": _jsonable(lora_config.modules_to_save),
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "trainable_total": trainable_total,
        "frozen_total": frozen_total,
        "trainable_ratio": trainable_total / max(trainable_total + frozen_total, 1),
        "trainable_parameters": trainable,
        "lora_parameter_names": lora_modules,
        "unexpected_trainable_parameters": unexpected_trainable,
    }
    if extra_summary:
        summary.update(extra_summary)

    logger.info(
        "{} LoRA trainable summary: trainable={} frozen={} ratio={:.6f}",
        model_label,
        trainable_total,
        frozen_total,
        summary["trainable_ratio"],
    )
    if lora_config.dump_trainable_path:
        dump_dir = os.path.dirname(lora_config.dump_trainable_path)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        with open(lora_config.dump_trainable_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        logger.info(
            "Wrote {} LoRA trainable summary to {}",
            model_label,
            lora_config.dump_trainable_path,
        )
