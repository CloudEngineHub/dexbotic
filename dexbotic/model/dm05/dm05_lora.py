"""DM05 LoRA helpers."""

import json
import os

import torch

from dexbotic.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration


def _is_lora_checkpoint(model_name_or_path: str | None) -> bool:
    return bool(
        model_name_or_path
        and os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    )


def _read_lora_base_model_path(adapter_path: str) -> str | None:
    adapter_config = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        return None
    with open(adapter_config, encoding="utf-8") as f:
        return json.load(f).get("base_model_name_or_path") or None


def unwrap_dm05_model(model: torch.nn.Module) -> DM05ForConditionalGeneration:
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def _all_linear_target_modules(model: DM05ForConditionalGeneration) -> list[str]:
    output_embeddings = (
        model.get_output_embeddings()
        if hasattr(model, "get_output_embeddings")
        else None
    )
    resolved: set[str] = set()
    for name, module in model.named_modules():
        if not name or not isinstance(module, torch.nn.Linear):
            continue
        if output_embeddings is not None and module is output_embeddings:
            continue
        if name.endswith("lm_head"):
            continue
        parts = name.split(".")
        if parts[-1].isdigit() and len(parts) >= 2:
            resolved.add(".".join(parts[-2:]))
        else:
            resolved.add(parts[-1])
    if not resolved:
        raise ValueError(
            "DM05 LoRA target_modules='all-linear' found no Linear modules."
        )
    return sorted(resolved)


def apply_lora_to_dm05_model(
    model: DM05ForConditionalGeneration,
    base_model_name_or_path: str | None,
) -> torch.nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError("DM05 LoRA requires `peft`") from exc

    action_expert = model.model.action_expert
    modules_to_save = [
        "action_in_proj",
        "action_out_proj",
        "time_mlp_in",
        "time_mlp_out",
        "final_time_modulator",
    ]
    modules_to_save.extend(
        f"input_time_modulators.{idx}"
        for idx in range(len(action_expert.input_time_modulators))
    )
    modules_to_save.extend(
        f"mlp_time_modulators.{idx}"
        for idx in range(len(action_expert.mlp_time_modulators))
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=_all_linear_target_modules(model),
        modules_to_save=modules_to_save,
    )
    peft_config.base_model_name_or_path = base_model_name_or_path or ""
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    for module in model.modules():
        if getattr(module, "_tied_weights_keys", None) is not None:
            module._tied_weights_keys = None
    model = get_peft_model(model, peft_config)
    for module in model.modules():
        if getattr(module, "_tied_weights_keys", None) is not None:
            module._tied_weights_keys = None
    frozen_dtype = next(
        (
            p.dtype
            for p in model.parameters()
            if not p.requires_grad and p.is_floating_point()
        ),
        None,
    )
    if frozen_dtype is not None:
        for param in model.parameters():
            if (
                param.requires_grad
                and param.is_floating_point()
                and param.dtype != frozen_dtype
            ):
                param.data = param.data.to(dtype=frozen_dtype)
    return model


def load_dm05_model_for_inference(
    model_name_or_path: str,
    config_overrides: dict | None = None,
    merge_and_unload: bool = True,
    **from_pretrained_kwargs,
) -> torch.nn.Module:
    def _config(path: str) -> DM05Config:
        config = DM05Config.from_pretrained(path)
        for attr, value in (config_overrides or {}).items():
            if hasattr(config, attr):
                setattr(config, attr, value)
        return config

    if not _is_lora_checkpoint(model_name_or_path):
        return DM05ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            config=_config(model_name_or_path),
            **from_pretrained_kwargs,
        )
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading a LoRA DM05 checkpoint requires `peft`") from exc
    base_model_path = _read_lora_base_model_path(model_name_or_path)
    if not base_model_path:
        raise ValueError(
            "LoRA adapter checkpoint does not record base_model_name_or_path."
        )
    config_source = (
        model_name_or_path
        if os.path.exists(os.path.join(model_name_or_path, "config.json"))
        else base_model_path
    )
    base_model = DM05ForConditionalGeneration.from_pretrained(
        base_model_path, config=_config(config_source), **from_pretrained_kwargs
    )
    model = PeftModel.from_pretrained(base_model, model_name_or_path)
    if not merge_and_unload:
        return model
    return model.merge_and_unload()
