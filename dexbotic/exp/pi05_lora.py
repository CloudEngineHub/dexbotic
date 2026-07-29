"""PI05 LoRA helpers for LIBERO SFT experiments."""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger

from dexbotic.exp.base_exp import Config
from dexbotic.exp.lora_utils import (
    apply_peft_lora,
    dump_lora_trainable_summary,
    is_lora_checkpoint,
    read_lora_base_model_path,
    resolve_lora_tokenizer_path,
)
from dexbotic.model.pi05.pi05_arch import Pi05Config, Pi05ForCausalLM

EXTRA_TRAINABLE_STATE_NAME = "extra_trainable_state.safetensors"
EXTRA_TRAINABLE_META_NAME = "extra_trainable_state.json"


def resolve_pi05_tokenizer_path(model_name_or_path: str) -> str:
    return resolve_lora_tokenizer_path(model_name_or_path)


def set_pi05_runtime_config(model: torch.nn.Module, chunk_size: int = 10) -> None:
    for owner in (model, getattr(model, "model", None)):
        config = getattr(owner, "config", None)
        if config is not None and hasattr(config, "chunk_size"):
            config.chunk_size = chunk_size
    logger.info("Using PI05 Libero runtime chunk_size={}", chunk_size)


def load_pi05_base_model(
    model_name_or_path: str,
    chunk_size: int = 10,
    **from_pretrained_kwargs,
) -> Pi05ForCausalLM:
    config = Pi05Config.from_pretrained(model_name_or_path)
    model = Pi05ForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        **from_pretrained_kwargs,
    )
    set_pi05_runtime_config(model, chunk_size=chunk_size)
    return model


def load_lora_extra_trainable_state(model: torch.nn.Module, adapter_path: str) -> None:
    meta_path = os.path.join(adapter_path, EXTRA_TRAINABLE_META_NAME)
    state_path = os.path.join(adapter_path, EXTRA_TRAINABLE_STATE_NAME)
    fallback_path = state_path + ".pt"
    if (
        not os.path.isfile(meta_path)
        and not os.path.isfile(state_path)
        and not os.path.isfile(fallback_path)
    ):
        return

    try:
        if os.path.isfile(state_path):
            from safetensors.torch import load_file

            state = load_file(state_path, device="cpu")
            loaded_path = state_path
        elif os.path.isfile(fallback_path):
            state = torch.load(fallback_path, map_location="cpu")
            loaded_path = fallback_path
        else:
            logger.warning(
                "extra_trainable_state metadata exists but state file is missing: {}",
                meta_path,
            )
            return
        incompatible = model.load_state_dict(state, strict=False)
        logger.info(
            "Loaded PI05 LoRA extra trainable state from {}; tensors={}; missing={}; unexpected={}",
            loaded_path,
            len(state),
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        if incompatible.unexpected_keys:
            logger.warning(
                "Unexpected keys while loading PI05 LoRA extra trainable state: {}",
                incompatible.unexpected_keys[:20],
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load PI05 LoRA extra trainable state from {adapter_path}"
        ) from exc


@dataclass
class Pi05LoraConfig(Config):
    r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    bias: str = field(default="none")
    target_modules: list[str] | str = field(
        default=(
            r".*(model\.)?(llm|action_expert).*"
            r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        )
    )
    modules_to_save: list[str] = field(
        default_factory=lambda: [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    )
    extra_trainable_regex: Optional[str] = field(
        default=(
            r"action_expert.*\.(input_layernorm|post_attention_layernorm|norm)"
            r"\.dense\.(base_layer\.)?(weight|bias)$"
        )
    )
    extra_trainable_names: list[str] = field(default_factory=list)
    dump_trainable_path: Optional[str] = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_pi05/trainable_summaries/"
            "pi05_lora_sft_libero_50k.json"
        )
    )


def _enable_extra_trainable_parameters(
    model: torch.nn.Module,
    lora_config: Pi05LoraConfig,
) -> list[str]:
    if not lora_config.extra_trainable_regex and not lora_config.extra_trainable_names:
        return []

    pattern = (
        re.compile(lora_config.extra_trainable_regex)
        if lora_config.extra_trainable_regex
        else None
    )
    extra_trainable_names = set(lora_config.extra_trainable_names)
    enabled = []
    for name, parameter in model.named_parameters():
        if (pattern and pattern.search(name)) or name in extra_trainable_names:
            parameter.requires_grad_(True)
            enabled.append(name)

    if enabled:
        logger.info(
            "Enabled {} PI05 LoRA extra trainable full-rank parameters",
            len(enabled),
        )
    return enabled


def apply_lora_to_pi05_model(
    model: Pi05ForCausalLM,
    lora_config: Pi05LoraConfig,
    base_model_name_or_path: str | None,
) -> torch.nn.Module:
    model, peft_config = apply_peft_lora(
        model,
        lora_config,
        base_model_name_or_path,
        model_label="PI05",
    )
    extra_trainable = _enable_extra_trainable_parameters(model, lora_config)
    logger.info(
        "Enabled PI05 LoRA: r={}, alpha={}, dropout={}, target_modules={}, modules_to_save={}, extra_trainable_regex={}",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.lora_dropout,
        lora_config.target_modules,
        lora_config.modules_to_save,
        lora_config.extra_trainable_regex,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    dump_lora_trainable_summary(
        model,
        lora_config,
        peft_config,
        model_label="PI05",
        extra_summary={
            "extra_trainable_regex": lora_config.extra_trainable_regex,
            "extra_trainable_parameter_names": extra_trainable,
        },
        extra_allowed_markers=extra_trainable,
    )
    return model


def load_pi05_lora_model_for_inference(
    model_name_or_path: str,
    base_model_name_or_path: str | None = None,
    merge_on_load: bool = True,
    chunk_size: int = 10,
    **from_pretrained_kwargs,
) -> torch.nn.Module:
    if not is_lora_checkpoint(model_name_or_path):
        return load_pi05_base_model(
            model_name_or_path,
            chunk_size=chunk_size,
            **from_pretrained_kwargs,
        )

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading a LoRA PI05 checkpoint requires `peft`") from exc

    base_model_path = base_model_name_or_path or read_lora_base_model_path(
        model_name_or_path
    )
    if not base_model_path:
        raise ValueError(
            "LoRA adapter checkpoint does not record base_model_name_or_path; "
            "pass base_model_name_or_path explicitly."
        )

    logger.info("Loading PI05 LoRA base model from {}", base_model_path)
    base_model = load_pi05_base_model(
        base_model_path,
        chunk_size=chunk_size,
        **from_pretrained_kwargs,
    )
    logger.info("Loading PI05 LoRA adapter from {}", model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_name_or_path)
    load_lora_extra_trainable_state(model, model_name_or_path)
    if merge_on_load:
        logger.info("Merging PI05 LoRA adapter into the base model for inference")
        model = model.merge_and_unload()
    set_pi05_runtime_config(model, chunk_size=chunk_size)
    return model
