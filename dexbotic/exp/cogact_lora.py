"""CogACT LoRA helpers for LIBERO SFT experiments."""

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
from dexbotic.model.cogact.cogact_arch import CogACTForCausalLM


@dataclass
class CogACTLoraConfig(Config):
    r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    bias: str = field(default="none")
    target_modules: list[str] | str = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list[str] = field(default_factory=lambda: ["action_head"])
    dump_trainable_path: Optional[str] = field(default=None)


def apply_lora_to_cogact_model(
    model: CogACTForCausalLM,
    lora_config: CogACTLoraConfig,
    base_model_name_or_path: str | None,
) -> torch.nn.Module:
    model, peft_config = apply_peft_lora(
        model,
        lora_config,
        base_model_name_or_path,
        model_label="CogACT",
    )
    logger.info(
        "Enabled CogACT LoRA: r={}, alpha={}, dropout={}, target_modules={}, "
        "modules_to_save={}",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.lora_dropout,
        lora_config.target_modules,
        lora_config.modules_to_save,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    dump_lora_trainable_summary(
        model,
        lora_config,
        peft_config,
        model_label="CogACT",
    )
    return model


def resolve_cogact_tokenizer_path(
    model_name_or_path: str,
    base_model_name_or_path: str | None = None,
) -> str:
    if is_lora_checkpoint(model_name_or_path):
        # Older CogACT adapters may contain incompatible tokenizer metadata.
        base_model_name_or_path = base_model_name_or_path or read_lora_base_model_path(
            model_name_or_path
        )
    return resolve_lora_tokenizer_path(
        model_name_or_path,
        base_model_name_or_path=base_model_name_or_path,
    )


def _ensure_cogact_action_head(
    model: CogACTForCausalLM,
    action_model_type: str,
    action_dim: int,
    chunk_size: int,
) -> None:
    config = model.config
    config.action_model_type = action_model_type
    config.action_dim = action_dim
    config.chunk_size = chunk_size

    inner_model = model.model
    inner_model.config.action_model_type = action_model_type
    inner_model.config.action_dim = action_dim
    inner_model.config.chunk_size = chunk_size
    if getattr(inner_model, "action_head", None) is None:
        action_head = inner_model._build_action_head_module(inner_model.config)
        reference = next(model.parameters(), None)
        if reference is not None:
            action_head.to(device=reference.device, dtype=reference.dtype)

    vision_tower = getattr(inner_model, "mm_vision_tower", None)
    if (
        vision_tower is not None
        and getattr(vision_tower, "_meta_initialized", False)
        and not getattr(vision_tower, "is_loaded", False)
    ):
        vision_tower.load_model()


def load_cogact_lora_model_for_inference(
    model_name_or_path: str,
    base_model_name_or_path: str | None = None,
    action_model_type: str = "DiT-B",
    action_dim: int = 7,
    chunk_size: int = 16,
    merge_on_load: bool = False,
    **from_pretrained_kwargs,
) -> torch.nn.Module:
    if not is_lora_checkpoint(model_name_or_path):
        return CogACTForCausalLM.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading a LoRA CogACT checkpoint requires `peft`") from exc

    base_model_path = base_model_name_or_path or read_lora_base_model_path(
        model_name_or_path
    )
    if not base_model_path:
        raise ValueError(
            "LoRA adapter checkpoint does not record base_model_name_or_path; "
            "pass base_model_name_or_path explicitly."
        )

    logger.info("Loading CogACT LoRA base model from {}", base_model_path)
    base_model = CogACTForCausalLM.from_pretrained(
        base_model_path,
        **from_pretrained_kwargs,
    )
    _ensure_cogact_action_head(
        base_model,
        action_model_type=action_model_type,
        action_dim=action_dim,
        chunk_size=chunk_size,
    )

    logger.info("Loading CogACT LoRA adapter from {}", model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_name_or_path)
    if merge_on_load:
        logger.info("Merging CogACT LoRA adapter into the base model for inference")
        model = model.merge_and_unload()
    model.eval()
    return model
