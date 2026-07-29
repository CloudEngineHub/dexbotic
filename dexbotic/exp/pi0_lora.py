"""PI0 LoRA helpers.

This module keeps PEFT/LoRA-specific wiring out of the core PI0 experiment
configuration. Experiment files should provide explicit LoRA settings instead
of relying on a large environment-variable surface.
"""

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
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM


def resolve_pi0_tokenizer_path(model_name_or_path: str) -> str:
    return resolve_lora_tokenizer_path(model_name_or_path)


@dataclass
class Pi0LoraConfig(Config):
    r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    bias: str = field(default="none")
    target_modules: list[str] | str = field(default="all-linear")
    modules_to_save: list[str] = field(
        default_factory=lambda: [
            "state_proj",
            "action_in_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "action_out_proj",
        ]
    )
    dump_trainable_path: Optional[str] = field(default=None)


def apply_lora_to_pi0_model(
    model: Pi0ForCausalLM,
    lora_config: Pi0LoraConfig,
    base_model_name_or_path: str | None,
) -> torch.nn.Module:
    model, peft_config = apply_peft_lora(
        model,
        lora_config,
        base_model_name_or_path,
        model_label="PI0",
    )
    logger.info(
        "Enabled PI0 LoRA: r={}, alpha={}, dropout={}, target_modules={}, modules_to_save={}",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.lora_dropout,
        lora_config.target_modules,
        lora_config.modules_to_save,
    )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    dump_lora_trainable_summary(model, lora_config, peft_config, model_label="PI0")
    return model


def load_pi0_model_for_inference(
    model_name_or_path: str,
    base_model_name_or_path: str | None = None,
    **from_pretrained_kwargs,
) -> torch.nn.Module:
    if not is_lora_checkpoint(model_name_or_path):
        return Pi0ForCausalLM.from_pretrained(
            model_name_or_path,
            **from_pretrained_kwargs,
        )

    try:
        from peft import PeftModel
    except ImportError as exc:
        raise ImportError("Loading a LoRA PI0 checkpoint requires `peft`") from exc

    base_model_path = base_model_name_or_path or read_lora_base_model_path(
        model_name_or_path
    )
    if not base_model_path:
        raise ValueError(
            "LoRA adapter checkpoint does not record base_model_name_or_path; "
            "pass base_model_name_or_path explicitly."
        )

    logger.info("Loading PI0 LoRA base model from {}", base_model_path)
    base_model = Pi0ForCausalLM.from_pretrained(
        base_model_path,
        **from_pretrained_kwargs,
    )
    logger.info("Loading PI0 LoRA adapter from {}", model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_name_or_path)
    logger.info("Merging PI0 LoRA adapter into the base model for inference")
    model = model.merge_and_unload()
    return model
