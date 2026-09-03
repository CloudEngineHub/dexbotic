"""DM05 model for Dexbotic (Gemma3 VLM + suffix-only action expert)."""

from transformers import AutoConfig, AutoModelForCausalLM

from .dm05_arch import DM05Config, DM05ForConditionalGeneration
from .dm05_lora import (
    apply_lora_to_dm05_model,
    load_dm05_model_for_inference,
    unwrap_dm05_model,
)

AutoConfig.register("dm05", DM05Config, exist_ok=True)
AutoModelForCausalLM.register(DM05Config, DM05ForConditionalGeneration, exist_ok=True)

__all__ = [
    "DM05Config",
    "DM05ForConditionalGeneration",
    "apply_lora_to_dm05_model",
    "load_dm05_model_for_inference",
    "unwrap_dm05_model",
]
