from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np


@dataclass
class ActionOutput:
    actions: np.ndarray  # [chunk_size, action_dim], physical units, no batch dim


@dataclass
class SamplingConfig:
    num_steps: int = 10       # flow matching / DDIM steps
    cfg_scale: float = 1.5
    seed: Optional[int] = None


@dataclass
class GenSamplingConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 0.95
    do_sample: bool = False
    stop_sequences: Sequence[str] = ()
    return_logprobs: bool = False


@dataclass
class GenerationOutput:
    text: str
    tokens: list
    logprobs: Optional[list] = None
    finish_reason: Literal["stop", "length", "eos"] = "stop"
    metadata: dict = field(default_factory=dict)
