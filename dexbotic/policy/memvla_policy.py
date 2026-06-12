from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token


class MemVLAPolicy(BasePolicy):
    """
    A-family policy for MemVLA.

    MemVLA keeps an internal memory bank that resets at episode start.
    Call reset() at the beginning of each episode; the first subsequent
    select_action call will pass episode_first_frame="True" to the model.

    observation keys:
        "prompt"              : str  — task instruction
        "images"              : list — file paths or PIL Images
        "episode_first_frame" : str  — "True"/"False", optional override
    """

    action_mode = "relative"
    state_used = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._is_first_frame: bool = True

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        text = observation["prompt"]
        raw_images = self._collect_images(observation)
        # allow explicit override from HTTP caller; default to internal state
        episode_first_frame = observation.get(
            "episode_first_frame",
            "True" if self._is_first_frame else "False",
        )

        pil_images = self._load_images(raw_images)
        image_tensor = self._images_to_tensor(pil_images)

        conv = conversation_lib.conv_templates[
            self.model.config.chat_template
        ].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], " ")
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        sc = sampling_config or SamplingConfig()
        inference_args = {
            "cfg_scale": sc.cfg_scale,
            "num_ddim_steps": sc.num_steps,
            "action_norms": self.norm_stats,
        }

        actions = self.model.inference_action(
            input_ids,
            image_tensor,
            episode_first_frame=episode_first_frame,
            inference_args=inference_args,
        )

        # after the first call the memory bank is populated
        self._is_first_frame = False

        return [ActionOutput(actions=np.array(actions))]

    def reset(self) -> None:
        """Mark episode boundary — next select_action will reset the memory bank."""
        self._is_first_frame = True
