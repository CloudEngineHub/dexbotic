from __future__ import annotations

from typing import Any, Optional

import numpy as np

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token


class DiscreteVLAPolicy(BasePolicy):
    """
    A-family policy for DiscreteVLA.

    inference_action receives conv / tokenizer / vocab_size instead of
    cfg_scale / num_ddim_steps — the model handles discrete decoding internally.

    observation keys:
        "prompt" : str  — task instruction
        "images" : list — file paths or PIL Images
    """

    action_mode = "relative"
    state_used = False

    def __init__(self, *args, vocab_size: int = 255, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        text = observation["prompt"]
        raw_images = self._collect_images(observation)

        pil_images = self._load_images(raw_images)
        image_tensor = self._images_to_tensor(pil_images)

        conv = conversation_lib.conv_templates[
            self.model.config.chat_template
        ].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        inference_args = {
            "conv": conv,
            "tokenizer": self.tokenizer,
            "vocab_size": self.vocab_size,
            "action_norms": self.norm_stats,
        }

        actions = self.model.inference_action(input_ids, image_tensor, inference_args)
        return [ActionOutput(actions=np.array(actions))]
