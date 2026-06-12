from __future__ import annotations

from typing import Optional

import numpy as np

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.process import GR00TN1Tokenization


class Gr00tN1Policy(BasePolicy):
    """
    A-family policy for GR00T-N1.

    Handles three chat_template variants:
      - "step"       : LLaVA-style with <im_start>/<im_end> markers
      - "qwen2-chat" : Qwen-style per-image <image N> prefixes
      - default      : standard DEFAULT_IMAGE_TOKEN prefix

    observation keys:
        "prompt" : str  — task instruction
        "images" : list — file paths or PIL Images
    """

    action_mode = "relative"
    state_used = False

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        text = observation["prompt"]
        raw_images = self._collect_images(observation)

        pil_images = self._load_images(raw_images)
        image_tensor = self._images_to_tensor(pil_images)

        chat_template = self.model.config.chat_template
        conv = conversation_lib.conv_templates[chat_template].copy()

        if chat_template == "step":
            conv.append_message(
                conv.roles[0],
                text + "<im_start>" + DEFAULT_IMAGE_TOKEN + "<im_end>",
            )
        elif chat_template == "qwen2-chat":
            if len(pil_images) == 1:
                prefix = "<image>\n"
            else:
                prefix = "".join(
                    f"<image {i + 1}><img><image></img>\n"
                    for i in range(len(pil_images))
                )
            conv.append_message(conv.roles[0], prefix + text)
        else:
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            GR00TN1Tokenization.tokenizer_image_token(
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

        actions = self.model.inference_action(input_ids, image_tensor, inference_args)
        return [ActionOutput(actions=np.array(actions))]
