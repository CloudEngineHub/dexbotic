from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from dexbotic.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig
from dexbotic.tokenization import conversation as conversation_lib
from dexbotic.tokenization.tokenization import tokenizer_image_token


class OFTPolicy(BasePolicy):
    """
    A-family policy for OFT (continuous action).

    observation keys (named camera format):
        "image/{cam}": PIL Image | ndarray | path str   — single sample
                       list of the above                — batch
        "prompt":      str (broadcast) | list[str]      — task instruction
        "state":       ndarray | list[ndarray]          — robot state (optional)

    camera_order controls which image/* keys are used and in what order.
    """

    action_mode = "relative"
    state_used = False
    state_required = False

    def get_capabilities(self) -> dict:
        caps = super().get_capabilities()
        use_proprio = bool(getattr(self.model.config, "use_proprio", False))
        caps["state"] = {
            "used": use_proprio,
            "required": use_proprio,
            "dim": getattr(self.model.config, "proprio_dim", None)
            if use_proprio
            else None,
        }
        return caps

    def _normalize_prompt(self, prompt: str) -> str:
        """Post-process the rendered conv prompt. Override in subclasses if needed."""
        return prompt

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        batch_size = self._infer_batch_size(observation)
        obs = self._normalize_obs(observation, batch_size)
        return [
            self._infer_single({k: v[i] for k, v in obs.items()}, sampling_config)
            for i in range(batch_size)
        ]

    def _infer_single(
        self,
        obs_single: dict,
        sampling_config: Optional[SamplingConfig],
    ) -> ActionOutput:
        text = obs_single["prompt"]
        state = obs_single.get("state")

        img_keys = sorted(
            (k for k in obs_single if k.startswith("image/")),
            key=lambda k: int(k.split("/", 1)[1]),
        )
        pil_images = [self._load_images([obs_single[k]])[0] for k in img_keys]

        image_tensor = self._images_to_tensor(pil_images)

        conv = conversation_lib.conv_templates[
            self.model.config.chat_template
        ].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + text)
        conv.append_message(conv.roles[1], " ")
        prompt = self._normalize_prompt(conv.get_prompt())

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

        if state is not None:
            inference_args["states"] = torch.tensor(
                np.array(state, dtype=np.float32),
                dtype=self.model.dtype,
                device=self.model.device,
            ).reshape(1, -1)

        actions = self.model.inference_action(input_ids, image_tensor, inference_args)
        return ActionOutput(actions=np.array(actions))


class OFTDiscretePolicy(OFTPolicy):
    """
    A-family policy for OFT-Discrete.

    Differences from OFTPolicy:
    - removes double-spaces introduced by the conv template
    - flips the gripper bit: model output <0.5 → open (1.0), ≥0.5 → close (-1.0)
    """

    def _normalize_prompt(self, prompt: str) -> str:
        return prompt.replace("  ", " ")

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        results = super().select_action(observation, sampling_config)
        flipped = []
        for out in results:
            actions = out.actions.copy()
            actions[:, -1] = np.where(actions[:, -1] < 0.5, 1.0, -1.0)
            flipped.append(ActionOutput(actions=actions))
        return flipped
