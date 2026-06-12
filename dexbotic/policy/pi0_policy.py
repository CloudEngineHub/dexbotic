from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig


class Pi0Policy(BasePolicy):
    """
    B-family policy for Pi0 (and Pi05, HybridPi05, DM0).

    observation keys (named camera format):
        "image/{cam}": PIL Image | ndarray | path str   — single sample
                       list of the above                — batch
        "prompt":      str (broadcast) | list[str]      — task instruction
        "state":       ndarray [action_dim] (optional)  — robot joint state

    camera_order controls which image/* keys are used and in what order.

    Requires input_pipeline  = Pipeline([PadState, ActionNorm, ToTensor])
    Requires output_pipeline = Pipeline([ToNumpy, ActionDenorm, AbsoluteAction])
    """

    action_mode = "absolute"
    state_used = True
    state_required = False

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        norm_stats: dict,
        input_pipeline: Callable,
        output_pipeline: Callable,
        tokenization_func: Callable,
        device: torch.device,
        num_images: int = 3,
        non_delta_mask: list = None,
        action_dim: int = 7,
        camera_order: Optional[list] = None,
    ) -> None:
        super().__init__(
            model, tokenizer, norm_stats, input_pipeline, output_pipeline,
            camera_order=camera_order,
        )
        self.tokenization_func = tokenization_func
        self.device = device
        self.num_images = num_images
        self.non_delta_mask = non_delta_mask if non_delta_mask is not None else [6]
        self.action_dim = action_dim
        self.state_dim = getattr(self.model.model.config, "action_dim", action_dim)

    def _build_message(self, text: str) -> dict:
        """Message dict passed to tokenization_func. Override in subclasses."""
        return {"value": text}

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        batch_size = self._infer_batch_size(observation)
        obs = self._normalize_obs(observation, batch_size)

        batch_images_tensor = []
        batch_image_masks = []
        for i in range(batch_size):
            slot_images = []
            for slot in range(self.num_images):
                name = (
                    self.camera_order[slot]
                    if slot < len(self.camera_order)
                    else None
                )
                if name is None or f"image/{slot}" not in obs:
                    slot_images.append(None)
                else:
                    slot_images.append(self._load_images([obs[f"image/{slot}"][i]])[0])

            present_images = [image for image in slot_images if image is not None]
            if not present_images:
                raise ValueError("Pi0Policy requires at least one image")
            present_tensors = self.model.process_images(present_images).to(
                dtype=self.model.dtype
            )
            pad_tensor = torch.zeros_like(present_tensors[0:1])

            tensors = []
            present_idx = 0
            for image in slot_images:
                if image is None:
                    tensors.append(pad_tensor)
                else:
                    tensors.append(present_tensors[present_idx : present_idx + 1])
                    present_idx += 1
            image_tensor = torch.cat(tensors, dim=0)
            batch_images_tensor.append(image_tensor)

            mask = torch.tensor(
                [image is not None for image in slot_images],
                dtype=torch.bool,
            )
            batch_image_masks.append(mask)

        batch_images_tensor = torch.stack(batch_images_tensor, dim=0)  # [B, num_images, C, H, W]
        batch_image_masks = torch.stack(batch_image_masks, dim=0)      # [B, num_images]

        prompts = obs["prompt"]  # list[str], len == batch_size
        input_ids = np.array([
            self.tokenization_func([self._build_message(p)])["input_ids"]
            for p in prompts
        ])
        attention_mask = np.array([ids != self.tokenizer.pad_token_id for ids in input_ids])

        raw_states = obs.get("state", [None] * batch_size)
        action_dim_model = self.model.model.config.action_dim
        batch_states = np.array([
            np.array(s, dtype=np.float32) if s is not None
            else np.zeros(action_dim_model, dtype=np.float32)
            for s in raw_states
        ])  # [B, state_dim]

        inference_args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": batch_images_tensor,
            "image_masks": batch_image_masks,
            "state": batch_states,
            "meta_data": {"non_delta_mask": np.array(self.non_delta_mask)},
        }

        inputs = self.input_pipeline(inference_args)
        inputs["states"] = inputs["state"]
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        raw_actions = self.model.inference_action(**inputs)  # [B, chunk, action_dim_model]

        outputs = {
            k: v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = raw_actions.detach().float().cpu().numpy()
        outputs = self.output_pipeline(outputs)

        actions_batch = outputs["action"][:, ..., : self.action_dim]  # [B, chunk, action_dim]
        return [ActionOutput(actions=actions_batch[i]) for i in range(batch_size)]


class Pi05Policy(Pi0Policy):
    """
    Pi05-family policy wrapper.

    Pi05 reuses the Pi0 inference wrapper to provide a states tensor for batch,
    device, dtype, and normalization plumbing, but the Pi05 action model does
    not consume proprio/state as an action-conditioning token.
    """

    state_used = False
    state_required = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.state_dim = None
