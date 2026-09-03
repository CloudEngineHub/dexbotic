from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from dexbotic.data.dataset.dm05_data import DM05DataCollator, DM05ImagePreprocess
from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig


class DM05Policy(BasePolicy):
    action_mode = "absolute"
    state_used = False
    state_required = False

    def __init__(
        self,
        model: Any,
        processor: Any,
        norm_stats: dict,
        input_pipeline: Callable,
        output_pipeline: Callable,
        device: torch.device,
        num_images: int = 2,
        action_dim: int = 7,
        model_action_dim: int = 32,
        chunk_size: int = 10,
        diffusion_steps: int = 10,
        model_max_length: int = 768,
        camera_order: list | None = None,
    ) -> None:
        super().__init__(
            model,
            processor.tokenizer,
            norm_stats,
            input_pipeline,
            output_pipeline,
            camera_order=camera_order,
        )
        self.processor = processor
        self.device = device
        self.num_images = num_images
        self.action_dim = action_dim
        self.model_action_dim = model_action_dim
        self.chunk_size = chunk_size
        self.diffusion_steps = diffusion_steps
        self.image_preprocess = DM05ImagePreprocess()
        self.collator = DM05DataCollator(
            processor=processor,
            max_length=model_max_length,
            valid_action_dim=action_dim,
            model_action_dim=model_action_dim,
            chunk_size=chunk_size,
        )

    def select_action(
        self, observation: dict, sampling_config: SamplingConfig | None = None
    ):
        images = []
        for slot in range(self.num_images):
            key = f"image/{slot}"
            if key not in observation:
                raise ValueError(f"DM05Policy requires {key}")
            loaded = self._load_images([observation[key]])[0]
            images.append(self.image_preprocess.process_pil(loaded))
        state = np.asarray(
            observation.get("state", np.zeros(self.model_action_dim, dtype=np.float32)),
            dtype=np.float32,
        )
        prompt = observation.get("prompt", "")
        inputs = self.input_pipeline({"prompt": prompt, "state": state})
        state_tensor = inputs["state"]
        if isinstance(state_tensor, torch.Tensor) and state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        chw = [
            torch.from_numpy(np.array(img.convert("RGB"), dtype=np.uint8)).permute(
                2, 0, 1
            )
            for img in images[: self.num_images]
        ]
        batch = self.collator(
            [
                {
                    "input_ids": torch.tensor(
                        list(prompt.encode("utf-8")), dtype=torch.long
                    ),
                    "image": torch.stack(chw, dim=0),
                    "action": torch.zeros(self.chunk_size, self.model_action_dim),
                }
            ]
        )
        model_dtype = next(self.model.parameters()).dtype
        batch = {
            k: (
                v.to(
                    device=self.device,
                    dtype=(
                        model_dtype
                        if v.is_floating_point()
                        and k in {"pixel_values", "action_mask"}
                        else v.dtype
                    ),
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }
        actions = self.model.inference_action(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            token_type_ids=batch["token_type_ids"],
            diffusion_steps=self.diffusion_steps,
            action_mask=batch["action_mask"],
        )
        state_out = state_tensor.detach().cpu().float().numpy()
        if state_out.ndim == 1:
            state_out = state_out[None, :]
        outputs = self.output_pipeline(
            {
                "action": actions.detach().cpu().float().numpy(),
                "state": state_out,
            }
        )
        return [ActionOutput(actions=outputs["action"][0, :, : self.action_dim])]
