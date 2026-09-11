from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from dexbotic.data.dataset.dm05_data import DM05DataCollator, DM05ImagePreprocess
from dexbotic.history import HistoryImageSpec
from dexbotic.infer.history import (
    history_image_capabilities,
    history_images_from_observation,
)
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
        history_enabled: bool = False,
        max_history_images: int = 5,
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
        self.history_spec = HistoryImageSpec(
            enabled=bool(history_enabled),
            max_images=int(max_history_images),
        )
        self.inference_runtime = None
        self.last_inference_metadata = None
        self.image_preprocess = DM05ImagePreprocess()
        self.collator = DM05DataCollator(
            processor=processor,
            max_length=model_max_length,
            valid_action_dim=action_dim,
            model_action_dim=model_action_dim,
            chunk_size=chunk_size,
            history_enabled=self.history_spec.enabled,
            max_history_images=self.history_spec.max_images,
        )

    def select_action(
        self, observation: dict, sampling_config: SamplingConfig | None = None
    ) -> list[ActionOutput]:
        if self.inference_runtime is not None:
            result = self.inference_runtime.infer(observation, sampling_config)
            self.last_inference_metadata = dict(result.metadata)
            return [result.output]
        batch, context = self.prepare_inputs(observation, sampling_config)
        model_inputs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            token_type_ids=batch["token_type_ids"],
            diffusion_steps=self.diffusion_steps,
            action_mask=batch["action_mask"],
        )
        for name in ("history_pixel_values", "history_mask"):
            if name in batch:
                model_inputs[name] = batch[name]
        actions = self.model.inference_action(**model_inputs)
        return [self.finalize_actions(actions, context)]

    def get_capabilities(self) -> dict:
        capabilities = super().get_capabilities()
        capabilities["history"] = history_image_capabilities(self.history_spec)
        return capabilities

    def prepare_inputs(
        self, observation: dict, sampling_config: SamplingConfig | None = None
    ) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        """Build the one-sample tensor request shared by all DM05 backends."""

        images = []
        for slot in range(self.num_images):
            key = f"image/{slot}"
            if key not in observation:
                raise ValueError(f"DM05Policy requires {key}")
            loaded = self._load_images([observation[key]])[0]
            images.append(self.image_preprocess.process_pil(loaded))
        history_images = history_images_from_observation(observation, self.history_spec)
        processed_history_images = [
            self.image_preprocess.process_pil(image)
            for image in self._load_images(history_images)
        ]
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
        instance = {
            "input_ids": torch.tensor(list(prompt.encode("utf-8")), dtype=torch.long),
            "image": torch.stack(chw, dim=0),
            "action": torch.zeros(self.chunk_size, self.model_action_dim),
        }
        if processed_history_images:
            instance["history_images"] = torch.stack(
                [
                    torch.from_numpy(
                        np.array(image.convert("RGB"), dtype=np.uint8)
                    ).permute(2, 0, 1)
                    for image in processed_history_images
                ],
                dim=0,
            )
        batch = self.collator([instance])
        # Valid action dimensions do not vary across the horizon. Preserve the
        # broadcastable shape so CUDA Graph capture does not materialize a
        # redundant [chunk, action_dim] mask.
        batch["action_mask"] = batch["action_mask"][:, :1, :]
        model_dtype = next(self.model.parameters()).dtype
        batch = {
            k: (
                v.to(
                    device=self.device,
                    dtype=(
                        model_dtype
                        if v.is_floating_point() and k == "action_mask"
                        else v.dtype
                    ),
                )
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in batch.items()
        }
        state_out = state_tensor.detach().cpu().float().numpy()
        if state_out.ndim == 1:
            state_out = state_out[None, :]
        return batch, {"state": state_out}

    def finalize_actions(
        self,
        actions: torch.Tensor,
        context: dict[str, np.ndarray],
    ) -> ActionOutput:
        """Apply the unchanged DM05 action denormalization/output contract."""

        outputs = self.output_pipeline(
            {
                "action": actions.detach().cpu().float().numpy(),
                "state": context["state"],
            }
        )
        return ActionOutput(actions=outputs["action"][0, :, : self.action_dim])
