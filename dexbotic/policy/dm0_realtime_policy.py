from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig


class _RealtimeImagePreprocessor:
    """CLIP-style preprocessing required by the fixed-shape realtime backend."""

    def __init__(self, size: int, interpolation_mode: str = "bicubic") -> None:
        interpolation = (
            InterpolationMode.BICUBIC
            if interpolation_mode == "bicubic"
            else InterpolationMode.BILINEAR
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
                transforms.Resize(
                    (size, size), interpolation=interpolation, antialias=True
                ),
            ]
        )

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0)


class DM0RealtimePolicy(BasePolicy):
    """DM0 v1 policy backed by the Triton/CUDA-graph realtime inference path.

    This path is intentionally narrower than the standard DM0 policy: it serves
    single-sample v1 inference with fixed image/language/action shapes required
    by the realtime kernels. Legacy ``/process_frame`` continues to use the
    original DM0 model path unless the experiment opts into this policy.
    """

    action_mode = "absolute"
    state_used = True
    state_required = False
    state_dim = None
    max_batch_size = 1

    def __init__(
        self,
        realtime_model,
        tokenizer,
        norm_stats: dict,
        input_pipeline: Callable,
        output_pipeline: Callable,
        tokenization_func: Callable,
        device: torch.device,
        num_images: int = 3,
        non_delta_mask: Optional[list] = None,
        action_dim: int = 7,
        model_action_dim: int = 32,
        camera_order: Optional[list] = None,
        max_lang_len: int = 100,
    ) -> None:
        super().__init__(
            model=realtime_model,
            tokenizer=tokenizer,
            norm_stats=norm_stats,
            input_pipeline=input_pipeline,
            output_pipeline=output_pipeline,
            camera_order=camera_order,
        )
        self.realtime_model = realtime_model
        self.tokenization_func = tokenization_func
        self.device = device
        self.num_images = num_images
        self.non_delta_mask = non_delta_mask if non_delta_mask is not None else [6]
        self.action_dim = action_dim
        self.model_action_dim = model_action_dim
        self.max_lang_len = max_lang_len
        self.image_preprocessor = _RealtimeImagePreprocessor(
            size=728, interpolation_mode="bilinear"
        )
        self.image_pad_color = (0, 0, 0)
        active_camera_count = len([name for name in self.camera_order if name is not None])
        if active_camera_count > self.num_images:
            raise ValueError(
                "DM0 realtime backend expects camera_order to contain at most "
                f"{self.num_images} active camera view(s), got {active_camera_count}: "
                f"{self.camera_order}"
            )

    def _build_message(self, text: str) -> dict:
        return {"from": "human", "value": text}

    def _process_images(
        self, pil_images: list[Image.Image | None]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(pil_images) != self.num_images:
            raise ValueError(
                "DM0 realtime inference expects one image slot per configured "
                f"camera ({self.num_images}), got {len(pil_images)}"
            )
        if not any(image is not None for image in pil_images):
            raise ValueError("DM0 realtime inference requires at least one image")
        tensors = []
        image_masks = []
        for image in pil_images:
            if image is None:
                tensors.append(torch.zeros(3, 728, 728))
                image_masks.append(False)
            else:
                image = self._expand2square(image, self.image_pad_color)
                tensors.append(self.image_preprocessor.preprocess(image)[0])
                image_masks.append(True)
        image_tensor = torch.stack(tensors, dim=0).to(
            device=self.device, dtype=torch.bfloat16
        )
        image_mask_tensor = torch.as_tensor(
            image_masks, dtype=torch.bool, device=self.device
        )
        return image_tensor, image_mask_tensor

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = self.tokenization_func([self._build_message(prompt)])["input_ids"]
        input_ids = np.asarray(input_ids, dtype=np.int64)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        input_ids = input_ids[input_ids != pad_token_id]
        if input_ids.shape[0] < self.max_lang_len:
            input_ids = np.pad(
                input_ids,
                (0, self.max_lang_len - input_ids.shape[0]),
                constant_values=pad_token_id,
            )
        else:
            input_ids = input_ids[: self.max_lang_len]
        input_ids_tensor = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask = input_ids_tensor != int(pad_token_id)
        return input_ids_tensor, attention_mask

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        batch_size = self._infer_batch_size(observation)
        if batch_size != 1:
            raise ValueError("DM0 realtime backend only supports batch_size=1")

        obs = self._normalize_obs(observation, batch_size)
        sampling_fields = getattr(sampling_config, "_provided_fields", None)
        if sampling_config is not None and (
            sampling_fields is None or "num_steps" in sampling_fields
        ):
            captured_steps = int(self.realtime_model.diffusion_steps)
            requested_steps = int(sampling_config.num_steps)
            if requested_steps != captured_steps:
                raise ValueError(
                    "DM0 realtime backend uses a CUDA graph captured with "
                    f"{captured_steps} diffusion steps; per-request "
                    f"num_steps={requested_steps} is not supported."
                )

        pil_images = []
        for slot, name in enumerate(self.camera_order[: self.num_images]):
            if name is None or f"image/{slot}" not in obs:
                pil_images.append(None)
            else:
                pil_images.append(self._load_images([obs[f"image/{slot}"][0]])[0])
        while len(pil_images) < self.num_images:
            pil_images.append(None)
        images, image_masks = self._process_images(pil_images)
        input_ids, attention_mask = self._tokenize(obs["prompt"][0])

        raw_states = obs.get("state", [None])[0]
        state = (
            np.asarray(raw_states, dtype=np.float32)
            if raw_states is not None
            else np.zeros(self.model_action_dim, dtype=np.float32)
        )
        inference_args = {
            "state": np.expand_dims(state, axis=0),
            "meta_data": {"non_delta_mask": np.array(self.non_delta_mask)},
        }
        inputs = self.input_pipeline(inference_args)
        inputs["states"] = inputs["state"]
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        generator = None
        if sampling_config is not None and sampling_config.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(sampling_config.seed))
        noise = torch.randn(
            self.realtime_model.chunk_size,
            self.model_action_dim,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )

        raw_actions = self.realtime_model.forward(
            images,
            input_ids,
            noise,
            image_masks=image_masks,
            attention_mask=attention_mask,
        )
        outputs = {
            k: v.detach().float().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = raw_actions.detach().float().cpu().numpy()[None]
        outputs = self.output_pipeline(outputs)
        actions = outputs["action"][0, ..., : self.action_dim]
        return [ActionOutput(actions=actions)]

    @staticmethod
    def _expand2square(pil_img: Image.Image, background_color: tuple[int, int, int]):
        width, height = pil_img.size
        if width == height:
            return pil_img
        size = max(width, height)
        result = Image.new(pil_img.mode, (size, size), background_color)
        result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
        return result
