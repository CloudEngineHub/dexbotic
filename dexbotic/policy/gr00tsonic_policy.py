from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
import torch

from dexbotic.policy.base_policy import BasePolicy
from dexbotic.policy.types import ActionOutput, SamplingConfig


try:
    from transformers import Qwen3VLProcessor
except ImportError:  # pragma: no cover - depends on transformers>=4.57
    Qwen3VLProcessor = None


class Gr00tSonicPolicy(BasePolicy):
    """Inference policy for GR00T N1.7 (gr00tsonic).

    Unlike the pi0-family policies (which use Dexbotic's siglip vision tower),
    gr00tsonic keeps the upstream monolithic Qwen3-VL backbone, so this policy
    builds the model inputs with the ``Qwen3VLProcessor`` exactly as the original
    Isaac-GR00T processor does:

        conversation -> apply_chat_template -> processor(text, images)
            -> {input_ids, attention_mask, pixel_values, image_grid_thw}

    observation keys (named camera format):
        "image/{slot}": PIL Image | ndarray | path str  — single sample
                        list of the above               — batch
        "prompt":       str (broadcast) | list[str]     — task instruction
        "state":        ndarray [state_dim] (optional)  — robot proprio state

    Requires input_pipeline  = Pipeline([PadState, ToTensor])  (state is padded, not normalized)
    Requires output_pipeline = Pipeline([ToNumpy, ActionDenorm])  (SONIC actions are absolute)
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
        device: torch.device,
        camera_order: Optional[list] = None,
        processor: Any = None,
        embodiment_id: int = 10,
        formalize_language: bool = True,
        action_dim: int = 7,
    ) -> None:
        super().__init__(
            model,
            tokenizer,
            norm_stats,
            input_pipeline,
            output_pipeline,
            camera_order=camera_order,
        )
        self.device = device
        cfg = self.model.model.config
        self.max_state_dim = cfg.max_state_dim
        self.max_action_dim = cfg.max_action_dim
        self.state_history_length = cfg.state_history_length
        self.action_horizon = cfg.action_horizon
        self.state_dim = self.max_state_dim
        self.action_dim = action_dim
        self.embodiment_id = embodiment_id
        self.formalize_language = formalize_language

        if processor is not None:
            self.processor = processor
        else:
            if Qwen3VLProcessor is None:
                raise ImportError(
                    "Qwen3VLProcessor is not available. Please use transformers>=4.57."
                )
            self.processor = Qwen3VLProcessor.from_pretrained(cfg.model_name)
        # Left padding for Flash-Attention compatibility (matches upstream).
        self.processor.tokenizer.padding_side = "left"

    # ── VLM input construction ───────────────────────────────────────────────

    def _formalize(self, text: str) -> str:
        if not self.formalize_language:
            return text
        import re

        return re.sub(r"[^\w\s]", "", text.lower())

    def _build_vlm_text(self, pil_images: list[Image.Image], language: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": language},
                ],
            }
        ]
        return self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

    # ── VLA inference ────────────────────────────────────────────────────────

    def select_action(
        self,
        observation: dict,
        sampling_config: Optional[SamplingConfig] = None,
    ) -> list[ActionOutput]:
        batch_size = self._infer_batch_size(observation)
        obs = self._normalize_obs(observation, batch_size)

        texts: list[str] = []
        per_sample_images: list[list[Image.Image]] = []
        for i in range(batch_size):
            pil_images = []
            for slot, name in enumerate(self.camera_order):
                if name is None or f"image/{slot}" not in obs:
                    continue
                pil_images.extend(self._load_images([obs[f"image/{slot}"][i]]))
            if not pil_images:
                raise ValueError("Gr00tSonicPolicy requires at least one image")
            language = self._formalize(obs["prompt"][i])
            texts.append(self._build_vlm_text(pil_images, language))
            per_sample_images.append(pil_images)

        all_images = [img for imgs in per_sample_images for img in imgs]
        vlm_inputs = self.processor(
            text=texts, images=all_images, return_tensors="pt", padding=True
        )

        # State: normalize via input pipeline, pad to max_state_dim, add history dim.
        raw_states = obs.get("state", [None] * batch_size)
        batch_states = []
        for s in raw_states:
            if s is None:
                arr = np.zeros(self.max_state_dim, dtype=np.float32)
            else:
                normed = self.input_pipeline({"state": np.asarray(s, dtype=np.float32)})
                arr = normed["state"]
                if isinstance(arr, torch.Tensor):
                    arr = arr.cpu().numpy()
                arr = np.asarray(arr, dtype=np.float32).reshape(-1)
            if arr.shape[0] < self.max_state_dim:
                arr = np.concatenate(
                    [arr, np.zeros(self.max_state_dim - arr.shape[0], dtype=np.float32)]
                )
            else:
                arr = arr[: self.max_state_dim]
            batch_states.append(arr)
        # [B, state_history_length, max_state_dim]
        state = torch.from_numpy(np.stack(batch_states))[:, None, :].repeat(
            1, self.state_history_length, 1
        )

        embodiment_id = torch.full(
            (batch_size,), self.embodiment_id, dtype=torch.long
        )

        inputs = {
            "input_ids": vlm_inputs["input_ids"],
            "attention_mask": vlm_inputs["attention_mask"],
            "pixel_values": vlm_inputs["pixel_values"],
            "image_grid_thw": vlm_inputs["image_grid_thw"],
            "state": state,
            "embodiment_id": embodiment_id,
        }
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        # Cast floating-point inputs to the model dtype.
        for k in ("pixel_values", "state"):
            inputs[k] = inputs[k].to(dtype=self.model.dtype)

        raw_actions = self.model.inference_action(**inputs)  # [B, horizon, max_action_dim]

        outputs = {"action": raw_actions.detach().float().cpu().numpy()}
        outputs = self.output_pipeline(outputs)
        actions_batch = outputs["action"][:, ..., : self.action_dim]
        return [ActionOutput(actions=actions_batch[i]) for i in range(batch_size)]
