"""RoboTwin2 DM05 inference service for the default and fast backends."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import torch
from flask import jsonify, request

from dexbotic.exp.dm05_exp import DM05Exp as _DM05Exp
from dexbotic.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from dexbotic.infer import history_images_from_observation
from dexbotic.model.dm05.dm05_utils import (
    HISTORY_IMAGE_TOKEN,
    HISTORY_PAD_TOKEN,
    HISTORY_TOKENS_PER_IMAGE,
)
from dexbotic.policy.dm05_policy import DM05Policy

ROBOT_TYPE = "Aloha RoboTwin2"
IMAGE_PROMPTS = ("Head", "Left wrist", "Right wrist")
STATE_DIM = 14
ACTION_DIM = 14
ACTION_HORIZON = 50


def quantize_normalized_state(state: np.ndarray, n_bins: int = 256) -> list[int]:
    """Convert normalized state values in [-1, 1] to DM05 state tokens."""

    clipped = np.clip(np.asarray(state, dtype=np.float32), -1.0, 1.0)
    normalized = (clipped + 1.0) / 2.0
    bins = np.floor(normalized * (n_bins - 1)).astype(np.int64)
    return np.clip(bins, 0, n_bins - 1).tolist()


def robotwin2_prompt_text(prompt: str, speed: str = "0.5") -> str:
    return f"Robot: {ROBOT_TYPE}\nOverall speed: {speed}\nTask: {prompt}.\n"


class RoboTwin2DM05Policy(DM05Policy):
    """Adapt DM05 inputs and outputs to the RoboTwin2 evaluator contract."""

    state_used = True
    state_required = True
    state_dim = STATE_DIM

    def prepare_inputs(self, observation: dict, sampling_config=None):
        images = []
        for slot in range(len(IMAGE_PROMPTS)):
            key = f"image/{slot}"
            if key not in observation:
                raise ValueError(f"RoboTwin2 DM05 requires {key}")
            loaded = self._load_images([observation[key]])[0]
            images.append(self.image_preprocess.process_pil(loaded).convert("RGB"))
        unexpected = [
            key
            for key in observation
            if key.startswith("image/")
            and key not in {f"image/{i}" for i in range(len(IMAGE_PROMPTS))}
        ]
        if unexpected:
            raise ValueError(f"unexpected RoboTwin2 image slots: {unexpected}")
        history_images = history_images_from_observation(observation, self.history_spec)
        processed_history_images = [
            self.image_preprocess.process_pil(image).convert("RGB")
            for image in self._load_images(history_images)
        ]

        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(
                f"RoboTwin2 state shape must be ({STATE_DIM},), got {state.shape}"
            )
        if not np.isfinite(state).all():
            raise ValueError("RoboTwin2 state contains non-finite values")

        prompt = str(observation.get("prompt", ""))
        transformed = self.input_pipeline({"prompt": prompt, "state": state})
        state_tensor = transformed["state"]
        if not isinstance(state_tensor, torch.Tensor) or state_tensor.ndim != 1:
            raise ValueError(
                "DM05 state pipeline did not return a one-dimensional tensor"
            )
        if state_tensor.shape[0] < STATE_DIM:
            raise ValueError("DM05 state pipeline returned fewer than 14 state values")
        normalized_state = state_tensor[:STATE_DIM].detach().cpu().float().numpy()
        user_content = [{"type": "text", "text": robotwin2_prompt_text(prompt)}]
        history_pixel_values = None
        if self.history_spec.enabled:
            n_valid = len(processed_history_images)
            user_content[-1]["text"] += "History images: "
            user_content[-1]["text"] += (
                HISTORY_PAD_TOKEN
                * (HISTORY_TOKENS_PER_IMAGE * (self.history_spec.max_images - n_valid))
                + ((HISTORY_IMAGE_TOKEN * HISTORY_TOKENS_PER_IMAGE) + "\n") * n_valid
            )
            if processed_history_images:
                history_pixel_values = self.processor.image_processor(
                    images=processed_history_images,
                    return_tensors="pt",
                )["pixel_values"]
        for label, image in zip(IMAGE_PROMPTS, images, strict=True):
            user_content[-1]["text"] += f"{label} image: "
            user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": ""})
        user_content[-1]["text"] = "States: " + " ".join(
            str(value) for value in quantize_normalized_state(normalized_state)
        )
        model_inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if history_pixel_values is not None:
            history_mask = model_inputs["input_ids"] == self.collator.history_token_id
            expected_tokens = (
                int(history_pixel_values.shape[0]) * HISTORY_TOKENS_PER_IMAGE
            )
            actual_tokens = int(history_mask.sum().item())
            if actual_tokens != expected_tokens:
                raise ValueError(
                    "RoboTwin2 DM05 history placeholder count does not match "
                    f"history images: expected {expected_tokens}, got "
                    f"{actual_tokens}."
                )
            model_inputs["token_type_ids"] = model_inputs["token_type_ids"].clone()
            model_inputs["token_type_ids"][history_mask] = 1
            model_inputs["history_pixel_values"] = history_pixel_values
            model_inputs["history_mask"] = history_mask
        action_mask = torch.zeros(
            1,
            1,
            self.model_action_dim,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        action_mask[..., : self.action_dim] = 1.0
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in model_inputs.items()
            if key
            in {
                "input_ids",
                "attention_mask",
                "pixel_values",
                "token_type_ids",
                "history_pixel_values",
                "history_mask",
            }
        }
        batch["action_mask"] = action_mask
        return batch, {"state": state_tensor.detach().cpu().float().numpy()[None, :]}


@dataclass
class RoboTwin2InferenceConfig(_DM05InferenceConfig):
    num_images: int = field(default=len(IMAGE_PROMPTS))
    action_dim: int = field(default=ACTION_DIM)
    model_action_dim: int = field(default=32)
    chunk_size: int = field(default=ACTION_HORIZON)
    camera_order: list = field(default_factory=lambda: list(IMAGE_PROMPTS))
    vision_trt_engine_path: str = field(
        default="checkpoints/trt_engines/robotwin2_dm05_vision.engine"
    )

    def _build_policy(self):
        return RoboTwin2DM05Policy(
            model=self.model,
            processor=self.processor,
            norm_stats=self.norm_stats,
            input_pipeline=self.input_transform,
            output_pipeline=self.output_transform,
            device=self.device,
            num_images=self.num_images,
            action_dim=self.action_dim,
            model_action_dim=self.model_action_dim,
            chunk_size=self.chunk_size,
            diffusion_steps=self.diffusion_steps,
            model_max_length=self.model_max_length,
            camera_order=self.camera_order,
            history_enabled=self.history_enabled,
            max_history_images=self._effective_max_history_images(),
        )

    def process_frame(self):
        images = request.files.getlist("image")
        if len(images) != len(IMAGE_PROMPTS):
            return (
                jsonify(
                    {
                        "error": (
                            f"expected {len(IMAGE_PROMPTS)} RGB images, "
                            f"got {len(images)}"
                        )
                    }
                ),
                400,
            )
        raw_state = request.form.get("states")
        if raw_state is None:
            return jsonify({"error": "form field 'states' is required"}), 400
        try:
            state = np.asarray(json.loads(raw_state), dtype=np.float32)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return jsonify({"error": f"states must be a JSON array: {exc}"}), 400
        if state.shape != (STATE_DIM,):
            return (
                jsonify(
                    {
                        "error": (
                            f"RoboTwin2 state shape must be ({STATE_DIM},), "
                            f"got {state.shape}"
                        )
                    }
                ),
                400,
            )
        if not np.isfinite(state).all():
            return jsonify({"error": "RoboTwin2 state contains non-finite values"}), 400
        return super().process_frame()


@dataclass
class DM05Exp(_DM05Exp):
    inference_config: RoboTwin2InferenceConfig = field(
        default_factory=RoboTwin2InferenceConfig
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name-or-path",
        "--model_name_or_path",
        dest="model_name_or_path",
        required=True,
        help="RoboTwin2 DM05 checkpoint directory.",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="Optional norm_stats.json path when it is outside the checkpoint.",
    )
    parser.add_argument(
        "--backend",
        choices=["default", "fast"],
        default="default",
        help="DM05 inference backend.",
    )
    parser.add_argument("--port", type=int, default=7891)
    parser.add_argument(
        "--vision-trt-engine-path",
        default="checkpoints/trt_engines/robotwin2_dm05_vision.engine",
        help="TensorRT vision engine path used by the fast backend.",
    )
    parser.add_argument(
        "--build-vision-engine-if-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build a matching three-image TensorRT engine during fast startup.",
    )
    parser.add_argument(
        "--force-rebuild-vision-engine",
        action="store_true",
        help="Rebuild the TensorRT vision engine even when one already exists.",
    )
    parser.add_argument(
        "--prefix-seq-len-buckets",
        type=int,
        nargs="+",
        default=None,
        help="Fixed prefix-length buckets available to the fast backend.",
    )
    parser.add_argument(
        "--fast-overflow-policy",
        choices=["error", "fallback"],
        default="fallback",
    )
    parser.add_argument(
        "--fast-prefix-qkv-mode",
        choices=["packed", "separate"],
        default="packed",
    )
    parser.add_argument(
        "--history-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Accept explicit history images during inference.",
    )
    parser.add_argument(
        "--max-history-images",
        type=int,
        default=5,
        help="Maximum number of explicit history images accepted per request.",
    )
    return parser.parse_args(argv)


def configure_inference(exp: DM05Exp, args: argparse.Namespace) -> None:
    config = exp.inference_config
    config.model_name_or_path = args.model_name_or_path
    config.backend = args.backend
    config.port = args.port
    config.vision_trt_engine_path = args.vision_trt_engine_path
    config.build_vision_engine_if_missing = args.build_vision_engine_if_missing
    config.force_rebuild_vision_engine = args.force_rebuild_vision_engine
    config.fast_overflow_policy = args.fast_overflow_policy
    config.fast_prefix_qkv_mode = args.fast_prefix_qkv_mode
    config.history_enabled = args.history_enabled
    config.max_history_images = args.max_history_images
    if args.prefix_seq_len_buckets is not None:
        config.prefix_seq_len_buckets = args.prefix_seq_len_buckets
    if args.norm_stats is not None:
        config.norm_stats = config.read_normalization_stats(args.norm_stats)


def main(argv: list[str] | None = None) -> None:
    exp = DM05Exp()
    configure_inference(exp, parse_args(argv))
    exp.inference()


if __name__ == "__main__":
    main()
