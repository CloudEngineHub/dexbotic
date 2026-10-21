"""Fast DM05 backend for the common inference runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from dexbotic.infer.runtime import InferenceRuntime


@dataclass(frozen=True)
class DM05FastBackendConfig:
    """Configure TensorRT, Triton, and CUDA Graph inference for DM05."""

    vision_engine_path: str = "checkpoints/trt_engines/dm05_vision.engine"
    build_engine_if_missing: bool = False
    force_rebuild_engine: bool = False
    prefix_buckets: tuple[int, ...] = field(default=(576, 704, 768, 896, 1024))
    overflow_policy: str = "fallback"
    prefix_qkv_mode: str = "packed"

    def __post_init__(self) -> None:
        buckets = tuple(int(value) for value in self.prefix_buckets)
        if not buckets or buckets != tuple(sorted(set(buckets))):
            raise ValueError(
                "prefix_buckets must be non-empty, strictly increasing, and unique."
            )
        if buckets[0] <= 0 or buckets[-1] > 1024:
            raise ValueError("prefix_buckets must be within 1..1024.")
        if self.overflow_policy not in {"error", "fallback"}:
            raise ValueError("overflow_policy must be 'error' or 'fallback'.")
        if self.prefix_qkv_mode not in {"packed", "separate"}:
            raise ValueError("prefix_qkv_mode must be 'packed' or 'separate'.")


def _model_inputs(tensors: dict) -> dict:
    inputs = {
        "input_ids": tensors["input_ids"],
        "attention_mask": tensors["attention_mask"],
        "pixel_values": tensors["pixel_values"],
        "token_type_ids": tensors["token_type_ids"],
        "action_mask": tensors["action_mask"],
    }
    for name in ("history_pixel_values", "history_mask", "diffusion_input_noise"):
        if name in tensors:
            inputs[name] = tensors[name]
    return inputs


class DM05FastBackend:
    def __init__(
        self,
        model,
        *,
        checkpoint: str,
        num_images: int,
        diffusion_steps: int,
        config: DM05FastBackendConfig,
        history_enabled: bool = False,
    ) -> None:
        model.set_attention_implementation(
            llm_attn_implementation="flex_attention",
            vision_attn_implementation="sdpa",
            action_attn_implementation="sdpa",
            bf16=True,
        )
        from dexbotic.model.dm05.infer.fast.vision_trt import (
            MAX_HISTORY_IMAGES,
            DM05VisionTensorRTRunner,
            ensure_dm05_vision_engine,
        )

        device = next(model.parameters()).device
        vision_image_slots = int(num_images) + (
            MAX_HISTORY_IMAGES if history_enabled else 0
        )
        self.prefix_mlp_backend = None
        engine_path = Path(config.vision_engine_path).expanduser()
        vision_runner = None

        def load_vision(path):
            runner = DM05VisionTensorRTRunner(path, device=device)
            if runner.num_images != vision_image_slots:
                raise ValueError(
                    "DM05 vision engine image count does not match configured "
                    "current/history slots: "
                    f"engine={runner.num_images}, configured={vision_image_slots}, "
                    f"path={path}."
                )
            return runner

        if not history_enabled and not config.force_rebuild_engine:
            from dexbotic.model.dm05.infer.fast.build_prefix_mlp_trt import (
                find_prepared_fp8_engines,
            )
            from dexbotic.model.dm05.infer.fast.prefix_mlp_trt import (
                PrefixMLPFP8Backend,
            )

            minimum = num_images * (
                int(model.model.vlm.model.config.mm_tokens_per_image) + 1
            )
            buckets = tuple(b for b in config.prefix_buckets if b >= minimum)
            if not buckets:
                raise ValueError(
                    "No DM05 prefix bucket can fit the configured image tokens: "
                    f"minimum={minimum}, buckets={config.prefix_buckets}."
                )

            prepared_engines = find_prepared_fp8_engines(checkpoint)
            if prepared_engines is not None:
                try:
                    prepared_vision = load_vision(prepared_engines[0])
                    prepared_prefix = PrefixMLPFP8Backend(
                        model.model.vlm,
                        prepared_engines[1],
                        buckets,
                        device=device,
                        vision_engine_identity=prepared_vision.engine_identity,
                    )
                except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
                    logger.warning(
                        "Prepared FP8 engines cannot be used; using BF16 fast "
                        "inference: {}",
                        exc,
                    )
                else:
                    engine_path = prepared_engines[0]
                    vision_runner = prepared_vision
                    self.prefix_mlp_backend = prepared_prefix
                    logger.info(
                        "Enabled native FP8 prefix MLP with vision engine {}",
                        engine_path,
                    )

        if vision_runner is None:
            if config.force_rebuild_engine or config.build_engine_if_missing:
                ensure_dm05_vision_engine(
                    checkpoint=checkpoint,
                    engine_path=engine_path,
                    num_images=vision_image_slots,
                    force_rebuild=config.force_rebuild_engine,
                )
            vision_runner = load_vision(engine_path)

        from dexbotic.model.dm05.infer.fast.runtime import DM05FastRuntime

        self.runtime = DM05FastRuntime(
            model,
            engine_path,
            prefix_buckets=config.prefix_buckets,
            diffusion_steps=diffusion_steps,
            overflow_policy=config.overflow_policy,
            prefix_qkv_mode=config.prefix_qkv_mode,
            history_enabled=history_enabled,
            vision_runner=vision_runner,
            prefix_mlp_backend=self.prefix_mlp_backend,
        )

    def execute(self, tensors: dict):
        model_inputs = _model_inputs(tensors)
        actions = self.runtime.infer(**model_inputs)
        has_history = "history_pixel_values" in model_inputs
        return (
            actions,
            {
                "dynamic_fallback_count": self.runtime.dynamic_fallback_count,
                "profile_capture_count": self.runtime.profile_capture_count,
                "graph_replay_count": self.runtime.graph_replay_count,
                "history_uncaptured_count": self.runtime.history_uncaptured_count,
                "history_backend": "fast_uncaptured" if has_history else "none",
            },
        )


def build_dm05_runtime(
    *,
    policy,
    checkpoint: str,
    num_images: int,
    diffusion_steps: int,
    fast_config: DM05FastBackendConfig,
    history_enabled: bool = False,
) -> InferenceRuntime:
    return InferenceRuntime(
        prepare=policy.prepare_inputs,
        finalize=policy.finalize_actions,
        backend=DM05FastBackend(
            policy.model,
            checkpoint=checkpoint,
            num_images=num_images,
            diffusion_steps=diffusion_steps,
            config=fast_config,
            history_enabled=history_enabled,
        ),
        serialize_calls=False,
    )
