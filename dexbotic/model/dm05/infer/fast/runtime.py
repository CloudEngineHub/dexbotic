"""Bucketed DM05 fast runtime with optional uncaptured history inference."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger
from transformers.cache_utils import Cache, CacheLayerMixin

from dexbotic.infer.cuda_graph import CudaGraphRunner, ensure_cuda_graph_available
from dexbotic.infer.profiles import SmallestFittingProfileRegistry
from dexbotic.model.dm05.dm05_arch import DM05ForConditionalGeneration
from dexbotic.model.dm05.dm05_utils import (
    HISTORY_TOKENS_PER_IMAGE,
    mask_history_pad_tokens_in_attention,
)
from dexbotic.model.dm05.infer.fast.arch import DM05FastForCausalLM
from dexbotic.model.dm05.infer.fast.vision_trt import (
    MAX_HISTORY_IMAGES,
    DM05VisionTensorRTRunner,
    pack_current_and_history_pixels,
    pool_image_features_to_history,
)


class StaticPrefixCacheLayer(CacheLayerMixin):
    """Address-stable overwrite cache used by CUDA Graph replay."""

    is_sliding = False

    def __init__(self) -> None:
        super().__init__()
        self.keys = None
        self.values = None
        self.seq_len = 0

    def reset_for_prefill(self) -> None:
        self.seq_len = 0

    reset = reset_for_prefill

    def lazy_initialization(self, key_states, value_states) -> None:
        self._allocate_like(key_states, value_states)

    def _allocate_like(self, key_states, value_states) -> None:
        self.keys = torch.empty_like(key_states)
        self.values = torch.empty_like(value_states)
        if self.keys.is_cuda:
            torch._dynamo.mark_static_address(self.keys)
            torch._dynamo.mark_static_address(self.values)
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.is_initialized = True
        self.seq_len = 0

    def update(self, key_states, value_states, cache_kwargs=None):
        if (
            self.keys is None
            or self.values is None
            or self.keys.shape != key_states.shape
            or self.values.shape != value_states.shape
            or self.keys.dtype != key_states.dtype
            or self.values.dtype != value_states.dtype
            or self.keys.device != key_states.device
            or self.values.device != value_states.device
        ):
            self._allocate_like(key_states, value_states)
        assert self.keys is not None and self.values is not None
        self.keys.copy_(key_states)
        self.values.copy_(value_states)
        self.seq_len = int(key_states.shape[-2])
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.seq_len + int(cache_position.shape[0]), 0

    def get_seq_length(self) -> int:
        return self.seq_len

    def get_max_cache_shape(self) -> int:
        return -1 if self.keys is None else int(self.keys.shape[-2])


@dataclass
class DM05GraphProfile:
    """Model-specific state for a captured prefix-prefill and denoising profile."""

    prefix_cache: Cache
    prefix_buffers: Any
    suffix_buffers: Any
    graph_runner: CudaGraphRunner | None = None


@dataclass
class DM05HistoryProfile:
    """Reusable uncaptured prefix state for one history bucket."""

    prefix_cache: Cache
    prefix_buffers: Any


class DM05FastRuntime:
    """Run TensorRT vision, then replay prefix prefill and denoising as a graph."""

    def __init__(
        self,
        model: DM05ForConditionalGeneration,
        vision_engine_path: str | Path,
        *,
        prefix_buckets: tuple[int, ...],
        diffusion_steps: int,
        overflow_policy: str = "fallback",
        prefix_qkv_mode: str = "packed",
        history_enabled: bool = False,
        vision_runner=None,
    ) -> None:
        ensure_cuda_graph_available()
        self.model = DM05FastForCausalLM(
            model,
            prefix_qkv_mode=prefix_qkv_mode,
        )
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise RuntimeError("DM05 fast inference model must be on CUDA.")
        self.dm05 = self.model.dm05
        self.action_expert = self.model.action_expert
        self.padding_idx = self.model.padding_idx
        self.suffix_len = self.model.suffix_len
        self.action_dim = self.model.action_dim
        self.noise_dtype = self.model.noise_dtype
        self.history_enabled = bool(history_enabled)
        self.diffusion_steps = int(diffusion_steps)
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        if overflow_policy not in {"error", "fallback"}:
            raise ValueError("overflow_policy must be 'error' or 'fallback'.")
        self.overflow_policy = overflow_policy
        self.vision_runner = vision_runner or DM05VisionTensorRTRunner(
            vision_engine_path,
            device=self.device,
        )
        self.vision_trt_num_images = int(self.vision_runner.num_images)
        if self.history_enabled:
            if self.vision_trt_num_images <= MAX_HISTORY_IMAGES:
                raise ValueError(
                    "History-enabled fast inference requires a vision TensorRT "
                    f"engine built for num_current+{MAX_HISTORY_IMAGES} images, "
                    f"got num_images={self.vision_trt_num_images}."
                )
            self.num_current_images = self.vision_trt_num_images - MAX_HISTORY_IMAGES
            self._vision_trt_full_features = torch.empty(
                self.vision_runner.output_shape,
                device=self.device,
                dtype=self.vision_runner.output_dtype,
            )
        else:
            self.num_current_images = self.vision_trt_num_images
            self._vision_trt_full_features = None
        # Preserve the historical attribute for no-history callers.
        self.num_images = self.num_current_images
        tokens_per_image = int(self.dm05.vlm.model.config.mm_tokens_per_image)
        minimum_prefix_len = self.num_current_images * (tokens_per_image + 1)
        if self.history_enabled:
            minimum_prefix_len += MAX_HISTORY_IMAGES * HISTORY_TOKENS_PER_IMAGE
        usable_buckets = tuple(
            int(size) for size in prefix_buckets if int(size) >= minimum_prefix_len
        )
        if not usable_buckets:
            raise ValueError(
                "No DM05 prefix bucket can fit the configured image/history tokens: "
                f"minimum={minimum_prefix_len}, buckets={tuple(prefix_buckets)}."
            )
        if usable_buckets != tuple(prefix_buckets):
            logger.info(
                "Ignoring DM05 prefix buckets smaller than the configured image "
                "minimum {}: requested={}, active={}",
                minimum_prefix_len,
                tuple(prefix_buckets),
                usable_buckets,
            )
        self.profiles = SmallestFittingProfileRegistry[DM05GraphProfile](usable_buckets)
        self.dynamic_fallback_count = 0
        self.profile_capture_count = 0
        self.graph_replay_count = 0
        self.history_uncaptured_count = 0
        self._history_profiles: dict[int, DM05HistoryProfile] = {}
        self.dynamic_prefix_cache = self._make_prefix_cache()
        self._lock = threading.Lock()

    def _make_prefix_cache(self) -> Cache:
        config = self.dm05.language_model.config
        layer_types = getattr(config, "layer_types", None)
        layer_count = (
            len(layer_types)
            if layer_types is not None
            else int(config.num_hidden_layers)
        )
        if hasattr(config, "num_kv_shared_layers"):
            layer_count -= int(config.num_kv_shared_layers)
        return Cache(layers=[StaticPrefixCacheLayer() for _ in range(layer_count)])

    def _encode_vision_features(
        self,
        current_images: torch.Tensor,
        history_pixel_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run unified TRT vision and pool the real history slots."""

        if int(current_images.shape[0]) != self.num_current_images:
            raise ValueError(
                "Current-view batch must match the TRT current-image count: "
                f"expected {self.num_current_images}, got "
                f"{int(current_images.shape[0])}."
            )
        if not self.history_enabled:
            if history_pixel_values is not None:
                raise ValueError(
                    "history_pixel_values require a history-enabled DM05 fast runtime."
                )
            return self.vision_runner(current_images), None

        packed, num_history = pack_current_and_history_pixels(
            current_images,
            history_pixel_values,
            max_history_images=MAX_HISTORY_IMAGES,
        )
        if int(packed.shape[0]) != self.vision_trt_num_images:
            raise ValueError(
                "Packed vision batch must match the TRT engine image count: "
                f"expected {self.vision_trt_num_images}, got "
                f"{int(packed.shape[0])}."
            )
        full_features = self.vision_runner(
            packed,
            output_tensor=self._vision_trt_full_features,
        )
        current_features = full_features[: self.num_current_images]
        if num_history <= 0:
            return current_features, None
        history_features = pool_image_features_to_history(
            full_features[
                self.num_current_images : self.num_current_images + num_history
            ]
        )
        return current_features, history_features

    def _history_profile(self, bucket_len: int) -> DM05HistoryProfile:
        profile = self._history_profiles.get(bucket_len)
        if profile is not None:
            return profile
        if not self.model.prefix_decoder_initialized:
            self.model.setup_fast_prefix_decoder()
        profile = DM05HistoryProfile(
            prefix_cache=self._make_prefix_cache(),
            prefix_buffers=self.action_expert.make_fast_prefix_buffers(bucket_len),
        )
        self._history_profiles[bucket_len] = profile
        return profile

    def _make_and_capture_profile(
        self,
        bucket_len: int,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        image_features: torch.Tensor,
        noise: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> DM05GraphProfile:
        request_len = int(input_ids.shape[1])
        if request_len > bucket_len:
            raise ValueError(
                f"Prefix length {request_len} exceeds graph bucket {bucket_len}."
            )
        if not self.model.prefix_decoder_initialized:
            self.model.setup_fast_prefix_decoder()
        self.model._get_suffix_time_modulations(
            batch_size=1,
            diffusion_steps=self.diffusion_steps,
            dtype=self.noise_dtype,
            device=self.device,
        )
        profile = DM05GraphProfile(
            prefix_cache=self._make_prefix_cache(),
            prefix_buffers=self.action_expert.make_fast_prefix_buffers(bucket_len),
            suffix_buffers=self.action_expert.make_fast_suffix_buffers(
                batch_size=1,
                seq_len=self.suffix_len,
                max_kv_len=bucket_len + self.suffix_len,
                dtype=self.model.suffix_kernel_dtype,
                device=self.device,
            ),
        )
        capture_input_ids = torch.full(
            (1, bucket_len),
            self.padding_idx,
            dtype=input_ids.dtype,
            device=self.device,
        )
        capture_input_ids[:, :request_len].copy_(input_ids)
        capture_attention_mask = torch.zeros(
            (1, bucket_len),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        capture_attention_mask[:, :request_len].copy_(attention_mask)
        capture_token_type_ids = torch.zeros(
            (1, bucket_len),
            dtype=token_type_ids.dtype,
            device=self.device,
        )
        capture_token_type_ids[:, :request_len].copy_(token_type_ids)
        capture_inputs = {
            "input_ids": capture_input_ids,
            "attention_mask": capture_attention_mask,
            "token_type_ids": capture_token_type_ids,
            "image_features": image_features,
            "noise": noise,
            "action_mask": action_mask,
        }
        profile.graph_runner = CudaGraphRunner.capture(
            lambda **inputs: self._run_graph_profile(profile, inputs),
            capture_inputs,
            clone_outputs=False,
        )
        self.profile_capture_count += 1
        logger.info(
            "Lazily captured DM05 prefix-prefill and denoising profile for "
            "prefix bucket {}",
            bucket_len,
        )
        return profile

    def _run_fast_profile(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        image_features: torch.Tensor,
        noise: torch.Tensor,
        action_mask: torch.Tensor,
        prefix_cache: Cache,
        prefix_buffers: Any,
        suffix_buffers: Any | None,
        history_features: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the bucketed fast prefill and diffusion decode core."""

        if history_mask is not None:
            attention_mask = mask_history_pad_tokens_in_attention(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        position_ids = (attention_mask.to(torch.long).cumsum(dim=-1) - 1).clamp_min_(0)
        prefix_cache, prefix_len = self.model.prefill_fast_from_image_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_features=image_features,
            token_type_ids=token_type_ids,
            prefix_cache=prefix_cache,
            prefix_buffers=prefix_buffers,
            history_mask=history_mask,
            history_features=history_features,
        )
        prefix_visible_mask = self.model.build_prefix_visible_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_len=prefix_len,
        )
        effective_prefix_len = prefix_visible_mask.long().sum(dim=1)
        suffix_position_ids = (
            effective_prefix_len[:, None]
            + torch.arange(self.suffix_len, device=self.device)[None, :]
        )
        context = self.model.prepare_fast_suffix_context(
            prefix_cache=prefix_cache,
            prefix_visible_mask=prefix_visible_mask,
            suffix_position_ids=suffix_position_ids,
            initial_noise=noise,
            diffusion_steps=self.diffusion_steps,
            suffix_buffers=suffix_buffers,
        )
        return self.model.decode_fast_prepared(
            initial_noise=noise,
            diffusion_steps=self.diffusion_steps,
            action_mask=action_mask,
            **context,
        )

    def _run_graph_profile(
        self,
        profile: DM05GraphProfile,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self._run_fast_profile(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            image_features=inputs["image_features"],
            noise=inputs["noise"],
            action_mask=inputs["action_mask"],
            prefix_cache=profile.prefix_cache,
            prefix_buffers=profile.prefix_buffers,
            suffix_buffers=profile.suffix_buffers,
        )

    @contextmanager
    def _prefix_attention_backend(self, implementation: str):
        vlm = self.dm05.vlm.model
        configs = [vlm.config.get_text_config(), vlm.language_model.config]
        missing = object()
        saved = []
        for config in {id(value): value for value in configs}.values():
            values = {
                name: getattr(config, name, missing)
                for name in ("_attn_implementation", "attn_implementation")
            }
            saved.append((config, values))
            config._attn_implementation = implementation
            config.attn_implementation = implementation
        try:
            yield
        finally:
            for config, values in saved:
                for name, value in values.items():
                    if value is missing:
                        delattr(config, name)
                    else:
                        setattr(config, name, value)

    def _run_dynamic_fallback(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        image_features: torch.Tensor,
        noise: torch.Tensor,
        action_mask: torch.Tensor,
        history_features: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with self._prefix_attention_backend("eager"):
            prefix_cache, prefix_len = self.model.prefill_from_image_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_features=image_features,
                token_type_ids=token_type_ids,
                prefix_cache=self.dynamic_prefix_cache,
                history_mask=history_mask,
                history_features=history_features,
            )
        attention_mask = mask_history_pad_tokens_in_attention(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.model.decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prefix_cache=prefix_cache,
            prefix_len=prefix_len,
            diffusion_input_noise=noise,
            diffusion_steps=self.diffusion_steps,
            action_mask=action_mask,
        )

    def _run_with_history_without_capture(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        noise: torch.Tensor,
        action_mask: torch.Tensor,
        history_pixel_values: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Use unified TRT and fast kernels without a request CUDA Graph."""

        self.history_uncaptured_count += 1
        if self.history_uncaptured_count == 1:
            logger.info(
                "History images present; using unified TRT "
                "(current+padded history) with uncaptured fast prefill/decode."
            )

        request_len = int(input_ids.shape[1])
        bucket_len = self.profiles.resolve_size(
            request_len,
            overflow_policy=self.overflow_policy,
            request_name="Prefix length",
            profile_name="bucket",
        )
        image_features, history_features = self._encode_vision_features(
            pixel_values,
            history_pixel_values,
        )
        if bucket_len is None:
            self.dynamic_fallback_count += 1
            return self._run_dynamic_fallback(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image_features=image_features,
                noise=noise,
                action_mask=action_mask,
                history_features=history_features,
                history_mask=history_mask,
            )

        profile = self._history_profile(bucket_len)
        padded_input_ids = torch.full(
            (1, bucket_len),
            self.padding_idx,
            dtype=torch.long,
            device=self.device,
        )
        padded_attention_mask = torch.zeros(
            (1, bucket_len),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        padded_token_type_ids = torch.zeros(
            (1, bucket_len),
            dtype=torch.long,
            device=self.device,
        )
        padded_history_mask = torch.zeros(
            (1, bucket_len),
            dtype=torch.bool,
            device=self.device,
        )
        padded_input_ids[:, :request_len].copy_(input_ids)
        padded_attention_mask[:, :request_len].copy_(attention_mask)
        padded_token_type_ids[:, :request_len].copy_(token_type_ids)
        history_len = min(request_len, int(history_mask.shape[1]))
        padded_history_mask[:, :history_len].copy_(history_mask[:, :history_len])

        return self._run_fast_profile(
            input_ids=padded_input_ids,
            attention_mask=padded_attention_mask,
            token_type_ids=padded_token_type_ids,
            image_features=image_features,
            noise=noise,
            action_mask=action_mask,
            prefix_cache=profile.prefix_cache,
            prefix_buffers=profile.prefix_buffers,
            suffix_buffers=None,
            history_features=history_features,
            history_mask=padded_history_mask,
        )

    @torch.inference_mode()
    def infer(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: torch.Tensor,
        action_mask: torch.Tensor,
        diffusion_input_noise: torch.Tensor | None = None,
        history_pixel_values: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one request and return a request-owned CPU action tensor."""

        with self._lock:
            output = self._infer_locked(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                token_type_ids=token_type_ids,
                action_mask=action_mask,
                diffusion_input_noise=diffusion_input_noise,
                history_pixel_values=history_pixel_values,
                history_mask=history_mask,
            )
            return self._materialize_output(output)

    @staticmethod
    def _materialize_output(output: torch.Tensor) -> torch.Tensor:
        """Copy a borrowed device output into request-owned CPU storage."""

        return output.detach().to(device="cpu", copy=True)

    def _infer_locked(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        token_type_ids: torch.Tensor,
        action_mask: torch.Tensor,
        diffusion_input_noise: torch.Tensor | None,
        history_pixel_values: torch.Tensor | None,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device, dtype=torch.long)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
        validated = self.model._validate_prefix_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            enforce_max_length=False,
        )
        input_ids, attention_mask, token_type_ids = validated
        if input_ids.shape[0] != 1:
            raise ValueError("DM05 fast backend supports only batch size 1.")
        pixel_values = pixel_values.to(
            self.device, dtype=self.vision_runner.input_dtype
        )
        if (history_pixel_values is None) != (history_mask is None):
            raise ValueError(
                "history_pixel_values and history_mask must be provided together."
            )
        if history_pixel_values is not None:
            if not self.history_enabled:
                raise ValueError(
                    "History tensors require a history-enabled DM05 fast runtime."
                )
            assert history_mask is not None
            history_pixel_values = history_pixel_values.to(
                self.device,
                dtype=self.vision_runner.input_dtype,
            )
            history_mask = history_mask.to(self.device, dtype=torch.bool)
        action_mask = action_mask.to(self.device, dtype=self.noise_dtype)
        if diffusion_input_noise is None:
            noise = torch.randn(
                (1, self.suffix_len, self.action_dim),
                device=self.device,
                dtype=self.noise_dtype,
            )
        else:
            noise = diffusion_input_noise.to(
                self.device,
                dtype=self.noise_dtype,
            )
        expected_shape = (1, self.suffix_len, self.action_dim)
        if tuple(noise.shape) != expected_shape:
            raise ValueError(
                f"diffusion_input_noise must be {expected_shape}, "
                f"got {tuple(noise.shape)}."
            )
        expected_action_mask_shape = (1, 1, self.action_dim)
        if tuple(action_mask.shape) != expected_action_mask_shape:
            raise ValueError(
                f"DM05 action_mask must be {expected_action_mask_shape}, "
                f"got {tuple(action_mask.shape)}."
            )

        has_history_pixels = False
        if history_pixel_values is not None:
            assert history_mask is not None
            has_history_pixels = int(history_pixel_values.shape[0]) > 0 and bool(
                history_mask.any().item()
            )
        if has_history_pixels:
            assert history_pixel_values is not None
            assert history_mask is not None
            if int(history_pixel_values.shape[0]) > MAX_HISTORY_IMAGES:
                raise ValueError(
                    f"At most {MAX_HISTORY_IMAGES} history images are supported, "
                    f"got {int(history_pixel_values.shape[0])}."
                )
            return self._run_with_history_without_capture(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                noise=noise,
                action_mask=action_mask,
                history_pixel_values=history_pixel_values,
                history_mask=history_mask,
            )

        request_len = int(input_ids.shape[1])
        bucket_len = self.profiles.resolve_size(
            request_len,
            overflow_policy=self.overflow_policy,
            request_name="Prefix length",
            profile_name="bucket",
        )
        if bucket_len is None:
            image_features, _ = self._encode_vision_features(pixel_values)
            self.dynamic_fallback_count += 1
            return self._run_dynamic_fallback(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image_features=image_features,
                noise=noise,
                action_mask=action_mask,
            )

        image_token_id = int(self.dm05.vlm.model.config.image_token_id)
        feature_tokens = self.num_current_images * int(
            self.dm05.vlm.model.config.mm_tokens_per_image
        )
        if int((input_ids == image_token_id).sum().item()) != feature_tokens:
            raise ValueError(
                "DM05 image token count does not match TRT image features."
            )
        image_features, _ = self._encode_vision_features(pixel_values)
        _, profile = self.profiles.select_or_create(
            request_len,
            lambda size: self._make_and_capture_profile(
                size,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                image_features=image_features,
                noise=noise,
                action_mask=action_mask,
            ),
        )
        profile = cast(DM05GraphProfile, profile)
        graph_runner = cast(CudaGraphRunner, profile.graph_runner)
        static = graph_runner.static_inputs
        static["input_ids"].fill_(self.padding_idx)
        static["input_ids"][:, :request_len].copy_(input_ids)
        static["attention_mask"].zero_()
        static["attention_mask"][:, :request_len].copy_(attention_mask)
        static["token_type_ids"].zero_()
        static["token_type_ids"][:, :request_len].copy_(token_type_ids)
        static["noise"].copy_(noise)
        static["action_mask"].copy_(action_mask)
        static["image_features"].copy_(image_features)
        output = graph_runner.replay()
        self.graph_replay_count += 1
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                "DM05 prefix-prefill and denoising graph must return one Tensor."
            )
        return output
