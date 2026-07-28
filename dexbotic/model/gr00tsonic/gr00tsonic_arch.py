# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# GR00T N1.7 ("gr00tsonic") migrated into Dexbotic.
#
# The architecture is kept faithful to the upstream Isaac-GR00T Gr00tN1d7 model:
#   * a monolithic Qwen3-VL (Cosmos-Reason2-2B) vision-language backbone, and
#   * a flow-matching DiT action head (AlternateVLDiT by default).
#
# It is wrapped in Dexbotic's conventions (DexboticConfig / DexboticForCausalLM /
# ActionOutputForCausalLM) the same way pi0/pi05 are, exposing a training
# ``forward`` and an ``inference_action`` entry point.

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from dexbotic.model.dexbotic_arch import (
    ActionOutputForCausalLM,
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticPretrainedModel,
    DexboticForCausalLM,
)
from dexbotic.model.gr00tsonic.action_head import Gr00tSonicActionHead
from dexbotic.model.gr00tsonic.modules.qwen3_backbone import Qwen3Backbone


# Default flow-matching DiT config — matches the released GR00T-N1.7 checkpoint
# config.json (NOT the smaller Gr00tN1d7Config dataclass default of 16 layers).
_DEFAULT_DIFFUSION_MODEL_CFG = {
    "positional_embeddings": None,
    "num_layers": 32,
    "num_attention_heads": 32,
    "attention_head_dim": 48,
    "norm_type": "ada_norm",
    "dropout": 0.2,
    "final_dropout": True,
    "output_dim": 1024,
    "interleave_self_attention": True,
}

# VL self-attention transformer — matches the released GR00T-N1.7 checkpoint
# (4 layers, inner_dim 32*64=2048). The dataclass default omits this entirely.
_DEFAULT_VL_SELF_ATTENTION_CFG = {
    "num_attention_heads": 32,
    "attention_head_dim": 64,
    "num_layers": 4,
    "dropout": 0.2,
    "final_dropout": True,
    "positional_embeddings": None,
}


class GR00TSonicConfig(DexboticConfig):
    """Configuration for the GR00T N1.7 (gr00tsonic) model.

    Mirrors the fields of Isaac-GR00T's ``Gr00tN1d7Config`` so checkpoints map
    one-to-one, while subclassing ``DexboticConfig`` for Dexbotic integration.
    """

    model_type = "dexbotic_gr00tsonic"

    # Defaults mirror upstream Gr00tN1d7Config.
    _DEFAULTS = {
        # dtype / backbone identification
        "model_dtype": "bfloat16",
        "model_name": "nvidia/Cosmos-Reason2-2B",
        "backbone_model_type": "qwen",
        "model_revision": None,
        # backbone tuning / sizing
        "tune_top_llm_layers": 0,
        "backbone_embedding_dim": 2048,
        "tune_llm": False,
        "tune_visual": False,
        "select_layer": 16,
        "reproject_vision": False,
        "use_flash_attention": True,
        "load_bf16": False,
        "backbone_trainable_params_fp32": True,
        # When True the Qwen3-VL base weights are pulled with from_pretrained on
        # construction; set False to build an empty backbone from config and let
        # a Dexbotic checkpoint provide the weights.
        "load_backbone_pretrained": True,
        # action head sizing
        "max_state_dim": 132,
        "max_action_dim": 132,
        "action_horizon": 40,
        "hidden_size": 1024,
        "input_embedding_dim": 1536,
        "state_history_length": 1,
        # global action-head parameters
        "add_pos_embed": True,
        "attn_dropout": 0.2,
        "use_vlln": True,
        "max_seq_len": 1024,
        "use_alternate_vl_dit": True,
        "attend_text_every_n_blocks": 2,
        "diffusion_model_cfg": None,  # filled with _DEFAULT_DIFFUSION_MODEL_CFG below
        "vl_self_attention_cfg": None,
        # flow matching parameters
        "num_inference_timesteps": 4,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        # training parameters
        "tune_projector": True,
        "tune_diffusion_model": True,
        "tune_vlln": True,
        # state augmentation / normalization (released GR00T-N1.7 ckpt uses 0.2)
        "state_dropout_prob": 0.2,
        "exclude_state": False,
        "use_mean_std": False,
        # multi-embodiment
        "max_num_embodiments": 32,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, default in self._DEFAULTS.items():
            setattr(self, key, kwargs.get(key, getattr(self, key, default)))
        if not getattr(self, "diffusion_model_cfg", None):
            self.diffusion_model_cfg = dict(_DEFAULT_DIFFUSION_MODEL_CFG)
        if not getattr(self, "vl_self_attention_cfg", None):
            self.vl_self_attention_cfg = dict(_DEFAULT_VL_SELF_ATTENTION_CFG)


class GR00TSonicModel(DexboticPretrainedModel):
    """Backbone (Qwen3-VL) + flow-matching action head container."""

    config_class = GR00TSonicConfig

    def __init__(self, config: GR00TSonicConfig):
        super().__init__(config)
        self.backbone = Qwen3Backbone(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            init_pretrained=getattr(config, "load_backbone_pretrained", True),
        )
        self.action_head = Gr00tSonicActionHead(config)
        # NOTE: we deliberately do NOT call post_init() here — it would re-init
        # the Qwen3-VL backbone weights loaded via from_pretrained.

    def set_trainable_parameters(
        self,
        tune_llm: bool,
        tune_visual: bool,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_vlln: bool,
    ):
        self.backbone.set_trainable_parameters(
            tune_llm=tune_llm,
            tune_visual=tune_visual,
            tune_top_llm_layers=self.config.tune_top_llm_layers,
        )
        self.action_head.set_trainable_parameters(
            tune_projector=tune_projector,
            tune_diffusion_model=tune_diffusion_model,
            tune_vlln=tune_vlln,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


class GR00TSonicForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = GR00TSonicConfig

    def _real_init(self, config: GR00TSonicConfig):
        self.model = GR00TSonicModel(config)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _build_backbone_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ) -> BatchFeature:
        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        )

    def _cast_float_inputs(self, *tensors):
        dtype = self.dtype
        return tuple(
            t.to(dtype=dtype) if isinstance(t, torch.Tensor) else t for t in tensors
        )

    def _build_action_inputs(
        self,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        action_mask: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        data = {"state": state, "embodiment_id": embodiment_id}
        if action is not None:
            data["action"] = action
        if action_mask is not None:
            data["action_mask"] = action_mask
        return BatchFeature(data=data)

    # ── training ─────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        state: Optional[torch.FloatTensor] = None,
        action: Optional[torch.FloatTensor] = None,
        action_mask: Optional[torch.FloatTensor] = None,
        embodiment_id: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputDexbotic:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        pixel_values, state, action, action_mask = self._cast_float_inputs(
            pixel_values, state, action, action_mask
        )

        backbone_inputs = self._build_backbone_inputs(
            input_ids, attention_mask, pixel_values, image_grid_thw
        )
        backbone_outputs = self.model.backbone(backbone_inputs)

        action_inputs = self._build_action_inputs(
            state=state,
            embodiment_id=embodiment_id,
            action=action,
            action_mask=action_mask,
        )
        action_outputs = self.model.action_head(backbone_outputs, action_inputs)

        loss = action_outputs["loss"]
        if not return_dict:
            return (loss,)
        # NOTE: action_loss/logits must be scalars — DexboticTrainer logs every
        # output key ending in "_loss" via .item()/torch.isclose, which is
        # ambiguous on the per-element action-loss tensor.
        return CausalLMOutputDexbotic(
            loss=loss,
            logits=loss.detach(),
            action_loss=loss.detach(),
        )

    # ── inference ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        state: Optional[torch.FloatTensor] = None,
        embodiment_id: Optional[torch.LongTensor] = None,
        action_mask: Optional[torch.FloatTensor] = None,
        action: Optional[torch.FloatTensor] = None,
        options: Optional[dict[str, Any]] = None,
        inference_args: dict = {},
        **kwargs,
    ):
        pixel_values, state, action, action_mask = self._cast_float_inputs(
            pixel_values, state, action, action_mask
        )
        backbone_inputs = self._build_backbone_inputs(
            input_ids, attention_mask, pixel_values, image_grid_thw
        )
        backbone_outputs = self.model.backbone(backbone_inputs)

        action_inputs = self._build_action_inputs(
            state=state,
            embodiment_id=embodiment_id,
            action=action,
            action_mask=action_mask,
        )
        action_outputs = self.model.action_head.get_action(
            backbone_outputs, action_inputs, options
        )
        return action_outputs["action_pred"]  # [B, action_horizon, max_action_dim]

    # gr00tsonic uses the Qwen3-VL processor (in the policy) for image handling,
    # not Dexbotic's siglip vision tower, so the inherited process_images path is
    # intentionally unused.
    def process_images(self, images):
        raise NotImplementedError(
            "gr00tsonic processes images via the Qwen3-VL processor in the policy; "
            "GR00TSonicForCausalLM.process_images is not used."
        )


def load_pretrained_gr00t(
    model: "GR00TSonicForCausalLM", ckpt_path: str, verbose: bool = True
):
    """Initialize a gr00tsonic model from an original Isaac-GR00T (Gr00tN1d7) checkpoint.

    The original model stores params under ``backbone.*`` / ``action_head.*``; the
    Dexbotic model wraps both inside ``GR00TSonicModel`` (named ``model``), so the
    only difference is a ``model.`` prefix. We remap and load with strict=False,
    then assert nothing important is missing.

    Requires the gr00tsonic config to match the original exactly (DiT layers,
    vl_self_attention, select_layer, dims) — otherwise shapes won't line up.
    """
    import glob
    import json
    import os

    from safetensors.torch import load_file

    # Accept either a local checkpoint dir or an HF repo id (e.g.
    # "nvidia/GR00T-N1.7-3B") resolved from the local HF cache.
    if not os.path.isdir(ckpt_path):
        from huggingface_hub import snapshot_download

        ckpt_path = snapshot_download(
            ckpt_path,
            local_files_only=True,
            allow_patterns=["*.safetensors", "*.json"],
        )

    idx_path = os.path.join(ckpt_path, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        weight_map = json.load(open(idx_path))["weight_map"]
        shard_files = sorted({os.path.join(ckpt_path, v) for v in weight_map.values()})
    else:
        shard_files = sorted(glob.glob(os.path.join(ckpt_path, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors found under {ckpt_path}")

    state = {}
    for f in shard_files:
        state.update(load_file(f))

    remapped = {f"model.{k}": v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    # Buffers like rotary inv_freq are non-persistent and legitimately absent;
    # flag any *parameter* that did not get a value.
    own = dict(model.named_parameters())
    missing_params = [k for k in missing if k in own]
    if verbose:
        print(
            f"[load_pretrained_gr00t] loaded {len(remapped)} tensors from {ckpt_path}; "
            f"missing={len(missing)} (params={len(missing_params)}), "
            f"unexpected={len(unexpected)}"
        )
        if missing_params:
            print(f"[load_pretrained_gr00t] MISSING PARAMS (first 10): {missing_params[:10]}")
        if unexpected:
            print(f"[load_pretrained_gr00t] UNEXPECTED (first 10): {list(unexpected)[:10]}")
    if missing_params:
        raise RuntimeError(
            f"{len(missing_params)} model parameters were not initialized from the "
            f"checkpoint — config likely does not match the original architecture."
        )
    return missing, unexpected


# Make the config / model discoverable through the HF Auto* registry.
AutoConfig.register("dexbotic_gr00tsonic", GR00TSonicConfig)
AutoModel.register(GR00TSonicConfig, GR00TSonicForCausalLM)
