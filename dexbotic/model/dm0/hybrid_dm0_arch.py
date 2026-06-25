"""Hybrid DM0 Model Architecture for Dexbotic.

This module implements the hybrid version of DM0, which jointly trains:
- Action prediction via flow matching (same as vanilla DM0).
- Language modeling on prefix tokens via cross-entropy (text loss).

The architecture is identical to ``DM0ForCausalLM`` (merged attention between
the VLM Qwen3 backbone and the action expert Qwen3). The only changes are:

1. ``forward`` accepts ``labels`` / ``has_action`` / ``has_text`` and returns a
   combined ``loss = text_loss + action_loss`` weighted by per-sample masks.
2. ``actions`` is treated as optional. When all samples are pure text (no
   action), the suffix forward is skipped. When the batch is mixed, dummy
   zero-actions are expected from the data pipeline (``AddActionFlag``) and the
   ``has_action`` mask zeros out those samples in the action loss.

"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache

from dexbotic.constants import IGNORE_INDEX
from dexbotic.model.dexbotic_arch import CausalLMOutputDexbotic
from dexbotic.model.dm0.dm0_arch import DM0Config, DM0ForCausalLM, DM0Model
from dexbotic.model.dm0.dm0_utils import make_attn_mask_2d, make_attn_mask_4d


class HybridDM0Config(DM0Config):
    """Config for hybrid DM0. Reuses the same fields as DM0Config but registers
    a separate ``model_type`` so that AutoConfig/AutoModel can route correctly
    while still being load-compatible with vanilla DM0 checkpoints."""

    model_type = "dexbotic_hybrid_dm0"


class HybridDM0Model(DM0Model):
    """Same submodules as DM0Model. Subclassed only to bind the new config."""


class HybridDM0ForCausalLM(DM0ForCausalLM):
    """Hybrid DM0: action flow-matching loss + language CE loss.

    Inherits all merged-attention plumbing from ``DM0ForCausalLM`` and only
    overrides ``_real_init`` (to use HybridDM0Model) and ``forward`` (to
    compute joint loss).
    """

    config_class = HybridDM0Config
    _tied_weights_keys = {
        "lm_head.weight": "model.llm.embed_tokens.weight",
    }

    def _real_init(self, config: HybridDM0Config):
        self.model = HybridDM0Model(config)
        if config.bf16:
            self.model.to_bfloat16_for_selected_params()
        else:
            self.model = self.model.to(torch.float32)
        self.lm_head = nn.Linear(
            config.llm_config.hidden_size, config.llm_config.vocab_size, bias=False
        )
        self.post_init()

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        actions: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
        has_action: Optional[torch.Tensor] = None,
        has_text: Optional[torch.Tensor] = None,
        text_loss_weight: float = 1.0,
        action_loss_weight: float = 1.0,
        **kwargs,
    ) -> CausalLMOutputDexbotic:
        """Joint forward for hybrid training.

        ``actions`` is expected to be present (zero-padded for non-action
        samples by the data pipeline). The ``has_action`` mask zeroes out
        those samples' contribution to the action loss. Symmetric handling
        applies to text via ``labels`` (IGNORE_INDEX) and ``has_text``.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # --- Embed prefix (images + language) ------------------------------
        prefix_hidden_states, prefix_padding_mask, prefix_attn_mask = (
            self.get_prefix_hidden_states(
                input_ids, attention_mask, images, image_masks
            )
        )

        suffix_out = None
        prefix_out = None
        u_t = None
        text_logits = None
        action_logits = None

        module_list = [
            self.model.llm,
            self.model.action_expert.model,
        ]

        if actions is not None:
            batch_size = actions.shape[0]

            # --- Flow matching: sample noise & time ------------------------
            noise = torch.normal(
                mean=torch.zeros_like(actions),
                std=torch.ones_like(actions),
            ).to(device=actions.device, dtype=actions.dtype)

            time = (
                torch.distributions.Beta(1.5, 1.0)
                .sample((batch_size,))
                .to(actions.device)
                * 0.999
                + 0.001
            ).to(dtype=actions.dtype)

            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

            # --- Embed suffix (noisy actions + time) -----------------------
            suffix_hidden_states, suffix_padding_mask, suffix_attn_mask = (
                self.get_suffix_hidden_states(x_t, time)
            )

            if self.model.config.bf16:
                suffix_hidden_states = suffix_hidden_states.to(dtype=torch.bfloat16)
                prefix_hidden_states = prefix_hidden_states.to(dtype=torch.bfloat16)

            # --- Build combined attention mask -----------------------------
            full_padding_mask = torch.cat(
                [prefix_padding_mask, suffix_padding_mask], dim=1
            )
            full_attn_mask = torch.cat(
                [prefix_attn_mask, suffix_attn_mask], dim=1
            )
            attn_mask_2d = make_attn_mask_2d(
                padding_mask=full_padding_mask, attn_mask=full_attn_mask
            )
            attn_mask = make_attn_mask_4d(
                attn_mask_2d, dtype=prefix_hidden_states.dtype
            )

            # --- Positions ------------------------------------------------
            prefix_positions = torch.cumsum(prefix_padding_mask, dim=1) - 1
            prefix_offsets = torch.sum(prefix_padding_mask, dim=-1)[:, None]
            suffix_positions = (
                prefix_offsets + torch.cumsum(suffix_padding_mask, dim=1) - 1
            )
            positions = torch.cat([prefix_positions, suffix_positions], dim=1)

            (prefix_out, suffix_out), _ = self._merged_attention_forward(
                module_list=module_list,
                attention_mask=attn_mask,
                position_ids=positions,
                past_key_values=None,
                input_embeds_list=[prefix_hidden_states, suffix_hidden_states],
                use_cache=False,
            )
        else:
            # Pure-text batch: no action expert forward.
            if self.model.config.bf16:
                prefix_hidden_states = prefix_hidden_states.to(dtype=torch.bfloat16)

            attn_mask_2d = make_attn_mask_2d(
                padding_mask=prefix_padding_mask, attn_mask=prefix_attn_mask
            )
            attn_mask = make_attn_mask_4d(
                attn_mask_2d, dtype=prefix_hidden_states.dtype
            )
            positions = torch.cumsum(prefix_padding_mask, dim=1) - 1

            (prefix_out, _), _ = self._merged_attention_forward(
                module_list=module_list,
                attention_mask=attn_mask,
                position_ids=positions,
                past_key_values=None,
                input_embeds_list=[prefix_hidden_states, None],
                use_cache=False,
            )

        # --- Text loss on prefix output -----------------------------------
        text_loss = None
        text_diag = {
            "has_text_mean": None,
            "has_action_mean": None,
            "has_text_sum": None,
            "has_action_sum": None,
            "valid_label_tokens_sum": None,
            "valid_label_tokens_text_sum": None,
            "text_loss_pre_mean": None,
            "text_loss_post_mean": None,
        }
        if labels is not None and input_ids is not None and prefix_out is not None:
            # ``prefix`` layout: [image_tokens..., language_tokens]
            # Only the language portion (last ``text_len`` positions) is used.
            text_len = input_ids.shape[1]
            # Match the dtype expected by lm_head to avoid dtype mismatch
            # when prefix_out is bf16 and lm_head weights are fp32.
            lm_head_dtype = self.lm_head.weight.dtype
            text_hidden = prefix_out[:, -text_len:, :].to(lm_head_dtype)
            text_logits = self.lm_head(text_hidden)

            # Standard LM shift: predict token t+1 from hidden at t.
            target_tokens = labels[:, 1:]
            pred_tokens = text_logits[:, :-1]

            token_loss = F.cross_entropy(
                pred_tokens.transpose(1, 2).float(),
                target_tokens,
                reduction="none",
            )
            token_mask = torch.where(target_tokens != IGNORE_INDEX, 1.0, 0.0)
            sample_loss = (token_loss * token_mask).sum(dim=-1) / torch.clamp(
                token_mask.sum(dim=-1), min=1.0
            )

            if has_text is None:
                has_text_mask = torch.ones(
                    sample_loss.shape[0],
                    device=sample_loss.device,
                    dtype=torch.float32,
                )
            else:
                has_text_mask = has_text.reshape(-1).to(sample_loss.device).float()

            text_loss = (sample_loss * has_text_mask).sum() / (
                has_text_mask.sum() + 1e-6
            )

            valid_tokens_per_sample = token_mask.sum(dim=-1)
            has_action_mask_for_diag = None
            if has_action is not None:
                has_action_mask_for_diag = (
                    has_action.reshape(-1).to(sample_loss.device).float()
                )
            text_diag = {
                "has_text_mean": has_text_mask.mean().detach(),
                "has_action_mean": (
                    has_action_mask_for_diag.mean().detach()
                    if has_action_mask_for_diag is not None
                    else torch.ones((), device=sample_loss.device)
                ),
                "has_text_sum": has_text_mask.sum().detach(),
                "has_action_sum": (
                    has_action_mask_for_diag.sum().detach()
                    if has_action_mask_for_diag is not None
                    else torch.tensor(float(sample_loss.shape[0]), device=sample_loss.device)
                ),
                "valid_label_tokens_sum": valid_tokens_per_sample.sum().detach(),
                "valid_label_tokens_text_sum": (
                    valid_tokens_per_sample * has_text_mask
                ).sum().detach(),
                "text_loss_pre_mean": sample_loss.mean().detach(),
                "text_loss_post_mean": text_loss.detach(),
            }

        # --- Action loss (flow matching MSE) ------------------------------
        action_loss = None
        if suffix_out is not None and u_t is not None:
            if u_t.dtype == torch.float32:
                suffix_out = suffix_out.to(torch.float32)
            suffix_out_final = suffix_out[:, -self.model.config.chunk_size :]
            action_logits = self.model.action_out_proj(suffix_out_final)

            per_sample_action_loss = F.mse_loss(
                action_logits, u_t, reduction="none"
            ).mean(dim=[1, 2])

            if has_action is None:
                has_action_mask = torch.ones(
                    per_sample_action_loss.shape[0],
                    device=per_sample_action_loss.device,
                    dtype=torch.float32,
                )
            else:
                has_action_mask = (
                    has_action.reshape(-1).to(per_sample_action_loss.device).float()
                )

            action_loss = (per_sample_action_loss * has_action_mask).sum() / (
                has_action_mask.sum() + 1e-6
            )

        # --- Combined loss ------------------------------------------------
        loss = None
        text_loss_weight = getattr(self.config, "text_loss_weight", text_loss_weight)
        action_loss_weight = getattr(self.config, "action_loss_weight", action_loss_weight)
        if text_loss is not None and action_loss is not None:
            loss = text_loss_weight * text_loss + action_loss_weight * action_loss
        elif text_loss is not None:
            loss = text_loss_weight * text_loss
        elif action_loss is not None:
            loss = action_loss_weight * action_loss

        logits = action_logits if action_logits is not None else text_logits
        if not return_dict:
            return (loss, logits, past_key_values, None, None)

        return CausalLMOutputDexbotic(
            loss=loss,
            text_loss=text_loss,
            action_loss=action_loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
