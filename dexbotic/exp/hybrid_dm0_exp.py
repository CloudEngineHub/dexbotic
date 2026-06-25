"""Hybrid DM0 experiment configuration.

This file enables DM0 co-training on mixed robot/VLA and general VLM data.
It follows the continuous-action hybrid pattern in ``hybrid_pi05_exp.py`` while
preserving DM0-specific defaults: 3-view input, 32D padded state/action tensors,
Flow Matching action loss, DM0 tokenization, and the DM0 FSDP2 profile.

Key behavior:
  * robot/VLA data contributes action loss through ``has_action``;
  * VLM conversation data contributes text loss through ``has_text``;
  * norm stats are computed from robot/action data only via ``norm_dataset_name``;
  * the LIBERO VLA action path intentionally matches
    ``playground/benchmarks/libero/libero_dm0.py``: preserve dataset-provided
    action fields, and only synthesize actions for action-less state data.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import megfile
import numpy as np
import torch
import transformers
import dexbotic.data.data_source  # noqa: F401 - register built-in data sources
from easydict import EasyDict
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer, BaseImageProcessor, PretrainedConfig

from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddAction,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import (
    AddActionFlag,
    AddStateFlag,
    Pipeline,
    ToDict,
    ToList,
    ToNumpy,
    ToTensor,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.exp.backend_resolver import FSDPProfile
from dexbotic.exp.base_exp import BaseExp
from dexbotic.exp.dm0_exp import (
    DM0ActionConfig,
    DM0ComputeNormActionConfig,
    DM0DataConfig,
    DM0InferenceConfig,
    DM0ModelConfig,
    DM0OptimizerConfig,
    DM0TokenizerConfig,
    DM0TrainerConfig,
)
from dexbotic.model.dm0.hybrid_dm0_arch import HybridDM0Config, HybridDM0ForCausalLM
from dexbotic.tokenization.process import DM0Tokenization


def _action_use_quantiles(default: bool = False) -> bool:
    value = os.environ.get("DEXBOTIC_ACTION_USE_QUANTILES")
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


class AddActionIfMissing(AddAction):
    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if "action" in episode_data_dict or "state" not in episode_data_dict:
            return episode_data_dict
        return super().__call__(episode_data_dict, **kwargs)


class HybridDM0AddTextFlag:
    def __init__(self, enable: bool = True):
        self.enable = enable

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        if not self.enable:
            return episode_data_dict
        if "has_text" not in episode_data_dict:
            episode_data_dict["has_text"] = np.array(
                [self._has_text_supervision(episode_data_dict)], dtype=bool
            )
        return episode_data_dict

    @staticmethod
    def _has_text_supervision(episode_data_dict: dict) -> bool:
        def non_empty(value) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple)):
                return any(non_empty(item) for item in value)
            return True

        if non_empty(episode_data_dict.get("answer")):
            return True

        conversations = episode_data_dict.get("conversations")
        if isinstance(conversations, dict):
            conversations = [conversations]
        if isinstance(conversations, (list, tuple)):
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("from", turn.get("role", ""))).lower()
                value = turn.get("value", turn.get("content"))
                if role in {"gpt", "assistant"} and non_empty(value):
                    return True
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    parser.add_argument(
        "--train-backend",
        type=str,
        default=None,
        choices=["deepspeed", "fsdp", "fsdp2", "ddp"],
    )
    args, _unknown = parser.parse_known_args()
    return args


@dataclass
class HybridDM0ModelConfig(DM0ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/DM0-base")
    action_hidden_size: int = field(
        default_factory=lambda: int(os.environ.get("DEXBOTIC_ACTION_HIDDEN_SIZE", "1024"))
    )
    action_intermediate_size: int = field(
        default_factory=lambda: int(os.environ.get("DEXBOTIC_ACTION_INTERMEDIATE_SIZE", "1536"))
    )
    text_loss_weight: float = field(
        default_factory=lambda: float(os.environ.get("DEXBOTIC_TEXT_LOSS_WEIGHT", "1.0"))
    )
    action_loss_weight: float = field(
        default_factory=lambda: float(os.environ.get("DEXBOTIC_ACTION_LOSS_WEIGHT", "1.0"))
    )

    def _synthesize_action_config(self, llm_config: PretrainedConfig) -> PretrainedConfig:
        llm_model_type = getattr(llm_config, "model_type", None)
        if llm_model_type != "qwen3":
            raise ValueError(
                "HybridDM0 can synthesize an action expert only for Qwen3 "
                f"backbones, got llm_config.model_type={llm_model_type!r}"
            )
        head_dim = getattr(llm_config, "head_dim", None)
        if head_dim is None:
            head_dim = llm_config.hidden_size // llm_config.num_attention_heads

        cfg_dict = {
            "model_type": llm_model_type,
            "vocab_size": getattr(llm_config, "vocab_size", 151936),
            "hidden_size": self.action_hidden_size,
            "intermediate_size": self.action_intermediate_size,
            "num_hidden_layers": llm_config.num_hidden_layers,
            "num_attention_heads": llm_config.num_attention_heads,
            "num_key_value_heads": llm_config.num_key_value_heads,
            "head_dim": head_dim,
            "hidden_act": getattr(llm_config, "hidden_act", "silu"),
            "max_position_embeddings": getattr(llm_config, "max_position_embeddings", 40960),
            "initializer_range": getattr(llm_config, "initializer_range", 0.02),
            "rms_norm_eps": getattr(llm_config, "rms_norm_eps", 1e-6),
            "use_cache": getattr(llm_config, "use_cache", True),
            "tie_word_embeddings": False,
            "rope_theta": getattr(llm_config, "rope_theta", 1000000.0),
            "rope_scaling": getattr(llm_config, "rope_scaling", None),
            "attention_dropout": getattr(llm_config, "attention_dropout", 0.0),
        }
        for optional_key in ("use_sliding_window", "sliding_window", "max_window_layers"):
            if hasattr(llm_config, optional_key):
                cfg_dict[optional_key] = getattr(llm_config, optional_key)
        if hasattr(llm_config, "layer_types"):
            cfg_dict["layer_types"] = getattr(llm_config, "layer_types")
        return type(llm_config)(**cfg_dict)

    def _prepare_hybrid_config(self) -> tuple[HybridDM0Config, bool]:
        config = HybridDM0Config.from_pretrained(self.model_name_or_path)
        config.text_loss_weight = self.text_loss_weight
        config.action_loss_weight = self.action_loss_weight
        logger.info(
            "HybridDM0 loss weights: text_loss_weight={}, action_loss_weight={}",
            config.text_loss_weight,
            config.action_loss_weight,
        )
        llm_config = getattr(config, "llm_config", None)
        checkpoint_vocab_size = getattr(config, "vocab_size", None)
        if (
            checkpoint_vocab_size is not None
            and llm_config is not None
            and getattr(llm_config, "vocab_size", None) != checkpoint_vocab_size
        ):
            logger.info(
                "Aligning llm_config.vocab_size with checkpoint vocab_size: "
                f"{getattr(llm_config, 'vocab_size', None)} -> {checkpoint_vocab_size}"
            )
            llm_config.vocab_size = checkpoint_vocab_size
        if getattr(config, "action_config", None) is None:
            config.action_config = self._synthesize_action_config(llm_config)
            config.model_type = HybridDM0Config.model_type
            config.architectures = ["HybridDM0ForCausalLM"]
            config.bf16 = getattr(config, "bf16", True)
            config.action_dim = getattr(config, "action_dim", 32)
            config.chunk_size = getattr(config, "chunk_size", 50)
            logger.info(
                "Synthesized HybridDM0 action_config from plain Dexbotic base: "
                f"llm_type={getattr(llm_config, 'model_type', None)}, "
                f"action_hidden_size={self.action_hidden_size}, "
                f"action_intermediate_size={self.action_intermediate_size}"
            )
            return config, True
        return config, False

    def build_model(self) -> HybridDM0ForCausalLM:
        config, synthesized = self._prepare_hybrid_config()
        kwargs = {"config": config}
        if synthesized:
            kwargs["ignore_mismatched_sizes"] = True
        model = HybridDM0ForCausalLM.from_pretrained(self.model_name_or_path, **kwargs)
        return model


@dataclass
class HybridDM0TrainerConfig(DM0TrainerConfig):
    train_backend: str = field(default="fsdp2")
    fsdp_profile: FSDPProfile = field(
        default_factory=lambda: FSDPProfile(
            enabled=True,
            cpu_ram_efficient_loading=False,
            transformer_layer_cls_to_wrap=("Qwen3MLP",),
            cast_model_to_bf16_backends=("fsdp", "fsdp2"),
        )
    )
    output_dir: str = field(default="./output/hybrid_dm0_fsdp2")
    model_max_length: int = field(default=256)
    num_train_steps: int = field(default=800)
    save_steps: int = field(default=800)
    save_total_limit: int = field(default=2)
    per_device_train_batch_size: int = field(default=2)
    gradient_accumulation_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    wandb_project: str = field(default="hybrid_dm0_cotrain")


@dataclass
class HybridDM0OptimizerConfig(DM0OptimizerConfig):
    pass


@dataclass
class HybridDM0TokenizerConfig(DM0TokenizerConfig):
    pass


@dataclass
class HybridDM0ComputeNormActionConfig(DM0ComputeNormActionConfig):
    """DM0 norm stats are robot/action-only and must not include VLM data."""

    def build_action_process_func(self) -> Pipeline:
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddActionIfMissing(predict_length=1),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(
                    trajectory_length=50,
                    flatten=False,
                    padding_mode="last",
                ),
                DeltaAction(enable=True),
                ToList(),
            ]
        )


@dataclass
class HybridDM0ActionConfig(DM0ActionConfig):
    statistic_mapping: Optional[str] = field(default=None)
    trajectory_length: int = field(default=50)

    def _read_norm_stats(self, norm_stats_path):
        assert megfile.smart_exists(
            norm_stats_path
        ), f"Norm stats file {norm_stats_path} not found"
        with megfile.smart_open(norm_stats_path, "r") as f:
            payload = json.load(f)
        return ToNumpy()(payload.get("norm_stats", payload))

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        use_quantiles = _action_use_quantiles(False)
        logger.info(f"HybridDM0 action normalization use_quantiles={use_quantiles}")
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddActionIfMissing(predict_length=1),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    flatten=False,
                    padding_mode="last",
                ),
                DeltaAction(enable=True),
                ActionNorm(
                    statistic_mapping=statistic_mapping,
                    strict=False,
                    use_quantiles=use_quantiles,
                ),
                LoadMultiModal(return_masks=True),
                ToList(select_frame=True),
                AddStateFlag(empty_state_value=np.zeros(32, dtype=np.float32)),
                AddActionFlag(
                    empty_action_value=np.zeros(
                        (self.trajectory_length, 32), dtype=np.float32
                    )
                ),
                HybridDM0AddTextFlag(),
            ]
        )
        return action_config


@dataclass
class HybridDM0DataConfig(DM0DataConfig):
    norm_dataset_name: Optional[str] = field(default=None)
    num_images: int = field(default=3)
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
            "state",
            "image_masks",
            "has_action",
            "has_text",
        ]
    )
    aug_policy: str | list[str] = field(
        default_factory=lambda: ["dm0", "color_dm0", "color_dm0"]
    )
    action_config: HybridDM0ActionConfig = field(default_factory=HybridDM0ActionConfig)
    image_pad_mode: str = field(default="zero")

    def robot_norm_dataset_name(self) -> str:
        if not self.norm_dataset_name:
            raise ValueError(
                "HybridDM0DataConfig.norm_dataset_name must be set explicitly. "
                "Use a robot/action dataset name for norm stats, not the mixed "
                "training dataset_name."
            )
        return self.norm_dataset_name

    def _log_dataset_mix(self, dataset: DexDataset) -> None:
        names = self.dataset_name.split("+")
        rows = []
        total_estimated = 0.0
        vla_estimated = 0.0
        vlm_estimated = 0.0
        for name, dataset_info in zip(names, dataset.datasets_info, strict=False):
            index_cache = dataset._get_index_cache(dataset_info["annotations"])
            base_samples = index_cache.get("meta_data", {}).get("total_samples", 0)
            frequency = float(dataset_info.get("frequency", 1))
            estimated_samples = base_samples * frequency
            is_vlm = name.startswith(("cambrian", "vlm"))
            if is_vlm:
                vlm_estimated += estimated_samples
            else:
                vla_estimated += estimated_samples
            total_estimated += estimated_samples
            rows.append((name, base_samples, frequency, estimated_samples, "VLM" if is_vlm else "VLA"))

        logger.info(
            "HybridDM0 dataset mix summary: dataset_name={} total_indexed_samples={}",
            self.dataset_name,
            len(dataset),
        )
        for name, base_samples, frequency, estimated_samples, modality in rows:
            share = estimated_samples / total_estimated if total_estimated else 0.0
            logger.info(
                "HybridDM0 dataset component: name={} modality={} base_samples={} frequency={} estimated_samples={:.0f} estimated_share={:.2%}",
                name,
                modality,
                base_samples,
                frequency,
                estimated_samples,
                share,
            )
        if vlm_estimated > 0:
            logger.info(
                "HybridDM0 estimated VLA:VLM sample ratio = {:.2f}:1.00 (vla_estimated={:.0f}, vlm_estimated={:.0f})",
                vla_estimated / vlm_estimated,
                vla_estimated,
                vlm_estimated,
            )
        else:
            logger.info(
                "HybridDM0 estimated VLA:VLM sample ratio = action-only (no VLM dataset)"
            )

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexDataset:
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = DM0Tokenization(tokenizer, chat_template="step")
        data_args = EasyDict(
            {
                "dataset_name": self.dataset_name,
                "num_images": self.num_images,
                "data_keys": self.data_keys,
                "images_keys": self.images_keys,
                "aug_policy": self.aug_policy,
                "image_aspect_ratio": self.image_aspect_ratio,
                "image_processor": image_processor,
                "chat_template": chat_template,
                "image_pad_mode": self.image_pad_mode,
            }
        )
        dataset = DexDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )
        self._log_dataset_mix(dataset)
        return dataset


@dataclass
class HybridDM0InferenceConfig(DM0InferenceConfig):
    model_name_or_path: Optional[str] = field(default="./checkpoints/libero/DM0-libero")

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        config, synthesized = HybridDM0ModelConfig(
            model_name_or_path=self.model_name_or_path
        )._prepare_hybrid_config()
        model = HybridDM0ForCausalLM.from_pretrained(
            self.model_name_or_path,
            config=config,
            ignore_mismatched_sizes=synthesized,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=os.environ.get("DEXBOTIC_USE_FAST_TOKENIZER", "false").lower()
            in {"1", "true", "yes"},
            trust_remote_code=True,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = DM0Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        use_quantiles = _action_use_quantiles(False)
        logger.info(f"HybridDM0 inference action normalization use_quantiles={use_quantiles}")
        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=use_quantiles
                ),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=use_quantiles
                ),
                AbsoluteAction(),
            ]
        )

    def _get_response(
        self,
        text: str | list[str],
        images: list[str],
        states: Optional[str | list[str]] = None,
        batch_size: int = 1,
    ) -> str:
        t0 = time.monotonic()
        batch_size = int(batch_size)
        assert len(images) % batch_size == 0, (
            f"Number of images {len(images)} is not divisible by batch size {batch_size}"
        )
        num_images = len(images) // batch_size
        images = [
            images[i * num_images : (i + 1) * num_images] for i in range(batch_size)
        ]
        if isinstance(text, str):
            text = [text] * batch_size

        batch_images = [
            [Image.open(i).convert("RGB") for i in image_items]
            for image_items in images
        ]
        batch_images_tensor = [
            self.model.process_images(image_items).to(dtype=self.model.dtype)
            for image_items in batch_images
        ]

        effective_num_images = min(num_images, self.num_images)
        if num_images != self.num_images:
            batch_images_tensor = [
                torch.cat(
                    [
                        image_tensor,
                        torch.zeros_like(image_tensor[0:1]).repeat(
                            self.num_images - num_images, 1, 1, 1
                        ),
                    ],
                    dim=0,
                )
                if len(image_tensor) < self.num_images
                else image_tensor[: self.num_images]
                for image_tensor in batch_images_tensor
            ]

        batch_image_masks = [
            torch.tensor(
                [True for _ in range(effective_num_images)]
                + [False for _ in range(self.num_images - effective_num_images)],
                device=image_tensor.device,
            )
            for image_tensor in batch_images_tensor
        ]
        batch_images_tensor = torch.stack(batch_images_tensor, dim=0)
        batch_image_masks = torch.stack(batch_image_masks, dim=0)

        self._save_image(images[0], text[0])

        batch_input_ids = np.array(
            [self.tokenization_func([{"from": "human", "value": p}])["input_ids"] for p in text]
        )
        logger.info(f"prompt: {self.tokenization_func.tokenizer.decode(batch_input_ids[0])}")
        batch_attention_mask = np.array(
            [np.array(ids != self.tokenizer.pad_token_id) for ids in batch_input_ids]
        )

        if states is not None:
            if isinstance(states, str):
                batch_states = np.array(json.loads(states))
                if batch_states.ndim == 1:
                    batch_states = batch_states[None]
                assert batch_states.shape[0] == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got length {len(batch_states)}."
                )
            elif isinstance(states, (list, tuple)) and all(
                isinstance(s, str) for s in states
            ):
                assert len(states) == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got {type(states)} with length {len(states)}."
                )
                batch_states = [json.loads(s) for s in states]
                batch_states = np.array(batch_states)
        else:
            batch_states = np.zeros(
                (
                    batch_size,
                    self.model.model.config.action_dim,
                ),
                dtype=np.float32,
            )

        inference_args = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": batch_images_tensor,
            "image_masks": batch_image_masks,
            "state": batch_states,
            "meta_data": {
                "non_delta_mask": np.array(self.non_delta_mask),
            },
        }

        inputs = self.input_transform(inference_args)
        inputs["states"] = inputs["state"]
        model_dtype = getattr(self.model, "dtype", torch.float32)
        inputs = {
            k: (
                v.to(self.device, dtype=model_dtype)
                if v.is_floating_point()
                else v.to(self.device)
            )
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
        }
        if self.device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16):
            with torch.autocast(device_type="cuda", dtype=model_dtype):
                actions = self.model.inference_action(**inputs)
        else:
            actions = self.model.inference_action(**inputs)
        outputs = {
            k: (
                v.detach().float().cpu().numpy()
                if v.is_floating_point()
                else v.detach().cpu().numpy()
            )
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
        }
        outputs["action"] = actions.detach().float().cpu().numpy()
        outputs = self.output_transform(outputs)
        logger.info(f"Processing time: {time.monotonic() - t0}")
        action_output = outputs["action"][..., : self.action_dim]
        if batch_size == 1:
            action_output = action_output[0]
        return action_output.tolist()


@dataclass
class HybridDM0Exp(BaseExp):
    model_config: HybridDM0ModelConfig = field(default_factory=HybridDM0ModelConfig)
    optimizer_config: HybridDM0OptimizerConfig = field(
        default_factory=HybridDM0OptimizerConfig
    )
    trainer_config: HybridDM0TrainerConfig = field(
        default_factory=HybridDM0TrainerConfig
    )
    data_config: HybridDM0DataConfig = field(default_factory=HybridDM0DataConfig)
    tokenizer_config: HybridDM0TokenizerConfig = field(
        default_factory=HybridDM0TokenizerConfig
    )
    inference_config: HybridDM0InferenceConfig = field(
        default_factory=HybridDM0InferenceConfig
    )

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        dataset_name = self.data_config.robot_norm_dataset_name()
        self.data_config.action_config = HybridDM0ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(dataset_name)

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        dataset_name = self.data_config.robot_norm_dataset_name()
        if self.local_rank == 0:
            print(
                f"Action config before auto compute norm: {self.data_config.action_config}"
            )
            print(f"Computing hybrid DM0 norm stats from robot dataset: {dataset_name}")
        _action_config = self.data_config.action_config
        norm_config = HybridDM0ComputeNormActionConfig()
        save_name = hashlib.md5(dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing norm stats on rank0")
            norm_config.compute_norm_stats(dataset_name)
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f"Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}"
                )
        _action_config.statistic_mapping = norm_file_path
        self.data_config.action_config = _action_config
        if self.local_rank == 0:
            print(
                f"Action config after auto compute norm: {self.data_config.action_config}"
            )


if __name__ == "__main__":
    args = parse_args()
    exp = HybridDM0Exp()
    if args.train_backend is not None:
        exp.trainer_config.train_backend = args.train_backend
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
