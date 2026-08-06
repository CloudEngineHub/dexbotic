import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import megfile
import torch
from loguru import logger
from transformers import AutoTokenizer

from dexbotic.data.dataset.transform.action import ActionNormAnd2String, AddTrajectory
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.language import AddPromptTemplate, ReplaceAnswer
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.base_exp import ComputeNormActionConfig
from dexbotic.exp.cogact_exp import CogACTActionConfig as _FullActionConfig
from dexbotic.exp.cogact_exp import CogACTOptimizerConfig as _FullOptimizerConfig
from dexbotic.exp.cogact_lora import CogACTLoraConfig as _CogACTLoraConfig
from dexbotic.exp.cogact_lora import (
    apply_lora_to_cogact_model,
    load_cogact_lora_model_for_inference,
    resolve_cogact_tokenizer_path,
)
from dexbotic.exp.lora_utils import is_lora_checkpoint
from dexbotic.model.cogact.cogact_arch import CogACTForCausalLM
from playground.benchmarks.libero.libero_cogact import (
    LiberoCogActDataConfig as _FullDataConfig,
)
from playground.benchmarks.libero.libero_cogact import LiberoCogActExp as _FullExp
from playground.benchmarks.libero.libero_cogact import (
    LiberoCogActInferenceConfig as _FullInferenceConfig,
)
from playground.benchmarks.libero.libero_cogact import (
    LiberoCogActModelConfig as _FullModelConfig,
)
from playground.benchmarks.libero.libero_cogact import (
    LiberoCogActTrainerConfig as _FullTrainerConfig,
)


class _ValidateRawActionDim:
    def __init__(self, expected_dim: int):
        self.expected_dim = expected_dim

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        action = episode_data_dict.get("action")
        if action is None:
            return episode_data_dict
        if action.ndim == 0 or action.shape[-1] != self.expected_dim:
            source = episode_data_dict.get("meta_data", {}).get(
                "jsonl_file", "<unknown>"
            )
            actual_dim = None if action.ndim == 0 else action.shape[-1]
            raise ValueError(
                f"{source}: CogACT raw action dim must be {self.expected_dim}, "
                f"got {actual_dim}"
            )
        return episode_data_dict


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
        "--trainer_backend",
        dest="train_backend",
        type=str,
        default="ddp",
        choices=["ddp"],
        help="CogACT LoRA SFT is validated and supported only with DDP.",
    )
    parser.add_argument(
        "--model_name_or_path",
        "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default=None,
        help="Full checkpoint or LoRA adapter checkpoint used by inference.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        "--base-model-name-or-path",
        dest="base_model_name_or_path",
        type=str,
        default=None,
        help="Optional base model path override for LoRA inference.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Inference service port.",
    )
    args, _ = parser.parse_known_args()
    return args


@dataclass
class CogACTLoraConfig(_CogACTLoraConfig):
    r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    target_modules: list[str] | str = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list[str] = field(default_factory=lambda: ["action_head"])
    dump_trainable_path: Optional[str] = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_cogact/trainable_summaries/"
            "cogact_lora_sft_libero_150k.json"
        )
    )


@dataclass
class CogACTModelConfig(_FullModelConfig):
    model_name_or_path: str = field(default="checkpoints/Dexbotic-Base")
    action_dim: int = field(default=7)
    chunk_size: int = field(default=16)
    lora_config: _CogACTLoraConfig = field(default_factory=CogACTLoraConfig)

    def build_model(self) -> CogACTForCausalLM:
        model = super().build_model()
        return apply_lora_to_cogact_model(
            model,
            self.lora_config,
            base_model_name_or_path=self.model_name_or_path,
        )


@dataclass
class CogACTTrainerConfig(_FullTrainerConfig):
    train_backend: str = field(default="ddp")
    deepspeed: Optional[str] = field(default=None)
    output_dir: str = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_cogact/" "cogact_lora_sft_libero_150k"
        )
    )
    wandb_project: str = field(default="cogact-libero-lora-sft")
    num_train_steps: int = field(default=150000)
    num_train_epochs: int = field(default=25)
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=100)
    save_only_model: bool = field(default=True)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    logging_steps: int = field(default=1)
    model_max_length: int = field(default=1024)


@dataclass
class CogACTOptimizerConfig(_FullOptimizerConfig):
    base_lr: float = field(default=5e-4)
    warmup_steps: int = field(default=500)
    adam_beta2: float = field(default=0.95)
    weight_decay: float = field(default=1e-10)


@dataclass
class CogACTActionConfig(_FullActionConfig):
    delta: bool = field(default=False)
    expected_action_dim: int = field(default=7)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                _ValidateRawActionDim(self.expected_action_dim),
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    padding_mode=self.trajectory_padding_model,
                    padding_action=self.padding_action,
                ),
                ActionNormAnd2String(
                    statistic_mapping=statistic_mapping,
                    vocab_size=self.vocab_size,
                    string_format=self.string_format,
                    add_answer=False,
                ),
                LoadMultiModal(),
                AddPromptTemplate(prompt_template=self.prompt_template),
                ReplaceAnswer(
                    default_answer=self.replace_with_default_answer,
                    replace_existing=True,
                ),
                ToList(),
            ]
        )


@dataclass
class CogACTComputeNormActionConfig(ComputeNormActionConfig):
    delta: bool = field(default=False)
    expected_action_dim: int = field(default=7)

    def build_action_process_func(self) -> Pipeline:
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                _ValidateRawActionDim(self.expected_action_dim),
                ToList(),
            ]
        )


@dataclass
class CogACTDataConfig(_FullDataConfig):
    dataset_name: str = field(default="libero_pi0_all")
    aug_policy: str = field(default="")
    action_config: CogACTActionConfig = field(default_factory=CogACTActionConfig)


@dataclass
class CogACTInferenceConfig(_FullInferenceConfig):
    model_name_or_path: Optional[str] = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_cogact/" "cogact_lora_sft_libero_150k"
        )
    )
    base_model_name_or_path: Optional[str] = field(default=None)
    action_model_type: str = field(default="DiT-B")
    action_dim: int = field(default=7)
    chunk_size: int = field(default=16)

    def _load_model(self) -> None:
        if not is_lora_checkpoint(self.model_name_or_path):
            return super()._load_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading model from {}", self.model_name_or_path)
        logger.info("Using device: {}", self.device)
        from_pretrained_kwargs = {
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.device.type == "cuda":
            from_pretrained_kwargs["device_map"] = {"": "cuda:0"}

        model = load_cogact_lora_model_for_inference(
            self.model_name_or_path,
            base_model_name_or_path=self.base_model_name_or_path,
            action_model_type=self.action_model_type,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            merge_on_load=False,
            **from_pretrained_kwargs,
        )
        tokenizer_path = resolve_cogact_tokenizer_path(
            self.model_name_or_path,
            base_model_name_or_path=self.base_model_name_or_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        logger.info("CogACT LoRA model loaded successfully")


@dataclass
class CogACTExp(_FullExp):
    model_config: CogACTModelConfig = field(default_factory=CogACTModelConfig)
    optimizer_config: CogACTOptimizerConfig = field(
        default_factory=CogACTOptimizerConfig
    )
    trainer_config: CogACTTrainerConfig = field(default_factory=CogACTTrainerConfig)
    data_config: CogACTDataConfig = field(default_factory=CogACTDataConfig)
    inference_config: CogACTInferenceConfig = field(
        default_factory=CogACTInferenceConfig
    )

    def _validate_train_backend(self) -> None:
        if self.trainer_config.train_backend != "ddp":
            raise ValueError("CogACT LoRA SFT supports only the DDP training backend.")
        super()._validate_train_backend()

    def compute_norm_stats(self) -> None:
        norm_config = CogACTComputeNormActionConfig(
            expected_action_dim=self.data_config.action_config.expected_action_dim
        )
        norm_config.compute_norm_stats(self.data_config.dataset_name)

    def _auto_compute_norm_stats(self) -> None:
        action_config = self.data_config.action_config
        if (
            not self.data_config.auto_norm
            or action_config.statistic_mapping is not None
        ):
            return

        norm_config = CogACTComputeNormActionConfig(
            norm_method=self.data_config.auto_norm_method,
            expected_action_dim=action_config.expected_action_dim,
        )
        cache_key = f"{self.data_config.dataset_name}:cogact_raw_action_v1"
        save_name = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing CogACT raw-action norm stats on rank0")
            norm_config.compute_norm_stats(self.data_config.dataset_name)
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                logger.info(
                    "Waiting for CogACT raw-action norm stats {} on rank{}",
                    norm_file_path,
                    self.local_rank,
                )
        action_config.statistic_mapping = norm_file_path


def main(args: argparse.Namespace) -> None:
    exp = CogACTExp()
    if args.train_backend is not None:
        exp.trainer_config.train_backend = args.train_backend
    if args.model_name_or_path is not None:
        exp.inference_config.model_name_or_path = args.model_name_or_path
    if args.base_model_name_or_path is not None:
        exp.inference_config.base_model_name_or_path = args.base_model_name_or_path
    if args.port is not None:
        exp.inference_config.port = args.port
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()


if __name__ == "__main__":
    main(parse_args())
