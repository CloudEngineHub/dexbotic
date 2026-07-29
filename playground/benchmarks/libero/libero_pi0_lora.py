import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from transformers import AutoTokenizer

from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.exp.pi0_lora import Pi0LoraConfig as _Pi0LoraConfig
from dexbotic.exp.pi0_lora import (
    apply_lora_to_pi0_model,
    load_pi0_model_for_inference,
    resolve_pi0_tokenizer_path,
)
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM
from dexbotic.tokenization.process import Pi0Tokenization
from playground.benchmarks.libero.libero_pi0 import (
    Pi0DataConfig as _LiberoFullDataConfig,
    Pi0Exp as _LiberoFullExp,
    Pi0InferenceConfig as _LiberoFullInferenceConfig,
    Pi0ModelConfig as _LiberoFullModelConfig,
    Pi0OptimizerConfig as _LiberoFullOptimizerConfig,
    Pi0TrainerConfig as _LiberoFullTrainerConfig,
)


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
        help="PI0 LoRA SFT is validated and supported only with DDP.",
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
class Pi0LoraConfig(_Pi0LoraConfig):
    r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    target_modules: list[str] | str = field(default="all-linear")
    modules_to_save: list[str] = field(
        default_factory=lambda: [
            "state_proj",
            "action_in_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            "action_out_proj",
        ]
    )
    dump_trainable_path: Optional[str] = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_pi0/trainable_summaries/"
            "pi0_lora_sft_libero_50k.json"
        )
    )


@dataclass
class Pi0ModelConfig(_LiberoFullModelConfig):
    model_name_or_path: str = field(default="checkpoints/Dexbotic-PI0")
    lora_config: _Pi0LoraConfig = field(default_factory=Pi0LoraConfig)

    def build_model(self) -> Pi0ForCausalLM:
        model = Pi0ForCausalLM.from_pretrained(self.model_name_or_path)
        return apply_lora_to_pi0_model(
            model,
            self.lora_config,
            base_model_name_or_path=self.model_name_or_path,
        )


@dataclass
class Pi0TrainerConfig(_LiberoFullTrainerConfig):
    train_backend: str = field(default="ddp")
    deepspeed: Optional[str] = field(default=None)
    output_dir: str = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_pi0/"
            "pi0_lora_sft_libero_50k"
        )
    )
    wandb_project: str = field(default="pi0-libero-sft")
    num_train_steps: int = field(default=50000)
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=50)
    save_only_model: bool = field(default=True)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    model_max_length: int = field(default=48)
    use_raw_backward: bool = field(default=False)
    use_raw_warmup: bool = field(default=False)


@dataclass
class Pi0OptimizerConfig(_LiberoFullOptimizerConfig):
    base_lr: float = field(default=5e-4)
    warmup_steps: int = field(default=500)


@dataclass
class Pi0DataConfig(_LiberoFullDataConfig):
    pass


@dataclass
class Pi0InferenceConfig(_LiberoFullInferenceConfig):
    model_name_or_path: Optional[str] = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_pi0/"
            "pi0_lora_sft_libero_50k"
        )
    )
    base_model_name_or_path: Optional[str] = field(default=None)

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        from_pretrained_kwargs = {
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if self.device.type == "cuda":
            from_pretrained_kwargs["device_map"] = {"": "cuda:0"}
        model = load_pi0_model_for_inference(
            self.model_name_or_path,
            base_model_name_or_path=self.base_model_name_or_path,
            **from_pretrained_kwargs,
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            resolve_pi0_tokenizer_path(self.model_name_or_path),
            use_fast=False,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = Pi0Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(statistic_mapping=self.norm_stats, strict=False),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False),
                AbsoluteAction(),
            ]
        )


@dataclass
class Pi0Exp(_LiberoFullExp):
    model_config: Pi0ModelConfig = field(default_factory=Pi0ModelConfig)
    optimizer_config: Pi0OptimizerConfig = field(default_factory=Pi0OptimizerConfig)
    trainer_config: Pi0TrainerConfig = field(default_factory=Pi0TrainerConfig)
    data_config: Pi0DataConfig = field(default_factory=Pi0DataConfig)
    inference_config: Pi0InferenceConfig = field(default_factory=Pi0InferenceConfig)

    def _validate_train_backend(self) -> None:
        if self.trainer_config.train_backend != "ddp":
            raise ValueError("PI0 LoRA SFT supports only the DDP training backend.")
        super()._validate_train_backend()


def main(args: argparse.Namespace) -> None:
    exp = Pi0Exp()
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
