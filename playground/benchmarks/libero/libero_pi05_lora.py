import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger
from transformers import AutoTokenizer

from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import ActionDenorm
from dexbotic.exp.pi05_lora import Pi05LoraConfig as _Pi05LoraConfig
from dexbotic.exp.pi05_lora import (
    apply_lora_to_pi05_model,
    load_pi05_base_model,
    load_pi05_lora_model_for_inference,
    resolve_pi05_tokenizer_path,
)
from dexbotic.model.pi05.pi05_arch import Pi05ForCausalLM
from dexbotic.tokenization.process import Pi0Tokenization
from playground.benchmarks.libero.libero_pi05 import (
    Pi05DataConfig as _LiberoFullDataConfig,
    Pi05Exp as _LiberoFullExp,
    Pi05InferenceConfig as _LiberoFullInferenceConfig,
    Pi05ModelConfig as _LiberoFullModelConfig,
    Pi05OptimizerConfig as _LiberoFullOptimizerConfig,
    Pi05TrainerConfig as _LiberoFullTrainerConfig,
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
        help="PI05 LoRA SFT is validated and supported only with DDP.",
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
class Pi05LoraConfig(_Pi05LoraConfig):
    r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    target_modules: list[str] | str = field(
        default=(
            r".*(model\.)?(llm|action_expert).*"
            r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
        )
    )
    modules_to_save: list[str] = field(
        default_factory=lambda: [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    )
    extra_trainable_regex: Optional[str] = field(
        default=(
            r"action_expert.*\.(input_layernorm|post_attention_layernorm|norm)"
            r"\.dense\.(base_layer\.)?(weight|bias)$"
        )
    )
    dump_trainable_path: Optional[str] = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_pi05/trainable_summaries/"
            "pi05_lora_sft_libero_50k.json"
        )
    )


@dataclass
class Pi05ModelConfig(_LiberoFullModelConfig):
    model_name_or_path: str = field(default="checkpoints/Dexbotic-PI05")
    lora_config: _Pi05LoraConfig = field(default_factory=Pi05LoraConfig)

    def build_model(self) -> Pi05ForCausalLM:
        model = load_pi05_base_model(self.model_name_or_path, chunk_size=10)
        return apply_lora_to_pi05_model(
            model,
            self.lora_config,
            base_model_name_or_path=self.model_name_or_path,
        )


@dataclass
class Pi05TrainerConfig(_LiberoFullTrainerConfig):
    train_backend: str = field(default="ddp")
    deepspeed: Optional[str] = field(default=None)
    output_dir: str = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_pi05/"
            "pi05_lora_sft_libero_50k"
        )
    )
    wandb_project: str = field(default="pi05-libero-lora-sft")
    num_train_steps: int = field(default=50000)
    save_steps: int = field(default=2000)
    save_total_limit: int = field(default=50)
    save_only_model: bool = field(default=True)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    dataloader_num_workers: int = field(default=4)
    model_max_length: int = field(default=200)
    use_raw_backward: bool = field(default=False)
    use_raw_warmup: bool = field(default=False)


@dataclass
class Pi05OptimizerConfig(_LiberoFullOptimizerConfig):
    base_lr: float = field(default=5e-4)
    warmup_steps: int = field(default=500)


@dataclass
class Pi05DataConfig(_LiberoFullDataConfig):
    pass


@dataclass
class Pi05InferenceConfig(_LiberoFullInferenceConfig):
    model_name_or_path: Optional[str] = field(
        default=(
            "user_checkpoints/dexbotic/libero_all_pi05/"
            "pi05_lora_sft_libero_50k"
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
        model = load_pi05_lora_model_for_inference(
            self.model_name_or_path,
            base_model_name_or_path=self.base_model_name_or_path,
            merge_on_load=True,
            chunk_size=10,
            **from_pretrained_kwargs,
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            resolve_pi05_tokenizer_path(self.model_name_or_path),
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
            ]
        )


@dataclass
class Pi05Exp(_LiberoFullExp):
    model_config: Pi05ModelConfig = field(default_factory=Pi05ModelConfig)
    optimizer_config: Pi05OptimizerConfig = field(default_factory=Pi05OptimizerConfig)
    trainer_config: Pi05TrainerConfig = field(default_factory=Pi05TrainerConfig)
    data_config: Pi05DataConfig = field(default_factory=Pi05DataConfig)
    inference_config: Pi05InferenceConfig = field(default_factory=Pi05InferenceConfig)

    def _validate_train_backend(self) -> None:
        if self.trainer_config.train_backend != "ddp":
            raise ValueError("PI05 LoRA SFT supports only the DDP training backend.")
        super()._validate_train_backend()


def main(args: argparse.Namespace) -> None:
    exp = Pi05Exp()
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
