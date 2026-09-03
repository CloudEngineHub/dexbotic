import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dexbotic.exp.dm05_exp import DM05Exp as _DM05Exp
from dexbotic.exp.dm05_exp import DM05InferenceConfig as _DM05InferenceConfig
from dexbotic.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig
from dexbotic.exp.dm05_exp import DM05OptimizerConfig as _DM05OptimizerConfig
from dexbotic.exp.dm05_exp import DM05TrainerConfig as _DM05TrainerConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    parser.add_argument(
        "--train-backend",
        dest="train_backend",
        type=str,
        default=None,
        choices=["deepspeed", "fsdp", "fsdp2", "ddp"],
    )
    parser.add_argument(
        "--model_name_or_path",
        "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default=None,
    )
    args, _ = parser.parse_known_args()
    return args


@dataclass
class DM05ModelConfig(_DM05ModelConfig):
    model_name_or_path: str = field(default="./checkpoints/DM05")
    llm_attn_implementation: str = field(default="eager")
    vision_attn_implementation: str = field(default="sdpa")
    action_attn_implementation: str = field(default="sdpa")
    vlm_gradient_checkpointing: bool = field(default=True)
    ae_gradient_checkpointing: bool = field(default=True)


@dataclass
class DM05OptimizerConfig(_DM05OptimizerConfig):
    base_lr: float = field(default=5e-4)
    warmup_steps: int = field(default=500)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    train_backend: str = field(default="ddp")
    deepspeed: Optional[str] = field(default=None)
    wandb_project: str = field(default="dm05_sft_libero_lora")
    model_max_length: int = field(default=1024)
    save_only_model: bool = field(default=True)
    save_hf_sidecar: bool = field(default=True)
    per_device_train_batch_size: int = field(default=4)
    output_dir: str = field(
        default=(
            "./user_checkpoints/dexbotic/libero_dm05_lora/"
            f"libero-lora-{datetime.now().strftime('%m%d')}"
        )
    )


@dataclass
class DM05InferenceConfig(_DM05InferenceConfig):
    model_max_length: int = field(default=1024)
    model_name_or_path: Optional[str] = field(
        default="./user_checkpoints/dexbotic/libero_dm05_lora/libero-lora-MMDD"
    )


@dataclass
class DM05Exp(_DM05Exp):
    use_lora: bool = field(default=True)
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    optimizer_config: DM05OptimizerConfig = field(default_factory=DM05OptimizerConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)
    inference_config: DM05InferenceConfig = field(default_factory=DM05InferenceConfig)


if __name__ == "__main__":
    args = parse_args()
    exp = DM05Exp()
    if args.train_backend is not None:
        exp.trainer_config.train_backend = args.train_backend
    if args.model_name_or_path is not None:
        exp.inference_config.model_name_or_path = args.model_name_or_path
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
