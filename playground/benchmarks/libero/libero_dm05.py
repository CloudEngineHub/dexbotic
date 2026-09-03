from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from dexbotic.exp.dm05_exp import DM05Exp as _DM05Exp
from dexbotic.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig
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
    llm_attn_implementation: str = field(default="flex_attention")
    vlm_gradient_checkpointing: bool = field(default=False)
    ae_gradient_checkpointing: bool = field(default=False)


@dataclass
class DM05TrainerConfig(_DM05TrainerConfig):
    train_backend: str = field(default="fsdp2")
    wandb_project: str = field(default="dm05_sft_libero")
    num_train_steps: int = field(default=50000)
    save_steps: int = field(default=10000)
    per_device_train_batch_size: int = field(default=8)
    output_dir: str = field(
        default="./user_checkpoints/dexbotic/libero_dm05/libero-sft"
    )


@dataclass
class DM05Exp(_DM05Exp):
    model_config: DM05ModelConfig = field(default_factory=DM05ModelConfig)
    trainer_config: DM05TrainerConfig = field(default_factory=DM05TrainerConfig)


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
