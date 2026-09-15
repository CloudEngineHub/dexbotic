from __future__ import annotations

import argparse
from dataclasses import dataclass, field

from dexbotic.exp.dm05_exp import DM05Exp as _DM05Exp
from dexbotic.exp.dm05_exp import DM05ModelConfig as _DM05ModelConfig
from dexbotic.exp.dm05_exp import DM05TrainerConfig as _DM05TrainerConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--backend",
        "--inference-backend",
        dest="backend",
        choices=["default", "fast"],
        default=None,
        help="DM05 inference backend. Defaults to 'default'.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Inference service port. Defaults to 7891.",
    )
    parser.add_argument(
        "--vision-trt-engine-path",
        "--vision_trt_engine_path",
        dest="vision_trt_engine_path",
        default=None,
        help="TensorRT vision engine path used by the fast backend.",
    )
    parser.add_argument(
        "--build-vision-engine-if-missing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Build a matching TensorRT vision engine during fast startup.",
    )
    parser.add_argument(
        "--force-rebuild-vision-engine",
        action="store_true",
        default=None,
        help="Rebuild the TensorRT vision engine even when one already exists.",
    )
    parser.add_argument(
        "--prefix-seq-len-buckets",
        type=int,
        nargs="+",
        default=None,
        help="Fixed prefix-length buckets available to the fast backend.",
    )
    parser.add_argument(
        "--fast-overflow-policy",
        choices=["error", "fallback"],
        default=None,
        help="Behavior when a request exceeds every fast prefix bucket.",
    )
    parser.add_argument(
        "--fast-prefix-qkv-mode",
        choices=["packed", "separate"],
        default=None,
        help="Prefix QKV projection mode used by the fast backend.",
    )
    parser.add_argument(
        "--history-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Accept explicit history images during inference.",
    )
    parser.add_argument(
        "--max-history-images",
        type=int,
        default=None,
        help="Maximum number of explicit history images accepted per request.",
    )
    args, _ = parser.parse_known_args(argv)
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


def configure_inference(exp: DM05Exp, args: argparse.Namespace) -> None:
    config = exp.inference_config
    for name in (
        "model_name_or_path",
        "backend",
        "port",
        "vision_trt_engine_path",
        "build_vision_engine_if_missing",
        "force_rebuild_vision_engine",
        "prefix_seq_len_buckets",
        "fast_overflow_policy",
        "fast_prefix_qkv_mode",
        "history_enabled",
        "max_history_images",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(config, name, value)


if __name__ == "__main__":
    args = parse_args()
    exp = DM05Exp()
    if args.train_backend is not None:
        exp.trainer_config.train_backend = args.train_backend
    configure_inference(exp, args)
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
