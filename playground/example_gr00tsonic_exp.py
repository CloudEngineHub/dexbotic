"""Example GR00T N1.7 ("gr00tsonic") experiment config.

gr00tsonic = the upstream Isaac-GR00T Gr00tN1d7 model (Qwen3-VL / Cosmos-Reason2-2B
backbone + flow-matching DiT action head) migrated into Dexbotic, fine-tuned on the
Unitree-G1 SONIC data through Dexbotic's DexDataset.

Train (8 GPUs)::
    torchrun --nproc_per_node=8 playground/example_gr00tsonic_exp.py --task train

Compute action norm stats only::
    python playground/example_gr00tsonic_exp.py --task compute_norm_stats

Inference server::
    python hardware/unitree_sonic/bridge.py

Notes
-----
* The model config defaults already match the released GR00T-N1.7 exactly (DiT 32
  layers, vl_self_attention 4 layers, select_layer 16). Training initializes from
  the GR00T-N1.7 base via ``pretrained_gr00t_path`` (HF id or local dir).
* Data: use the pre-extracted image-frame dataset (fast DexDataset path). Convert
  mp4 episodes with ``hardware/unitree_sonic/extract_frames.py`` first.
"""

from dataclasses import dataclass, field
from datetime import datetime

from dexbotic.exp.gr00tsonic_exp import (
    Gr00tSonicDataConfig,
    Gr00tSonicExp,
    Gr00tSonicInferenceConfig,
    Gr00tSonicModelConfig,
    Gr00tSonicOptimizerConfig,
    Gr00tSonicTrainerConfig,
    parse_args,
)


@dataclass
class OptimizerConfig(Gr00tSonicOptimizerConfig):
    base_lr: float = field(default=1e-4)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=1e-5)


@dataclass
class TrainerConfig(Gr00tSonicTrainerConfig):
    """Only the knobs you most likely want to tune.

    NOTE on training length: effective batch = per_device_train_batch_size *
    num_gpus * gradient_accumulation_steps. With 32 * 8 * 1 = 256 and a ~90k-frame
    dataset, 100k steps is ~286 epochs (overkill for fine-tuning). 10k-15k steps
    (~30-40 epochs) is usually plenty and cuts wall-time proportionally.
    """

    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=30000)
    per_device_train_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=1)
    deepspeed: str = field(default='./script/deepspeed/zero2.json')
    # Backbone is frozen; GC only affects the trainable action head. Turn off if
    # memory allows for ~20% faster steps.
    gradient_checkpointing: bool = field(default=False)
    # 180 cores / 8 GPUs → plenty of headroom; keep the GPUs fed.
    dataloader_num_workers: int = field(default=16)
    output_dir: str = field(
        default=f"./checkpoints"
    )
    wandb_project: str = field(default="dexbotic_gr00tsonic")


@dataclass
class DataConfig(Gr00tSonicDataConfig):
    # Image-frame dataset (fast path). Use "sonic_dexbotic_pingzi" for raw mp4 video
    # (slow) or your own registered dataset name.
    dataset_name: str = field(default="sonic_beef_pie_xsh")
    # SONIC is a single ego view. For multi-view data (images_1, images_2, ...),
    # set this to the number of camera views — the collator + backbone handle the rest.
    num_images: int = field(default=2)


@dataclass
class ModelConfig(Gr00tSonicModelConfig):
    # Initialize from the GR00T-N1.7 base (HF id resolved from the shared cache, or
    # a local Gr00tN1d7 checkpoint dir). Set to "" to train the action head from
    # scratch on top of the raw Cosmos backbone instead.
    pretrained_gr00t_path: str = field(default="nvidia/GR00T-N1.7-3B")
    from_scratch: bool = field(default=True)
    # Module tuning: GR00T fine-tuning trains the action head, freezes the backbone.
    tune_llm: bool = field(default=False)
    tune_visual: bool = field(default=False)
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)
    tune_vlln: bool = field(default=True)


@dataclass
class InferenceConfig(Gr00tSonicInferenceConfig):
    # Checkpoint to serve; if empty, defaults to the trainer output_dir.
    model_name_or_path: str = field(default="")
    port: int = field(default=7891)
    # SONIC ego view; action_dim = 64 motion_token + 7+7 hand joints = 78.
    camera_order: list = field(default_factory=lambda: ["ego"])
    action_dim: int = field(default=78)


@dataclass
class Gr00tSonicExampleExp(Gr00tSonicExp):
    model_config: ModelConfig = field(default_factory=ModelConfig)
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    inference_config: InferenceConfig = field(default_factory=InferenceConfig)


if __name__ == "__main__":
    args = parse_args()
    exp = Gr00tSonicExampleExp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.local_rank = 0
        exp.compute_norm_stats()
