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
# Experiment wiring for GR00T N1.7 ("gr00tsonic"), trained through Dexbotic's
# DexDataset on the Unitree-G1 SONIC dataset . The model architecture is the upstream Isaac-GR00T
# Gr00tN1d7 (Qwen3-VL backbone + flow-matching DiT action head).
#
# Data path (DexDataset → Qwen3-VL):
#   action pipeline (explicit 78-d action → 40-step trajectory → norm → pad 132)
#   + Gr00tSonicImagePreprocess (per-view uint8) + Gr00tSonicTokenization (prompt)
#   collated by Gr00tSonicDataCollator into Qwen3-VL inputs + action-head inputs.

import os


import argparse
from dataclasses import dataclass, field
import hashlib
import json
import time

import megfile
import numpy as np
import torch
from loguru import logger

from dexbotic.data.dataset.dex_dataset import DexDataset
from dexbotic.data.dataset.gr00tsonic_data import (
    Gr00tSonicDataCollator,
    Gr00tSonicDexDataset,
    Gr00tSonicImagePreprocess,
    Gr00tSonicTokenization,
)
from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddTrajectory,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import (
    Pipeline,
    ToDict,
    ToList,
    ToNumpy,
    ToTensor,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import ActionDenorm
from dexbotic.exp.base_exp import (
    ActionConfig,
    BaseExp,
    ComputeNormActionConfig,
    DataConfig,
    FSDPProfile,
    InferenceConfig as BaseInferenceConfig,
    ModelConfig,
    OptimizerConfig,
    TokenizerConfig,
    TrainerConfig,
)
from dexbotic.exp.trainer import DexboticTrainer, safe_save_model_for_hf_trainer
from dexbotic.exp.utils import NumpyEncoder
from dexbotic.model.gr00tsonic.gr00tsonic_arch import (
    GR00TSonicConfig,
    GR00TSonicForCausalLM,
    load_pretrained_gr00t,
)
from dexbotic.policy.gr00tsonic_policy import Gr00tSonicPolicy

try:
    from transformers import Qwen3VLProcessor
except ImportError:  # pragma: no cover
    Qwen3VLProcessor = None


# Cosmos-Reason2-2B (Qwen3-VL) backbone, resolved by id from the shared HF hub
DEFAULT_COSMOS = "nvidia/Cosmos-Reason2-2B"

# SONIC-specific data dimensions.
# Image-frame version (fast DexDataset path); pre-extract with
# hardware/unitree_sonic/extract_frames.py.
DATASET_NAME = "sonic_beef_pie_xsh"
SONIC_VALID_ACTION_DIM = 78
SONIC_VALID_STATE_DIM = 46
SONIC_EMBODIMENT_ID = 11  # "unitree_g1_sonic" projector index (upstream mapping)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    args, unknown = parser.parse_known_args()
    return args


# ── action / norm-stats configs ──────────────────────────────────────────────


@dataclass
class Gr00tSonicActionConfig(ActionConfig):
    """Builds the SONIC action pipeline using the *explicit* 78-d action field.

    Unlike the default ActionConfig (which derives action by shifting the state),
    SONIC episodes carry an explicit absolute action, so we skip AddAction and
    build the trajectory directly from it.
    """

    trajectory_length: int = field(default=40)  # == model action_horizon
    delta: bool = field(default=False)  # SONIC actions are absolute
    max_action_dim: int = field(default=132)
    max_state_dim: int = field(default=132)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        return Pipeline(
            [
                ToDict(),
                ToNumpy(),
                # NOTE: no AddAction — use the dataset's explicit `action` field.
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    flatten=False,
                    padding_mode="last",
                    padding_action=True,
                ),
                ActionNorm(
                    statistic_mapping=statistic_mapping,
                    strict=False,
                    use_quantiles=True,
                ),
                PadAction(ndim=self.max_action_dim, axis=-1),
                PadState(ndim=self.max_state_dim, axis=-1),
                LoadMultiModal(return_masks=False),
                ToList(),
            ]
        )


@dataclass
class Gr00tSonicComputeNormActionConfig(ComputeNormActionConfig):
    """Computes action norm stats from the explicit action field (q01/q99)."""

    def build_action_process_func(self) -> Pipeline:
        # Keep the raw per-frame explicit action; norm stats are computed on it.
        return Pipeline([ToDict(), ToNumpy(), ToList()])

    def _merge_norm_stats(self, norm_files, per_task_norm=False):
        min_list, max_list = [], []
        for _name, (norm_file, _path) in norm_files.items():
            with open(norm_file, "r") as f:
                stats = json.load(f)["norm_stats"]["action"]
            min_list.append(stats["q01"])
            max_list.append(stats["q99"])
        min_arr = np.array(min_list).min(axis=0).tolist()
        max_arr = np.array(max_list).max(axis=0).tolist()
        # Emit BOTH a 'default' and an 'action' entry: ActionNorm/ActionDenorm key
        # off the data field name ('action'), so the 'action' entry is required.
        norm_stats = {
            "default": {"min": min_arr, "max": max_arr},
            "action": {"min": min_arr, "max": max_arr},
        }
        with open(os.path.join(self.norm_save_path, "norm_stats.json"), "w") as f:
            json.dump({"norm_stats": norm_stats}, f, indent=2)


# ── optimizer / trainer ──────────────────────────────────────────────────────


@dataclass
class Gr00tSonicOptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=1e-4)
    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
    weight_decay: float = field(default=1e-5)

    def _get_optimizer_grouped_parameters(self, model) -> list:
        return [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.weight_decay,
            }
        ]


@dataclass
class Gr00tSonicTrainerConfig(TrainerConfig):
    fsdp_profile: FSDPProfile = field(
        default_factory=lambda: FSDPProfile(
            enabled=True,
            cpu_ram_efficient_loading=True,
        )
    )
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=30000)
    per_device_train_batch_size: int = field(default=32)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)
    model_max_length: int = field(default=4096)
    dataloader_num_workers: int = field(default=8)
    logging_steps: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})


# ── model ────────────────────────────────────────────────────────────────────


@dataclass
class Gr00tSonicModelConfig(ModelConfig):
    cosmos_model_name: str = field(default=DEFAULT_COSMOS)
    from_scratch: bool = field(default=True)
    model_config_overrides: dict = field(default_factory=dict)
    # GR00T-N1.7 base model to initialize from — an HF repo id (resolved from the
    # local HF cache) or a local Gr00tN1d7 checkpoint dir. The gr00tsonic config
    # defaults already match it exactly (DiT 32 layers, vl_self_attention 4 layers,
    # select_layer 16), and it provides BOTH the Qwen3-VL backbone and the
    # pretrained action head, so we skip the separate Cosmos load.
    pretrained_gr00t_path: str = field(default="nvidia/GR00T-N1.7-3B")

    tune_llm: bool = field(default=False)
    tune_visual: bool = field(default=False)
    tune_projector: bool = field(default=True)
    tune_diffusion_model: bool = field(default=True)
    tune_vlln: bool = field(default=True)

    def build_model(self) -> GR00TSonicForCausalLM:
        if self.from_scratch:
            overrides = dict(self.model_config_overrides)
            overrides.setdefault("model_name", self.cosmos_model_name)
            use_gr00t_init = bool(self.pretrained_gr00t_path)
            # If initializing from a gr00t ckpt, it provides the backbone too, so
            # build an empty backbone from config; otherwise pull Cosmos as VLM init.
            cfg = GR00TSonicConfig(
                load_backbone_pretrained=not use_gr00t_init, **overrides
            )
            model = GR00TSonicForCausalLM(cfg)
            if use_gr00t_init:
                load_pretrained_gr00t(model, self.pretrained_gr00t_path)
        else:
            model = GR00TSonicForCausalLM.from_pretrained(self.model_name_or_path)

        model.model.set_trainable_parameters(
            tune_llm=self.tune_llm,
            tune_visual=self.tune_visual,
            tune_projector=self.tune_projector,
            tune_diffusion_model=self.tune_diffusion_model,
            tune_vlln=self.tune_vlln,
        )
        return model


@dataclass
class Gr00tSonicTokenizerConfig(TokenizerConfig):
    use_fast_tokenizer: bool = field(default=True)


# ── data ─────────────────────────────────────────────────────────────────────


@dataclass
class Gr00tSonicDataConfig(DataConfig):
    dataset_name: str = field(default=DATASET_NAME)
    num_images: int = field(default=1)
    data_keys: list[str] = field(default_factory=lambda: ["action", "state"])
    images_keys: list[str] = field(default=None)
    aug_policy: str | list[str] = field(default=None)
    image_aspect_ratio: str = field(default=None)
    auto_norm: bool = field(default=True)
    auto_norm_method: str = field(default="default")

    cosmos_model_name: str = field(default=DEFAULT_COSMOS)
    image_target_size: tuple = field(default=(256, 256))
    max_state_dim: int = field(default=132)
    max_action_dim: int = field(default=132)
    action_horizon: int = field(default=40)
    valid_action_dim: int = field(default=SONIC_VALID_ACTION_DIM)
    embodiment_id: int = field(default=SONIC_EMBODIMENT_ID)
    formalize_language: bool = field(default=True)

    action_config: Gr00tSonicActionConfig = field(
        default_factory=Gr00tSonicActionConfig
    )

    def build_data(self, processor):
        dataset = self._build_dataset()
        collator = Gr00tSonicDataCollator(
            processor=processor,
            max_state_dim=self.max_state_dim,
            max_action_dim=self.max_action_dim,
            action_horizon=self.action_horizon,
            state_history_length=1,
            valid_action_dim=self.valid_action_dim,
            embodiment_id=self.embodiment_id,
            formalize_language=self.formalize_language,
        )
        return dataset, collator

    def _build_dataset(self) -> DexDataset:
        from easydict import EasyDict

        data_args = EasyDict(
            {
                "dataset_name": self.dataset_name,
                "num_images": self.num_images,
                "data_keys": self.data_keys,
                "images_keys": self.images_keys,
                "aug_policy": self.aug_policy,
                "image_aspect_ratio": self.image_aspect_ratio,
            }
        )
        action_process_func = self.action_config.build_action_process_func()
        image_process_func = [
            Gr00tSonicImagePreprocess(self.image_target_size)
            for _ in range(self.num_images)
        ]
        dataset = Gr00tSonicDexDataset(
            data_args=data_args,
            tokenization_func=Gr00tSonicTokenization(),
            action_process_func=action_process_func,
            image_process_func=image_process_func,
            depth_process_func=(lambda *a, **k: None),
            # Step-level window == action_horizon: only this many frames are
            # processed per sample instead of the whole episode.
            window_size=self.action_horizon,
        )
        return dataset


# ── inference ────────────────────────────────────────────────────────────────


@dataclass
class Gr00tSonicInferenceConfig(BaseInferenceConfig):
    cosmos_model_name: str = field(default=DEFAULT_COSMOS)
    camera_order: list = field(default_factory=lambda: ["ego"])
    embodiment_id: int = field(default=SONIC_EMBODIMENT_ID)
    action_dim: int = field(default=SONIC_VALID_ACTION_DIM)
    formalize_language: bool = field(default=True)

    def read_normalization_stats(self, action_norm_file):
        # Return the FULL stats dict (keep the 'action' entry) rather than only
        # 'default', because ActionDenorm keys off the 'action' field name.
        logger.info(f"Reading normalization stats from {action_norm_file}")
        if action_norm_file is None or not megfile.smart_exists(action_norm_file):
            return {"default": {"min": -1, "max": 1}, "action": {"min": -1, "max": 1}}
        with megfile.smart_open(action_norm_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def _build_policy(self):
        return Gr00tSonicPolicy(
            model=self.model,
            tokenizer=self.tokenizer,
            norm_stats=self.norm_stats,
            input_pipeline=self.input_transform,
            output_pipeline=self.output_transform,
            device=self.device,
            camera_order=self.camera_order,
            processor=self.processor,
            embodiment_id=self.embodiment_id,
            formalize_language=self.formalize_language,
            action_dim=self.action_dim,
        )

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading gr00tsonic model from {self.model_name_or_path}")
        model = GR00TSonicForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)
        model.eval()
        self.model = model
        self.processor = Qwen3VLProcessor.from_pretrained(self.cosmos_model_name)
        self.tokenizer = self.processor.tokenizer
        self.model_config = model.config
        logger.info("Model loaded successfully")

        state_dim = self.model.model.config.max_state_dim
        # State is only padded (not normalized) — matches training.
        self.input_transform = Pipeline(
            [PadState(ndim=state_dim, axis=-1), ToTensor()]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=True
                ),
            ]
        )

    def _initialize_inference(self) -> None:
        if self.norm_stats is None:
            norm_stats_file = os.path.join(self.model_name_or_path, "norm_stats.json")
            self.norm_stats = self.read_normalization_stats(norm_stats_file)
        elif isinstance(self.norm_stats, str):
            self.norm_stats = self.read_normalization_stats(self.norm_stats)
        logger.info(f"Normalization stats: {self.norm_stats}")

        self._load_model()
        self.prev_text = None
        self.timestep = 0
        self.episode = 0
        self.policy = self._build_policy()


# ── experiment ───────────────────────────────────────────────────────────────


@dataclass
class Gr00tSonicExp(BaseExp):
    model_config: Gr00tSonicModelConfig = field(default_factory=Gr00tSonicModelConfig)
    optimizer_config: Gr00tSonicOptimizerConfig = field(
        default_factory=Gr00tSonicOptimizerConfig
    )
    trainer_config: Gr00tSonicTrainerConfig = field(
        default_factory=Gr00tSonicTrainerConfig
    )
    data_config: Gr00tSonicDataConfig = field(default_factory=Gr00tSonicDataConfig)
    tokenizer_config: Gr00tSonicTokenizerConfig = field(
        default_factory=Gr00tSonicTokenizerConfig
    )
    inference_config: Gr00tSonicInferenceConfig = field(
        default_factory=Gr00tSonicInferenceConfig
    )

    # ── norm stats ───────────────────────────────────────────────────────────

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        norm_config = Gr00tSonicComputeNormActionConfig(
            delta=self.data_config.action_config.delta,
            norm_method=self.data_config.auto_norm_method,
        )
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing gr00tsonic norm stats on rank0")
            norm_config.compute_norm_stats(self.data_config.dataset_name)
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                logger.info(f"Waiting for norm stats: {norm_file_path}")
        self.data_config.action_config.statistic_mapping = norm_file_path

    def compute_norm_stats(self) -> None:
        self._auto_compute_norm_stats()

    # ── training ──────────────────────────────────────────────────────────────

    def _initialize_train(self):
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        logger.info(f"Local rank: {self.local_rank}")
        if self.local_rank != 0:
            logger.remove()
            logger.add(lambda msg: None)

        self._validate_train_backend()

        # Step 0: norm stats (explicit-action based).
        self._auto_compute_norm_stats()

        # Step 1: Qwen3-VL processor (owns tokenization for the backbone).
        processor = Qwen3VLProcessor.from_pretrained(self.data_config.cosmos_model_name)
        processor.tokenizer.padding_side = "left"
        self.processor = processor
        self.tokenizer = processor.tokenizer

        # Step 2: model.
        self.model = self.model_config.build_model()
        self.model.config.use_cache = False

        # Step 3: data (custom DexDataset + Qwen3-VL collator).
        train_dataset, data_collator = self.data_config.build_data(processor)

        # Step 4: trainer.
        trainer = DexboticTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            exp_config=self,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        self.trainer = trainer
        self._log_fsdp_runtime_state()

        # Step 5: persist norm stats next to the checkpoint.
        if self.local_rank == 0 and hasattr(
            train_dataset.action_process_func, "statistic_mapping"
        ):
            os.makedirs(self.trainer_config.output_dir, exist_ok=True)
            with open(
                os.path.join(self.trainer_config.output_dir, "norm_stats.json"), "w"
            ) as f:
                json.dump(
                    train_dataset.action_process_func.statistic_mapping,
                    f,
                    indent=2,
                    cls=NumpyEncoder,
                )

        self._apply_fsdp_model_dtype()

    def train(self):
        self._initialize_train()
        try:
            resume_checkpoint = self._resolve_auto_resume_checkpoint()
            if resume_checkpoint is not None:
                logger.info("Resuming training from checkpoint {}", resume_checkpoint)
                self.trainer.train(resume_from_checkpoint=resume_checkpoint)
            else:
                self.trainer.train()

            self.trainer.save_state()
            self.model.config.use_cache = True
            safe_save_model_for_hf_trainer(
                trainer=self.trainer, output_dir=self.trainer_config.output_dir
            )
            logger.info(
                f"Training completed and model saved to {self.trainer_config.output_dir}"
            )
        finally:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                except Exception as exc:
                    logger.warning("Failed to destroy process group cleanly: {}", exc)

    def inference(self) -> None:
        self.inference_config.run()


if __name__ == "__main__":
    args = parse_args()
    exp = Gr00tSonicExp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.local_rank = 0
        exp.compute_norm_stats()
