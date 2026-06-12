from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from playground.benchmarks.libero.libero_dm0 import (
    DM0Exp as _DM0Exp,
    DM0InferenceConfig as _DM0InferenceConfig,
    parse_args,
)
from dexbotic.data.dataset.transform.action import ActionNorm, PadState
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy, ToTensor
from dexbotic.data.dataset.transform.output import ActionDenorm, AbsoluteAction
from dexbotic.exp.dm0_exp import (
    DM0InferenceConfig as _BaseDM0InferenceConfig,
)


@dataclass
class DM0RealtimeInferenceConfig(_DM0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default="./checkpoints/libero/libero_dm0"
    )
    num_images: int = field(default=3)
    camera_order: list = field(
        default_factory=lambda: ["agentview", "wrist", None]
    )
    use_realtime_backend: bool = field(default=True)
    realtime_weight_path: Optional[str] = field(default="")
    realtime_max_lang_len: int = field(default=100)
    realtime_diffusion_steps: int = field(default=10)
    realtime_auto_convert: bool = field(default=True)

    def _load_model(self) -> None:
        _BaseDM0InferenceConfig._load_model(self)
        model_action_dim = getattr(self.model_config, "action_dim", 32)
        self.input_transform = Pipeline(
            [
                PadState(ndim=model_action_dim, axis=-1),
                ActionNorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=False
                ),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(
                    statistic_mapping=self.norm_stats, strict=False, use_quantiles=False
                ),
                AbsoluteAction(),
            ]
        )


@dataclass
class DM0RealtimeExp(_DM0Exp):
    inference_config: DM0RealtimeInferenceConfig = field(
        default_factory=DM0RealtimeInferenceConfig
    )


if __name__ == "__main__":
    args = parse_args()
    exp = DM0RealtimeExp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
