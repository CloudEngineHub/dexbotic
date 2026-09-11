"""DM05 vision TensorRT adapter built on the common engine runner."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from loguru import logger

from dexbotic.infer.trt import TensorRTRunner, engine_shape
from dexbotic.model.dm05.dm05_utils import HISTORY_POOL_SIZE

MAX_HISTORY_IMAGES = 5


def pool_image_features_to_history(
    image_features: torch.Tensor,
    *,
    pool_size: int = HISTORY_POOL_SIZE,
) -> torch.Tensor:
    """Pool projected vision tokens from ``(N, T, H)`` to a spatial grid."""

    tokens = int(image_features.shape[1])
    spatial = int(tokens**0.5)
    if spatial * spatial != tokens:
        raise ValueError(
            "image_features token count must be a perfect square for 2D pooling, "
            f"got tokens={tokens}."
        )
    hidden = int(image_features.shape[-1])
    grid = image_features.view(-1, spatial, spatial, hidden).permute(0, 3, 1, 2)
    grid = F.adaptive_avg_pool2d(grid, output_size=(pool_size, pool_size))
    return grid.permute(0, 2, 3, 1).reshape(-1, pool_size * pool_size, hidden)


def pack_current_and_history_pixels(
    current_pixel_values: torch.Tensor,
    history_pixel_values: torch.Tensor | None = None,
    *,
    max_history_images: int = MAX_HISTORY_IMAGES,
) -> tuple[torch.Tensor, int]:
    """Pack current views and zero-padded history into one fixed TRT batch."""

    if current_pixel_values.ndim != 4:
        raise ValueError(
            "current_pixel_values must be (N, C, H, W), got "
            f"shape={tuple(current_pixel_values.shape)}."
        )
    num_current = int(current_pixel_values.shape[0])
    if num_current <= 0:
        raise ValueError("current_pixel_values must contain at least one image.")
    if max_history_images < 0:
        raise ValueError(f"max_history_images must be >= 0, got {max_history_images}.")

    num_history = 0
    if history_pixel_values is not None and int(history_pixel_values.shape[0]) > 0:
        if history_pixel_values.ndim != 4:
            raise ValueError(
                "history_pixel_values must be (N, C, H, W), got "
                f"shape={tuple(history_pixel_values.shape)}."
            )
        if tuple(history_pixel_values.shape[1:]) != tuple(
            current_pixel_values.shape[1:]
        ):
            raise ValueError(
                "history_pixel_values spatial/channel shape must match current "
                f"views: current={tuple(current_pixel_values.shape[1:])}, "
                f"history={tuple(history_pixel_values.shape[1:])}."
            )
        num_history = int(history_pixel_values.shape[0])
        if num_history > max_history_images:
            raise ValueError(
                f"At most {max_history_images} history images are supported, got "
                f"{num_history}."
            )

    packed = current_pixel_values.new_zeros(
        (num_current + max_history_images, *current_pixel_values.shape[1:])
    )
    packed[:num_current].copy_(current_pixel_values)
    if num_history > 0:
        packed[num_current : num_current + num_history].copy_(
            history_pixel_values.to(
                device=current_pixel_values.device,
                dtype=current_pixel_values.dtype,
            )
        )
    return packed, num_history


class DM05VisionTensorRTRunner:
    """Run the static vision-tower + multimodal-projector TensorRT engine."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        self.runner = TensorRTRunner(
            engine_path,
            device=device,
            context="DM05 vision TensorRT inference",
        )
        if len(self.runner.input_names) != 1 or len(self.runner.output_names) != 1:
            raise RuntimeError(
                "DM05 vision engine must have exactly one input and one output; "
                f"got {self.runner.input_names}, {self.runner.output_names}."
            )
        self.device = self.runner.device
        self.input_name = self.runner.input_names[0]
        self.output_name = self.runner.output_names[0]
        self.input_dtype = self.runner.input_dtypes[self.input_name]
        self.output_dtype = self.runner.output_dtypes[self.output_name]
        self.input_shape = engine_shape(self.runner.engine, self.input_name)
        self.output_shape = engine_shape(self.runner.engine, self.output_name)
        if len(self.input_shape) != 4 or any(dim <= 0 for dim in self.input_shape):
            raise RuntimeError(
                f"DM05 vision engine requires a static 4D input, got {self.input_shape}."
            )
        self.num_images = int(self.input_shape[0])

    @torch.inference_mode()
    def __call__(
        self,
        pixel_values: torch.Tensor,
        *,
        output_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_tensor = self.runner.prepare_input(self.input_name, pixel_values)
        input_shape = tuple(input_tensor.shape)
        if input_shape != self.input_shape:
            raise ValueError(
                f"DM05 vision input shape must be {self.input_shape}, got {input_shape}."
            )
        self.runner.set_input_shape(self.input_name, input_shape)
        output_shape = self.runner.output_shape(self.output_name)
        if any(dim <= 0 for dim in output_shape):
            raise RuntimeError(f"Unresolved DM05 vision output shape: {output_shape}.")
        if output_tensor is None:
            output_tensor = torch.empty(
                output_shape,
                device=self.device,
                dtype=self.output_dtype,
            )
        elif (
            tuple(output_tensor.shape) != tuple(output_shape)
            or output_tensor.device != self.device
            or output_tensor.dtype != self.output_dtype
            or not output_tensor.is_contiguous()
        ):
            raise ValueError(
                "DM05 vision output buffer does not match the TensorRT output: "
                f"expected shape={tuple(output_shape)}, dtype={self.output_dtype}, "
                f"device={self.device}, contiguous=True; got "
                f"shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, "
                f"device={output_tensor.device}, "
                f"contiguous={output_tensor.is_contiguous()}."
            )
        self.runner.execute(
            inputs={self.input_name: input_tensor},
            outputs={self.output_name: output_tensor},
        )
        return output_tensor


def ensure_dm05_vision_engine(
    *,
    checkpoint: str | Path,
    engine_path: str | Path,
    num_images: int,
    force_rebuild: bool = False,
) -> Path:
    engine_path = Path(engine_path).expanduser()
    command = [
        sys.executable,
        "-m",
        "dexbotic.model.dm05.infer.fast.build_vision_trt",
        "--checkpoint",
        str(Path(checkpoint).expanduser()),
        "--onnx-path",
        str(engine_path.with_suffix(".onnx")),
        "--engine-path",
        str(engine_path),
        "--num-images",
        str(int(num_images)),
    ]
    if force_rebuild:
        command.append("--force-rebuild")
    logger.info("Ensuring DM05 vision engine via: {}", shlex.join(command))
    subprocess.run(command, check=True)
    return engine_path
