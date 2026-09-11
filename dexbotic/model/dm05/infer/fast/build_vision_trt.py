"""Export the DM05 vision tower/projector and build its TensorRT engine."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn

from dexbotic.infer.trt import (
    EngineManifest,
    TensorManifest,
    build_fp16_engine_from_onnx,
    engine_shape,
    load_tensorrt,
    resolve_io_names,
)
from dexbotic.model.dm05.dm05_arch import DM05ForConditionalGeneration
from dexbotic.model.dm05.dm05_lora import load_dm05_model_for_inference


class DM05VisionFeatureModule(nn.Module):
    def __init__(self, vlm_model: nn.Module) -> None:
        super().__init__()
        self.vlm_model = vlm_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_dtype = next(self.vlm_model.vision_tower.parameters()).dtype
        projector = self.vlm_model.multi_modal_projector
        projector_dtype = projector.mm_input_projection_weight.dtype
        vision_output = self.vlm_model.vision_tower(
            pixel_values=pixel_values.to(dtype=vision_dtype), return_dict=True
        ).last_hidden_state
        return projector(vision_output.to(dtype=projector_dtype))


def _load_model(checkpoint: Path) -> DM05ForConditionalGeneration:
    model = load_dm05_model_for_inference(
        str(checkpoint),
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.model.vlm.set_attn_implementation({"vision_config": "sdpa"})
    return model.eval().to(device="cuda", dtype=torch.float16)


def export_vision_onnx(model, onnx_path: Path, num_images: int, opset: int):
    vlm_model = model.model.vlm.model
    image_size = int(vlm_model.config.vision_config.image_size)
    wrapper = DM05VisionFeatureModule(vlm_model).eval().to("cuda")
    dummy = torch.randn(
        num_images,
        3,
        image_size,
        image_size,
        dtype=torch.float16,
        device="cuda",
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        output = wrapper(dummy)
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_features"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=None,
            dynamo=False,
        )
    return tuple(dummy.shape), tuple(output.shape)


def _engine_num_images(path: Path) -> int | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    trt = load_tensorrt("Inspecting DM05 vision engine")
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        return None
    inputs, _ = resolve_io_names(engine, trt)
    if len(inputs) != 1:
        return None
    shape = engine_shape(engine, inputs[0])
    return int(shape[0]) if len(shape) == 4 and shape[0] > 0 else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-path", type=Path, required=True)
    parser.add_argument("--engine-path", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--keep-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Building the DM05 vision engine requires CUDA.")
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive.")
    if (
        not args.force_rebuild
        and _engine_num_images(args.engine_path) == args.num_images
    ):
        print(f"Matching vision engine already exists: {args.engine_path}")
        return
    model = _load_model(args.checkpoint)
    input_shape, output_shape = export_vision_onnx(
        model, args.onnx_path, args.num_images, args.opset
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    build_fp16_engine_from_onnx(
        onnx_path=args.onnx_path,
        engine_path=args.engine_path,
        workspace_gb=args.workspace_gb,
        context="DM05 vision",
    )
    EngineManifest(
        model_family="dm05",
        model_revision=str(args.checkpoint),
        component="vision_tower_projector",
        precision="fp16",
        inputs=(TensorManifest("pixel_values", input_shape, "float16"),),
        outputs=(TensorManifest("image_features", output_shape, "float16"),),
        metadata={"opset": args.opset, "num_images": args.num_images},
    ).write(args.engine_path.with_suffix(args.engine_path.suffix + ".json"))
    if args.onnx_path.exists() and not args.keep_onnx:
        args.onnx_path.unlink()
    print(f"Built DM05 vision engine: {args.engine_path}")


if __name__ == "__main__":
    main()
