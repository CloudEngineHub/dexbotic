"""Model-agnostic inference runtime primitives.

Optimized model implementations live below ``dexbotic.model.<family>.infer``.
This package deliberately avoids importing optional CUDA, Triton, or TensorRT
dependencies during normal eager inference.
"""

from dexbotic.infer.history import (
    history_image_capabilities,
    history_images_from_observation,
)
from dexbotic.infer.runtime import InferenceResult, InferenceRuntime

__all__ = [
    "InferenceResult",
    "InferenceRuntime",
    "history_image_capabilities",
    "history_images_from_observation",
]
