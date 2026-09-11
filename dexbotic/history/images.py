"""Model-agnostic history-image contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class HistoryImageSpec:
    """Contract for an ordered history-image sequence.

    The contract is shared by data pipelines and inference adapters. Sampling,
    augmentation, padding, and model-specific encoding remain with their
    respective consumers.
    """

    enabled: bool = False
    max_images: int = 0
    ordering: Literal["oldest_to_newest"] = "oldest_to_newest"

    def __post_init__(self) -> None:
        if self.max_images < 0:
            raise ValueError("max_images must be non-negative")
        if self.ordering != "oldest_to_newest":
            raise ValueError("Only oldest_to_newest history ordering is supported")


def validate_history_images(
    value: Any,
    spec: HistoryImageSpec,
    *,
    value_name: str = "history_images",
) -> list[Any]:
    """Validate a history sequence and preserve its oldest-to-newest order."""

    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{value_name} must be a list")

    images = list(value)
    if images and not spec.enabled:
        raise ValueError("history_images were provided but history images are disabled")
    if len(images) > spec.max_images:
        raise ValueError(
            f"At most {spec.max_images} history images are supported, got {len(images)}"
        )
    return images
