"""Inference adapters for the shared history-image contract."""

from __future__ import annotations

from typing import Any, Mapping

from dexbotic.history import HistoryImageSpec, validate_history_images


def history_image_capabilities(spec: HistoryImageSpec) -> dict[str, Any]:
    """Serialize a shared history spec as an inference capability."""

    return {
        "supported": spec.enabled,
        "max_images": spec.max_images,
        "ordering": spec.ordering,
        "management": "explicit",
    }


def history_images_from_observation(
    observation: Mapping[str, Any],
    spec: HistoryImageSpec,
) -> list[Any]:
    """Validate and return ``observation.history_images`` without reordering it."""

    return validate_history_images(
        observation.get("history_images"),
        spec,
        value_name="observation.history_images",
    )
