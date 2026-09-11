"""Batch utilities for variable-length history-image tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HistoryImageBatch:
    """Packed history images and their sample-level ownership metadata.

    ``pixel_values`` preserves sample order and, within each sample, the input
    history order. ``offsets`` uses the usual packed-sequence convention and
    therefore has ``batch_size + 1`` entries.
    """

    pixel_values: torch.Tensor | None
    counts: torch.LongTensor
    offsets: torch.LongTensor
    valid_mask: torch.BoolTensor

    @property
    def batch_size(self) -> int:
        return int(self.counts.numel())

    @property
    def total_images(self) -> int:
        return int(self.offsets[-1].item())


def collate_history_image_tensors(
    histories: Sequence[torch.Tensor | None],
    *,
    max_images: int | None = None,
) -> HistoryImageBatch:
    """Pack per-sample ``[history, ...]`` tensors without reordering them.

    When ``max_images`` is provided, ``valid_mask`` is padded to that fixed
    width. Otherwise its width is the longest history in this batch.
    """

    if max_images is not None and max_images < 0:
        raise ValueError("max_images must be non-negative")

    tensors: list[torch.Tensor] = []
    counts_list: list[int] = []
    trailing_shape: tuple[int, ...] | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None

    for sample_index, tensor in enumerate(histories):
        if tensor is None:
            counts_list.append(0)
            continue
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"histories[{sample_index}] must be a torch.Tensor or None")
        if tensor.ndim < 1:
            raise ValueError(
                f"histories[{sample_index}] must have a leading history dimension"
            )

        count = int(tensor.shape[0])
        if max_images is not None and count > max_images:
            raise ValueError(
                f"At most {max_images} history images are supported per sample, "
                f"got {count} for sample {sample_index}"
            )

        if trailing_shape is None:
            trailing_shape = tuple(tensor.shape[1:])
            dtype = tensor.dtype
            device = tensor.device
        else:
            if tuple(tensor.shape[1:]) != trailing_shape:
                raise ValueError(
                    "History image tensors must have matching non-history "
                    f"dimensions: expected {trailing_shape}, got "
                    f"{tuple(tensor.shape[1:])} for sample {sample_index}"
                )
            if tensor.dtype != dtype:
                raise ValueError(
                    "History image tensors must have matching dtypes: "
                    f"expected {dtype}, got {tensor.dtype} for sample {sample_index}"
                )
            if tensor.device != device:
                raise ValueError(
                    "History image tensors must be on the same device: "
                    f"expected {device}, got {tensor.device} for sample {sample_index}"
                )

        tensors.append(tensor)
        counts_list.append(count)

    metadata_device = device if device is not None else torch.device("cpu")
    counts = torch.tensor(counts_list, dtype=torch.long, device=metadata_device)
    offsets = torch.zeros(
        len(counts_list) + 1,
        dtype=torch.long,
        device=metadata_device,
    )
    if counts_list:
        offsets[1:] = counts.cumsum(dim=0)

    mask_width = max_images if max_images is not None else max(counts_list, default=0)
    valid_mask = torch.arange(
        mask_width,
        dtype=torch.long,
        device=metadata_device,
    ).unsqueeze(0) < counts.unsqueeze(1)
    pixel_values = torch.cat(tensors, dim=0) if tensors else None
    return HistoryImageBatch(
        pixel_values=pixel_values,
        counts=counts,
        offsets=offsets,
        valid_mask=valid_mask,
    )
