"""Model-independent helpers for offline tensor calibration datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Collection, Iterable, Mapping

from dexbotic.infer.artifacts import canonical_json, sha256_file

if TYPE_CHECKING:
    import torch

__all__ = [
    "MaxAbsObserver",
    "load_tensor_record_files",
    "tensor_record_fingerprint",
]


@dataclass
class MaxAbsObserver:
    """Accumulate one finite maximum-absolute range over selected tensor rows."""

    max_abs: float = 0.0
    rows: int = 0
    elements: int = 0

    def update(
        self,
        value: torch.Tensor,
        *,
        row_mask: torch.Tensor | None = None,
    ) -> None:
        import torch

        if not isinstance(value, torch.Tensor):
            raise TypeError("Calibration values must be tensors.")
        detached = value.detach()
        if row_mask is not None:
            if not isinstance(row_mask, torch.Tensor) or row_mask.dtype != torch.bool:
                raise TypeError("Calibration row_mask must be a boolean tensor.")
            if detached.ndim < 2 or tuple(detached.shape[:-1]) != tuple(row_mask.shape):
                raise ValueError(
                    "Calibration row mask must match every tensor dimension "
                    "except the feature dimension."
                )
            selected = detached[row_mask]
        elif detached.ndim == 0:
            selected = detached.reshape(1, 1)
        elif detached.ndim == 1:
            selected = detached.reshape(1, -1)
        else:
            selected = detached.reshape(-1, detached.shape[-1])

        absolute = selected.float().abs()
        if absolute.numel() == 0 or not bool(torch.isfinite(absolute).all().item()):
            raise ValueError("Calibration values must be nonempty and finite.")
        self.max_abs = max(self.max_abs, float(absolute.max().item()))
        self.rows += int(selected.shape[0])
        self.elements += int(absolute.numel())

    def as_dict(self) -> dict:
        return {
            "max_abs": self.max_abs,
            "rows": self.rows,
            "elements": self.elements,
        }


def tensor_record_fingerprint(
    record: Mapping[str, torch.Tensor], keys: Iterable[str]
) -> str:
    """Hash selected tensor names, shapes, dtypes, and exact CPU bytes."""
    import torch

    keys = tuple(sorted(set(keys)))
    if not keys:
        raise ValueError("Fingerprint keys must not be empty.")
    digest = hashlib.sha256()
    for key in keys:
        try:
            value = record[key]
        except KeyError as exc:
            raise ValueError(f"Tensor record is missing required key {key!r}.") from exc
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Tensor record value {key!r} must be a tensor.")
        value = value.detach().to(device="cpu").contiguous()
        identity = [key, list(value.shape), str(value.dtype)]
        digest.update(canonical_json(identity))
        digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def load_tensor_record_files(
    paths: Iterable[str | Path],
    *,
    required_keys: Collection[str],
    optional_keys: Collection[str] = (),
    validate_record: Callable[[dict, Path, int], None] | None = None,
) -> tuple[list[dict[str, torch.Tensor]], list[dict], int]:
    """Load safe tensor-only records with provenance and distinct-input count."""
    import torch

    required = set(required_keys)
    allowed = required | set(optional_keys)
    if not required:
        raise ValueError("required_keys must not be empty")

    paths = tuple(paths)
    if not paths:
        raise ValueError("At least one tensor record file is required")
    samples: list[dict[str, torch.Tensor]] = []
    provenance: list[dict] = []
    fingerprints = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        payload = torch.load(path, map_location="cpu", weights_only=True)
        records = payload if isinstance(payload, list) else [payload]
        if not records:
            raise ValueError(f"Tensor record file contains no samples: {path}.")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or not required <= record.keys():
                raise ValueError(f"{path} sample {index} must contain {required}.")
            if record.keys() - allowed:
                raise ValueError(f"Unsupported keys in {path} sample {index}.")
            if not all(isinstance(value, torch.Tensor) for value in record.values()):
                raise ValueError("Tensor record inputs must contain tensors only.")
            if validate_record is not None:
                validate_record(record, path, index)
            fingerprints.add(tensor_record_fingerprint(record, required))
            samples.append(record)
        provenance.append(
            {"path": str(path), "sha256": sha256_file(path), "samples": len(records)}
        )
    return samples, provenance, len(fingerprints)
