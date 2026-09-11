"""Transport-neutral orchestration for single-request inference."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class InferenceResult:
    """Finalized output and metadata for one inference request."""

    output: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class InferenceRuntime:
    """Own backend lifecycle, serialization, timing, and request boundaries."""

    def __init__(
        self,
        *,
        prepare: Callable,
        finalize: Callable,
        backend: Any,
        serialize_calls: bool = True,
    ) -> None:
        self._prepare = prepare
        self._finalize = finalize
        self.backend = backend
        self._lock = threading.Lock() if serialize_calls else None

    def infer(
        self, observation: Mapping[str, Any], sampling_config=None
    ) -> InferenceResult:
        """Run one request and return its finalized output and backend metadata."""

        tensors, context = self._prepare(dict(observation), sampling_config)
        started_at = time.perf_counter()
        if self._lock is None:
            raw_output, backend_metadata = self.backend.execute(tensors)
        else:
            with self._lock:
                raw_output, backend_metadata = self.backend.execute(tensors)
        output = self._finalize(raw_output, dict(context))
        metadata = {
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            **dict(backend_metadata),
        }
        return InferenceResult(output=output, metadata=metadata)
