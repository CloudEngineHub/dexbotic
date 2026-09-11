"""Generic smallest-fitting profile routing."""

from __future__ import annotations

import bisect
from typing import Generic, Iterable, TypeVar

ProfileT = TypeVar("ProfileT")


class SmallestFittingProfileRegistry(Generic[ProfileT]):
    """Route a dynamic scalar to the smallest initialized fitting profile."""

    def __init__(self, sizes: Iterable[int]) -> None:
        self.sizes = tuple(int(size) for size in sizes)
        if not self.sizes or self.sizes != tuple(sorted(set(self.sizes))):
            raise ValueError("Profile sizes must be non-empty, increasing, and unique.")
        if self.sizes[0] <= 0:
            raise ValueError("Profile sizes must be positive.")
        self._profiles: dict[int, ProfileT] = {}

    def resolve_size(
        self,
        requested_size: int,
        *,
        overflow_policy: str,
        request_name: str = "Requested size",
        profile_name: str = "profile",
    ) -> int | None:
        """Fit a size, returning ``None`` only when fallback is requested."""

        if overflow_policy not in {"error", "fallback"}:
            raise ValueError("overflow_policy must be 'error' or 'fallback'.")
        index = bisect.bisect_left(self.sizes, int(requested_size))
        size = None if index == len(self.sizes) else self.sizes[index]
        if size is not None or overflow_policy == "fallback":
            return size
        raise ValueError(
            f"{request_name} {int(requested_size)} exceeds largest "
            f"{profile_name} {self.sizes[-1]}."
        )

    def select_or_create(
        self, requested_size: int, factory
    ) -> tuple[int | None, ProfileT | None]:
        """Return the smallest fitting profile, creating it only if needed."""

        size = self.resolve_size(requested_size, overflow_policy="fallback")
        if size is None:
            return None, None
        if size not in self._profiles:
            self._profiles[size] = factory(size)
        return size, self._profiles[size]

    def select(self, requested_size: int) -> tuple[int | None, ProfileT | None]:
        if not self._profiles:
            raise RuntimeError("Profiles must be initialized before selection.")
        size = self.resolve_size(requested_size, overflow_policy="fallback")
        if size is None:
            return None, None
        return size, self._profiles[size]
