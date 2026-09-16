"""Shared Go-style result type for fallible operations."""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Result(Generic[T]):
    """Result of a fallible operation: exactly one of `data`/`error` is set."""

    data: T | None
    error: str | None
