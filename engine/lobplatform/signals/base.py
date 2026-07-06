"""Signal interface: finalized bars in, bounded score/confidence out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..data.bar_builder import Bar


@dataclass(frozen=True)
class SignalOutput:
    score: float        # [-1, 1]; sign = direction
    confidence: float   # [0, 1]

    def __post_init__(self) -> None:
        assert -1.0 <= self.score <= 1.0, self.score
        assert 0.0 <= self.confidence <= 1.0, self.confidence


class Signal(ABC):
    name: str = "base"

    @abstractmethod
    def compute(self, symbol: str, bars: list[Bar]) -> SignalOutput:
        """`bars` are finalized 5-min bars, oldest first, last = most recent."""
