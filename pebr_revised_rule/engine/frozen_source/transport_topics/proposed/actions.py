"""Canonical complete borrowing actions."""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BorrowingAction:
    donors: tuple[int, ...] = ()
    beta: float = 0.0

    def __post_init__(self):
        donors = tuple(int(j) for j in self.donors)
        if any(j < 0 for j in donors) or len(set(donors)) != len(donors):
            raise ValueError('Donor indices must be distinct and nonnegative')
        if not math.isfinite(self.beta) or not 0 <= self.beta < 1:
            raise ValueError('Borrowing weight must be finite and in [0,1)')
        if bool(donors) != (self.beta > 0):
            raise ValueError('Only the empty action has zero borrowing weight')
        object.__setattr__(self, 'donors', tuple(sorted(donors)))
        object.__setattr__(self, 'beta', float(self.beta))

    @property
    def tie_key(self):
        # Retained document indices follow the original source-row order.
        return self.beta, len(self.donors), self.donors


NO_BORROWING = BorrowingAction()
