"""Uniform random agent: the baseline every learning curve should beat."""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from rlx.core.agent import Agent


class RandomAgent(Agent):
    name: ClassVar[str] = "Random (baseline)"
    reference: ClassVar[str] = "baseline"

    def act(self, state: int, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def values(self) -> np.ndarray:
        return np.zeros(self.n_states)

    def policy(self) -> np.ndarray:
        return np.full((self.n_states, self.n_actions), 1.0 / self.n_actions)
