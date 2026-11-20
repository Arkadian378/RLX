"""k-armed bandits (Sutton & Barto, Chapter 2).

Bandits have a single state, so they get their own tiny interface instead of
``Env``/``Agent``. The testbed is implemented; the agents are for you.

TODO(you): implement ``EpsilonGreedy``, ``UCB`` and ``GradientBandit``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BanditTestbed:
    """The 10-armed testbed of S&B 2.3: q*(a) ~ N(0, 1), R ~ N(q*(a), 1)."""

    def __init__(self, k: int = 10, seed: int | None = None) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(0.0, 1.0, size=k)

    @property
    def optimal_action(self) -> int:
        return int(np.argmax(self.q_star))

    def pull(self, action: int) -> float:
        return float(self.rng.normal(self.q_star[action], 1.0))


class BanditAgent(ABC):
    def __init__(self, k: int, seed: int | None = None) -> None:
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros(k)  # action-value estimates
        self.N = np.zeros(k)  # pull counts

    @abstractmethod
    def act(self) -> int: ...

    @abstractmethod
    def update(self, action: int, reward: float) -> None: ...


class EpsilonGreedy(BanditAgent):
    """S&B 2.4: incremental sample average Q += (R - Q) / N."""

    def __init__(self, k: int, epsilon: float = 0.1, seed: int | None = None) -> None:
        super().__init__(k, seed)
        self.epsilon = epsilon

    def act(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError


class UCB(BanditAgent):
    """S&B 2.7: argmax_a Q(a) + c sqrt(ln t / N(a)); untried arms first."""

    def __init__(self, k: int, c: float = 2.0, seed: int | None = None) -> None:
        super().__init__(k, seed)
        self.c = c
        self.t = 0

    def act(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError


class GradientBandit(BanditAgent):
    """S&B 2.8: softmax preferences H(a), updated with a reward baseline."""

    def __init__(self, k: int, alpha: float = 0.1, seed: int | None = None) -> None:
        super().__init__(k, seed)
        self.alpha = alpha
        self.H = np.zeros(k)
        self.baseline = 0.0

    def act(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError
