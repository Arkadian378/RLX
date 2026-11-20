"""Environment interfaces.

Two levels of abstraction:

* ``Env`` is the minimal interaction contract (``reset`` / ``step``) that every
  environment in RLX implements, tabular or not.
* ``DiscreteEnv`` is a finite MDP that *also* exposes its full transition model
  ``P[s][a] = [(prob, next_state, reward, done), ...]``. Model-based methods
  (Dynamic Programming) read ``P`` directly; model-free methods must only use
  ``reset`` / ``step``.

``DiscreteEnv.step`` samples from ``P`` itself, so the model a planner sees and
the dynamics a learner experiences can never disagree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# One outcome of taking action a in state s: (probability, next_state, reward, done).
Outcome = tuple[float, int, float, bool]
# Full transition model: P[s][a] is the list of possible outcomes.
TransitionModel = list[list[list[Outcome]]]


@dataclass(frozen=True)
class StepResult:
    """What ``Env.step`` returns.

    ``terminated`` means the MDP reached a terminal state (bootstrap with 0).
    ``truncated`` means we stopped for an external reason, e.g. a step limit
    (the state is *not* terminal, so a learner should still bootstrap).
    """

    state: int
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class Env(ABC):
    """Minimal episodic environment contract."""

    n_actions: int

    @abstractmethod
    def reset(self, seed: int | None = None) -> int:
        """Start a new episode and return the initial state."""

    @abstractmethod
    def step(self, action: int) -> StepResult:
        """Apply ``action`` and return the resulting transition."""


class DiscreteEnv(Env):
    """A finite MDP with a known transition model.

    Subclasses only need to build ``P`` and choose the initial state
    distribution; sampling, seeding and step limits are handled here.
    """

    def __init__(
        self,
        P: TransitionModel,
        initial_state_dist: np.ndarray,
        max_steps: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.P = P
        self.n_states = len(P)
        self.n_actions = len(P[0])
        self.initial_state_dist = np.asarray(initial_state_dist, dtype=float)
        self.max_steps = max_steps
        self._validate()

        self.rng = np.random.default_rng(seed)
        self.state: int | None = None
        self.t = 0

    # ------------------------------------------------------------------ API
    def reset(self, seed: int | None = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = int(self.rng.choice(self.n_states, p=self.initial_state_dist))
        self.t = 0
        return self.state

    def step(self, action: int) -> StepResult:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Invalid action {action}; expected 0..{self.n_actions - 1}.")

        outcomes = self.P[self.state][action]
        probs = [o[0] for o in outcomes]
        idx = self.rng.choice(len(outcomes), p=probs) if len(outcomes) > 1 else 0
        _, next_state, reward, terminated = outcomes[idx]

        self.state = next_state
        self.t += 1
        truncated = not terminated and self.max_steps is not None and self.t >= self.max_steps
        return StepResult(next_state, float(reward), bool(terminated), truncated)

    # ------------------------------------------------------------- helpers
    def expected_reward(self) -> np.ndarray:
        """R[s, a] = E[r | s, a]. Handy for DP and for tests."""
        R = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                R[s, a] = sum(p * r for p, _, r, _ in self.P[s][a])
        return R

    def transition_matrix(self) -> np.ndarray:
        """T[s, a, s'] = Pr(s' | s, a)."""
        T = np.zeros((self.n_states, self.n_actions, self.n_states))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                for p, s2, _, _ in self.P[s][a]:
                    T[s, a, s2] += p
        return T

    def _validate(self) -> None:
        if not np.isclose(self.initial_state_dist.sum(), 1.0):
            raise ValueError("initial_state_dist must sum to 1.")
        for s, row in enumerate(self.P):
            if len(row) != self.n_actions:
                raise ValueError(f"State {s} has {len(row)} actions, expected {self.n_actions}.")
            for a, outcomes in enumerate(row):
                total = sum(o[0] for o in outcomes)
                if not np.isclose(total, 1.0):
                    raise ValueError(f"P[{s}][{a}] probabilities sum to {total}, not 1.")
