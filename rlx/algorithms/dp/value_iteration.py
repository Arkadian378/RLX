"""Value Iteration (Sutton & Barto, Section 4.4).

Reference implementation: this is the one algorithm shipped complete, so the
rest of the stack (training loop, dashboard, tests) works end to end.

Each sweep applies the Bellman *optimality* backup to every state:

    V(s) <- max_a sum_{s', r} p(s', r | s, a) [ r + gamma * V(s') ]

and the greedy policy w.r.t. the current V is extracted after every sweep so
the dashboard can show the policy arrows changing while values propagate.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from rlx.core.agent import PlanningAgent
from rlx.core.env import DiscreteEnv


class ValueIteration(PlanningAgent):
    name: ClassVar[str] = "Value Iteration"
    reference: ClassVar[str] = "S&B 4.4"
    default_params: ClassVar[dict[str, Any]] = {"gamma": 0.99}

    def __init__(self, env: DiscreteEnv, gamma: float = 0.99, seed: int | None = None) -> None:
        super().__init__(env, gamma=gamma, seed=seed)

    def plan_step(self) -> float:
        # Synchronous ("two-array") sweep: compute every Q from the *old* V.
        # S&B's pseudocode updates V in place (Gauss-Seidel), which also
        # converges and is often faster; synchronous is easier to reason about
        # and to visualise sweep by sweep.
        Q = self.q_from_v(self.V)
        new_V = Q.max(axis=1)
        delta = float(np.max(np.abs(new_V - self.V)))
        self.V = new_V
        self.pi = self._greedy_policy(Q)
        return delta

    @staticmethod
    def _greedy_policy(Q: np.ndarray) -> np.ndarray:
        """Deterministic greedy policy, splitting probability evenly among ties."""
        best = np.isclose(Q, Q.max(axis=1, keepdims=True))
        return best / best.sum(axis=1, keepdims=True)
