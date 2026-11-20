"""On-policy first-visit Monte Carlo control (Sutton & Barto, Section 5.4).

TODO(you): implement ``observe`` and ``end_episode``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from rlx.core.agent import TabularAgent, Transition


class MonteCarloControl(TabularAgent):
    """First-visit MC control with an epsilon-soft (epsilon-greedy) policy.

    * ``observe``: append ``(state, action, reward)`` to ``self.episode``.
    * ``end_episode``: walk the episode *backwards* accumulating
      ``G <- gamma * G + R``; for the first visit of each (s, a) update
      ``Q(s, a)`` toward ``G``. Use either the sample average
      (``self.counts``) or a constant step size ``alpha`` (S&B eq. 2.4 vs 2.5).
      Then clear ``self.episode``.

    Note: a truncated episode gives a biased return. That's fine here, just be
    aware of it.
    """

    name: ClassVar[str] = "Monte Carlo Control"
    reference: ClassVar[str] = "S&B 5.4"
    implemented: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {"gamma": 0.99, "alpha": 0.1, "epsilon": 0.1}

    def __init__(self, n_states: int, n_actions: int, **kwargs) -> None:
        super().__init__(n_states, n_actions, **kwargs)
        self.episode: list[tuple[int, int, float]] = []
        self.counts = np.zeros_like(self.Q)

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("MonteCarloControl.observe: see the docstring.")

    def end_episode(self) -> None:
        raise NotImplementedError("MonteCarloControl.end_episode: see the docstring.")
