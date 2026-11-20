"""n-step SARSA (Sutton & Barto, Section 7.2).

TODO(you): implement ``observe`` and ``end_episode``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, ClassVar

from rlx.core.agent import TabularAgent, Transition


class NStepSarsa(TabularAgent):
    """Keep the last ``n`` transitions in ``self.buffer``.

    * In ``observe``: append the transition; once the buffer holds ``n``
      items, compute ``G = sum_{i<n} gamma^i r_i + gamma^n Q(s_n, a_n)`` and
      update the *oldest* (s, a), then pop it. As in SARSA, you need the next
      action, so choose and store it here.
    * When the episode ends (terminated), flush the remaining items without
      bootstrapping. ``end_episode`` is a good place for that, then clear.
    """

    name: ClassVar[str] = "n-step SARSA"
    reference: ClassVar[str] = "S&B 7.2"
    implemented: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {
        "gamma": 0.99,
        "alpha": 0.1,
        "epsilon": 0.1,
        "n": 4,
    }

    def __init__(self, n_states: int, n_actions: int, n: int = 4, **kwargs) -> None:
        super().__init__(n_states, n_actions, **kwargs)
        self.n = int(n)
        self.buffer: deque[Transition] = deque()
        self.next_action: int | None = None

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("NStepSarsa.observe")

    def end_episode(self) -> None:
        raise NotImplementedError("NStepSarsa.end_episode")
