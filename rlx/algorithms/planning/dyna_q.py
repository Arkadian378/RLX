"""Tabular Dyna-Q (Sutton & Barto, Section 8.2).

TODO(you): implement ``observe``.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rlx.core.agent import TabularAgent, Transition


class DynaQ(TabularAgent):
    """Q-learning + a learned deterministic model + ``planning_steps`` replays.

    In ``observe``:

    (d) direct RL: Q-learning update on the real transition;
    (e) model learning: ``self.model[(s, a)] = (r, s', terminated)``;
    (f) planning: repeat ``planning_steps`` times, sample a previously seen
        (s, a) from ``self.model`` (``self.rng``!) and do a Q-learning update
        on the simulated transition.

    Dyna-Q builds its *own* model from experience; it must not read ``env.P``.
    """

    name: ClassVar[str] = "Dyna-Q"
    reference: ClassVar[str] = "S&B 8.2"
    implemented: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {
        "gamma": 0.95,
        "alpha": 0.1,
        "epsilon": 0.1,
        "planning_steps": 10,
    }

    def __init__(self, n_states: int, n_actions: int, planning_steps: int = 10, **kwargs) -> None:
        super().__init__(n_states, n_actions, **kwargs)
        self.planning_steps = int(planning_steps)
        self.model: dict[tuple[int, int], tuple[float, int, bool]] = {}

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("DynaQ.observe")
