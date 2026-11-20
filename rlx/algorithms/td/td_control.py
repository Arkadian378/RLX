"""One-step TD control: SARSA, Q-learning, Expected SARSA (S&B 6.4-6.6).

TODO(you): implement each ``observe``. They differ only in the *target*:

    SARSA           r + gamma * Q(s', a')            a' = action actually taken next
    Q-learning      r + gamma * max_a Q(s', a)
    Expected SARSA  r + gamma * sum_a pi(a|s') Q(s', a)

and all share the update ``Q(s,a) += alpha * (target - Q(s,a))``.
If ``transition.terminated`` the bootstrap term is 0. If it's only
``truncated``, keep bootstrapping: the state isn't actually terminal.
"""

from __future__ import annotations

from typing import ClassVar

from rlx.core.agent import TabularAgent, Transition


class QLearning(TabularAgent):
    name: ClassVar[str] = "Q-learning"
    reference: ClassVar[str] = "S&B 6.5"
    implemented: ClassVar[bool] = False

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("QLearning.observe")


class Sarsa(TabularAgent):
    """SARSA is on-policy: the target needs A_{t+1}, the action the agent will take next.

    The loop calls ``act(s')`` *after* ``observe``, so the cleanest approach is:
    in ``observe``, pick ``a' = self.act(s')`` yourself, store it in
    ``self.next_action`` and return it from the following ``act`` call.
    """

    name: ClassVar[str] = "SARSA"
    reference: ClassVar[str] = "S&B 6.4"
    implemented: ClassVar[bool] = False

    def __init__(self, n_states: int, n_actions: int, **kwargs) -> None:
        super().__init__(n_states, n_actions, **kwargs)
        self.next_action: int | None = None

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("Sarsa.observe")


class ExpectedSarsa(TabularAgent):
    name: ClassVar[str] = "Expected SARSA"
    reference: ClassVar[str] = "S&B 6.6"
    implemented: ClassVar[bool] = False

    def observe(self, transition: Transition) -> None:
        raise NotImplementedError("ExpectedSarsa.observe (hint: self.policy() gives pi)")
