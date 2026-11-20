"""Policy Evaluation and Policy Iteration (Sutton & Barto, Sections 4.1 and 4.3).

TODO(you): implement both ``plan_step`` methods. Run ``pytest`` afterwards: the
tests in ``tests/test_algorithms_learn.py`` are skipped while the stub raises
``NotImplementedError`` and start checking your code once it doesn't.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rlx.core.agent import PlanningAgent


class PolicyEvaluation(PlanningAgent):
    """Iterative policy evaluation of ``self.pi`` (uniform random by default).

    One ``plan_step`` = one sweep of the Bellman *expectation* backup:

        V(s) <- sum_a pi(a|s) sum_{s', r} p(s', r | s, a) [ r + gamma V(s') ]

    Return max_s |V_new(s) - V_old(s)|. ``self.pi`` is never changed.
    Hint: ``self.q_from_v(self.V)`` gives you the inner sum for every (s, a).
    """

    name: ClassVar[str] = "Policy Evaluation (random policy)"
    reference: ClassVar[str] = "S&B 4.1"
    implemented: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {"gamma": 0.99}

    def plan_step(self) -> float:
        raise NotImplementedError("PolicyEvaluation.plan_step: see the docstring.")


class PolicyIteration(PlanningAgent):
    """Policy Iteration: alternate full evaluation and greedy improvement.

    Suggested design for ``plan_step`` (one call = one evaluate+improve cycle):

    1. Evaluate ``self.pi`` until max |delta| < ``self.eval_tol``.
    2. Make ``self.pi`` greedy w.r.t. ``self.q_from_v(self.V)``.
    3. Return the max change in V during this cycle (0 when the policy is stable).
    """

    name: ClassVar[str] = "Policy Iteration"
    reference: ClassVar[str] = "S&B 4.3"
    implemented: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {"gamma": 0.99, "eval_tol": 1e-6}

    def __init__(self, env, gamma: float = 0.99, eval_tol: float = 1e-6, seed: int | None = None):
        super().__init__(env, gamma=gamma, seed=seed)
        self.eval_tol = eval_tol

    def plan_step(self) -> float:
        raise NotImplementedError("PolicyIteration.plan_step: see the docstring.")
