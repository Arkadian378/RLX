"""Algorithm registry.

``ALGORITHMS`` is the single place the dashboard and tests look up agents.
To add an algorithm: write the class, then add one line here.
"""

from __future__ import annotations

from typing import Any

from rlx.algorithms.baselines.random_agent import RandomAgent
from rlx.algorithms.dp.policy_iteration import PolicyEvaluation, PolicyIteration
from rlx.algorithms.dp.value_iteration import ValueIteration
from rlx.algorithms.mc.mc_control import MonteCarloControl
from rlx.algorithms.planning.dyna_q import DynaQ
from rlx.algorithms.td.n_step import NStepSarsa
from rlx.algorithms.td.td_control import ExpectedSarsa, QLearning, Sarsa
from rlx.core.agent import Agent
from rlx.core.env import DiscreteEnv

# Ordered like the Sutton & Barto chapters, which is the suggested study order.
ALGORITHMS: dict[str, type[Agent]] = {
    "random": RandomAgent,
    "policy_evaluation": PolicyEvaluation,
    "policy_iteration": PolicyIteration,
    "value_iteration": ValueIteration,
    "mc_control": MonteCarloControl,
    "sarsa": Sarsa,
    "q_learning": QLearning,
    "expected_sarsa": ExpectedSarsa,
    "n_step_sarsa": NStepSarsa,
    "dyna_q": DynaQ,
}


def make_agent(algo_id: str, env: DiscreteEnv, seed: int | None = None, **params: Any) -> Agent:
    """Instantiate an agent for ``env``.

    Planning agents receive the env (they need ``P``); online agents only get
    the sizes of the state and action spaces, so they *cannot* peek at the model.
    """
    if algo_id not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{algo_id}'. Available: {sorted(ALGORITHMS)}")
    cls = ALGORITHMS[algo_id]
    unknown = set(params) - set(cls.default_params)
    if unknown:
        raise ValueError(f"{cls.name} does not accept {sorted(unknown)}.")
    kwargs = {**cls.default_params, **params}
    if cls.kind == "planning":
        return cls(env, seed=seed, **kwargs)
    return cls(env.n_states, env.n_actions, seed=seed, **kwargs)


__all__ = ["ALGORITHMS", "make_agent"]
