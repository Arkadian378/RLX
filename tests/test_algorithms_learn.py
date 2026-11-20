"""Behavioural tests for the algorithms you are going to write.

Each test is *skipped* while the algorithm is still a stub (it raises
``NotImplementedError``) and turns into a real check once you implement it.
"""

import numpy as np
import pytest

from rlx.algorithms import ALGORITHMS, make_agent
from rlx.envs import GridWorld
from rlx.training import evaluate, train

ONLINE = [k for k, cls in ALGORITHMS.items() if cls.kind == "online" and k != "random"]


def _skip_if_stub(fn):
    try:
        return fn()
    except NotImplementedError as exc:
        pytest.skip(f"not implemented yet: {exc}")


@pytest.mark.parametrize("algo_id", ONLINE)
def test_online_algorithm_solves_gridworld(algo_id):
    env = GridWorld(seed=0)
    agent = make_agent(algo_id, env, seed=0, gamma=0.95)
    _skip_if_stub(lambda: train(env, agent, n_episodes=500, seed=0))
    # Optimal greedy return is 3 (8 steps); allow a slightly longer path.
    assert evaluate(env, agent, n_episodes=1) >= 0


def test_policy_evaluation_matches_linear_solve():
    env = GridWorld()
    gamma = 0.9
    agent = make_agent("policy_evaluation", env, gamma=gamma)
    _skip_if_stub(lambda: train(env, agent, max_iterations=10_000, tol=1e-10))

    # Solve V = R_pi + gamma * P_pi V exactly, where P_pi only keeps
    # transitions that don't end the episode (terminal successors are worth 0).
    pi = agent.policy()
    P_pi = np.zeros((env.n_states, env.n_states))
    R_pi = np.zeros(env.n_states)
    for s in range(env.n_states):
        for a in range(env.n_actions):
            for p, s2, r, done in env.P[s][a]:
                R_pi[s] += pi[s, a] * p * r
                if not done:
                    P_pi[s, s2] += pi[s, a] * p
    V = np.linalg.solve(np.eye(env.n_states) - gamma * P_pi, R_pi)
    assert np.allclose(agent.V, V, atol=1e-6)


def test_policy_iteration_matches_value_iteration():
    env = GridWorld()
    pi_agent = make_agent("policy_iteration", env, gamma=0.9)
    _skip_if_stub(lambda: train(env, pi_agent, max_iterations=100, tol=1e-10))
    vi_agent = make_agent("value_iteration", env, gamma=0.9)
    train(env, vi_agent, max_iterations=10_000, tol=1e-12)
    assert np.allclose(pi_agent.V, vi_agent.V, atol=1e-5)
