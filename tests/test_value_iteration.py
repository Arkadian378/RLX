import numpy as np
import pytest

from rlx.algorithms import make_agent
from rlx.envs import CliffWalking, FrozenLake, GridWorld
from rlx.training import History, evaluate, train


def solve(env, gamma):
    agent = make_agent("value_iteration", env, gamma=gamma)
    train(env, agent, max_iterations=10_000, tol=1e-10)
    return agent


def test_gridworld_shortest_path():
    env = GridWorld(seed=0)
    agent = solve(env, gamma=1.0)
    # 8 moves to the goal: 7 * (-1) + 10.
    assert agent.V[0] == pytest.approx(3.0)
    assert evaluate(env, agent, n_episodes=5) == pytest.approx(3.0)


def test_cliffwalking_optimal_path_hugs_the_cliff():
    env = CliffWalking(seed=0)
    agent = solve(env, gamma=1.0)
    assert agent.V[env.start_states[0]] == pytest.approx(-13.0)


def test_frozenlake_success_probability():
    # Known optimum for the 4x4 slippery map without a step limit: ~0.8235.
    agent = solve(FrozenLake(slippery=True), gamma=1.0)
    assert agent.V[0] == pytest.approx(0.8235, abs=1e-3)


def test_fixed_point_satisfies_bellman_optimality():
    env = FrozenLake(slippery=True)
    agent = solve(env, gamma=0.9)
    assert np.allclose(agent.V, agent.q_from_v(agent.V).max(axis=1), atol=1e-8)


def test_policy_is_greedy_and_normalised():
    env = GridWorld()
    agent = solve(env, gamma=0.9)
    pi = agent.policy()
    assert np.allclose(pi.sum(axis=1), 1.0)
    Q = agent.q_from_v(agent.V)
    for s in range(env.n_states):
        assert Q[s, pi[s].argmax()] == pytest.approx(Q[s].max())


def test_iterations_are_logged_and_delta_shrinks():
    env = GridWorld()
    hist = History()
    train(env, make_agent("value_iteration", env, gamma=0.9), callbacks=[hist], tol=1e-6)
    deltas = [it.delta for it in hist.iterations]
    assert deltas[-1] < 1e-6
    assert deltas[-1] <= deltas[0]
