import numpy as np
import pytest

from rlx.envs import ENVS, CliffWalking, FrozenLake, GridEnv, GridRewards, GridWorld, make_env
from rlx.envs.grid import DOWN, LEFT, RIGHT, UP


@pytest.mark.parametrize("env_id", ENVS)
def test_model_is_a_valid_distribution(env_id):
    env = make_env(env_id)
    assert len(env.P) == env.n_states
    for s in range(env.n_states):
        for a in range(env.n_actions):
            assert sum(p for p, *_ in env.P[s][a]) == pytest.approx(1.0)


def test_step_samples_from_the_model():
    """Empirical next-state frequencies must match P (FrozenLake is stochastic)."""
    env = FrozenLake(slippery=True, seed=0)
    s, a, n = env.to_state(1, 0), RIGHT, 6000
    counts = np.zeros(env.n_states)
    for _ in range(n):
        env.state, env.t = s, 0
        counts[env.step(a).state] += 1
    expected = env.transition_matrix()[s, a]
    assert np.allclose(counts / n, expected, atol=0.03)


def test_walls_and_borders_block_movement():
    env = GridEnv(["S#", ".G"], max_steps=None)
    env.reset()
    assert env.step(RIGHT).state == 0  # wall
    assert env.step(UP).state == 0  # border
    assert env.step(LEFT).state == 0  # border


def test_goal_is_terminal_and_absorbing():
    env = GridWorld()
    goal = env.to_state(4, 4)
    for a in range(4):
        assert env.P[goal][a] == [(1.0, goal, 0.0, True)]


def test_cliff_sends_back_to_start_with_penalty():
    env = CliffWalking()
    env.reset()
    step = env.step(RIGHT)
    assert step.state == env.start_states[0]
    assert step.reward == -100
    assert not step.terminated


def test_truncation_is_not_termination():
    env = GridWorld(max_steps=3)
    env.reset()
    steps = [env.step(UP) for _ in range(3)]
    assert steps[-1].truncated and not steps[-1].terminated
    assert not any(s.truncated for s in steps[:-1])


def test_reaching_goal_terminates_with_reward():
    env = GridEnv(["SG"], GridRewards(step=-1, goal=5))
    env.reset()
    step = env.step(RIGHT)
    assert step.terminated and step.reward == 5


def test_slip_splits_probability_between_perpendicular_moves():
    env = GridEnv(["...", ".S.", "..."], slip=0.2)
    s = env.to_state(1, 1)
    probs = {s2: p for p, s2, _, _ in env.P[s][UP]}
    assert probs[env.to_state(0, 1)] == pytest.approx(0.8)
    assert probs[env.to_state(1, 0)] == pytest.approx(0.1)
    assert probs[env.to_state(1, 2)] == pytest.approx(0.1)
    assert env.to_state(2, 1) not in probs  # never moves backwards


def test_seeding_is_reproducible():
    def rollout(seed):
        env = FrozenLake(seed=seed)
        env.reset()
        return [env.step(DOWN).state for _ in range(20)]

    assert rollout(1) == rollout(1)


@pytest.mark.parametrize(
    "layout",
    [[], ["S..", ".."], ["S.X"], ["..."]],
    ids=["empty", "ragged", "unknown-tile", "no-start"],
)
def test_invalid_layouts_are_rejected(layout):
    with pytest.raises(ValueError):
        GridEnv(layout)


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError):
        GridWorld().step(0)
