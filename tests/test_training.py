import numpy as np

from rlx.algorithms import make_agent
from rlx.core import Agent, Transition
from rlx.envs import GridWorld
from rlx.training import Callback, EarlyStopping, History, run_episode, train


class Recorder(Agent):
    """A fake agent that records which hooks the loop calls, in order."""

    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions, seed=0)
        self.calls = []

    def act(self, state, greedy=False):
        return 1  # always right

    def begin_episode(self, state):
        self.calls.append("begin")

    def observe(self, transition):
        assert isinstance(transition, Transition)
        self.calls.append("observe")

    def end_episode(self):
        self.calls.append("end")

    def values(self):
        return np.zeros(self.n_states)

    def policy(self):
        return np.full((self.n_states, self.n_actions), 0.25)


def test_hooks_are_called_in_order():
    env = GridWorld(max_steps=3)
    agent = Recorder(env.n_states, env.n_actions)
    train(env, agent, n_episodes=2)
    episode = ["begin", "observe", "observe", "observe", "end"]
    assert agent.calls == episode * 2


def test_greedy_rollout_does_not_learn():
    env = GridWorld(max_steps=3)
    agent = Recorder(env.n_states, env.n_actions)
    log = run_episode(env, agent, learn=False)
    assert agent.calls == []
    assert log.length == 3 and len(log.states) == 4


def test_callbacks_receive_every_episode():
    env = GridWorld(seed=0)
    hist = History()
    events = []

    class Spy(Callback):
        def on_train_begin(self, run):
            events.append("begin")

        def on_train_end(self, run):
            events.append("end")

    run = train(env, make_agent("random", env, seed=0), 25, [hist, Spy()])
    assert run.episodes_done == 25
    assert [e.episode for e in hist.episodes] == list(range(25))
    assert events == ["begin", "end"]


def test_early_stopping():
    env = GridWorld(seed=0)
    run = train(
        env, make_agent("random", env, seed=0), 1000, [EarlyStopping(target=-1e9, window=5)]
    )
    assert run.stop_requested and run.episodes_done == 5


def test_training_is_reproducible_with_seed():
    def returns():
        env = GridWorld(seed=0)
        hist = History()
        train(env, make_agent("random", env, seed=0), 10, [hist], seed=42)
        return hist.returns

    assert np.array_equal(returns(), returns())
