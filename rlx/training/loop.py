"""The training loop.

``train`` is the only entry point most code needs. It dispatches on the
agent's kind:

* online agents learn from ``n_episodes`` of interaction;
* planning agents run ``plan_step`` until convergence (or ``max_iterations``),
  and optionally evaluate their greedy policy after each step so they produce
  a learning curve comparable to online agents.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from rlx.core.agent import Agent, PlanningAgent, Transition
from rlx.core.env import Env
from rlx.training.callbacks import Callback, CallbackList, EpisodeLog, IterationLog


@dataclass
class Run:
    """Shared state of one training run, visible to every callback."""

    env: Env
    agent: Agent
    stop_requested: bool = False
    episodes_done: int = 0
    iterations_done: int = 0


def run_episode(env: Env, agent: Agent, learn: bool = True, seed: int | None = None) -> EpisodeLog:
    """Play one episode. With ``learn=False`` the agent acts greedily and isn't updated."""
    state = env.reset(seed=seed)
    states = [state]
    if learn:
        agent.begin_episode(state)

    ret, length = 0.0, 0
    while True:
        action = agent.act(state, greedy=not learn)
        step = env.step(action)
        if learn:
            agent.observe(
                Transition(state, action, step.reward, step.state, step.terminated, step.truncated)
            )
        ret += step.reward
        length += 1
        state = step.state
        states.append(state)
        if step.done:
            break

    if learn:
        agent.end_episode()
    return EpisodeLog(-1, ret, length, step.terminated, tuple(states))


def evaluate(env: Env, agent: Agent, n_episodes: int = 10, seed: int | None = None) -> float:
    """Mean undiscounted return of the greedy policy."""
    returns = [
        run_episode(env, agent, learn=False, seed=None if seed is None else seed + i).ret
        for i in range(n_episodes)
    ]
    return float(np.mean(returns))


def train(
    env: Env,
    agent: Agent,
    n_episodes: int = 500,
    callbacks: Iterable[Callback] = (),
    *,
    max_iterations: int = 1000,
    tol: float = 1e-8,
    eval_episodes: int = 0,
    seed: int | None = None,
) -> Run:
    """Train ``agent`` on ``env`` and return the finished ``Run``.

    Args:
        n_episodes: episodes of interaction (online agents).
        max_iterations, tol: planning budget and convergence threshold.
        eval_episodes: planning agents only, greedy rollouts after every step.
        seed: seeds the first reset; later episodes continue the env's RNG stream.
    """
    run = Run(env, agent)
    cbs = CallbackList(callbacks)
    cbs.on_train_begin(run)

    if isinstance(agent, PlanningAgent):
        _plan(run, cbs, max_iterations, tol, eval_episodes, seed)
    else:
        _learn(run, cbs, n_episodes, seed)

    cbs.on_train_end(run)
    return run


def _learn(run: Run, cbs: CallbackList, n_episodes: int, seed: int | None) -> None:
    for ep in range(n_episodes):
        log = run_episode(run.env, run.agent, learn=True, seed=seed if ep == 0 else None)
        log = EpisodeLog(ep, log.ret, log.length, log.terminated, log.states)
        run.episodes_done = ep + 1
        cbs.on_episode_end(run, log)
        if run.stop_requested:
            break


def _plan(
    run: Run,
    cbs: CallbackList,
    max_iterations: int,
    tol: float,
    eval_episodes: int,
    seed: int | None,
) -> None:
    agent = run.agent
    assert isinstance(agent, PlanningAgent)
    if seed is not None:
        run.env.reset(seed=seed)
    for it in range(max_iterations):
        delta = agent.plan_step()
        eval_ret = evaluate(run.env, agent, eval_episodes) if eval_episodes > 0 else None
        run.iterations_done = it + 1
        cbs.on_iteration_end(run, IterationLog(it, delta, eval_ret))
        if delta < tol or run.stop_requested:
            break
