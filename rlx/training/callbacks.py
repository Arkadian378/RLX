"""Callbacks: observe training without touching the loop.

The loop calls these hooks; a callback can record metrics, print, stream to
the dashboard, or stop training by setting ``run.stop_requested = True``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rlx.training.loop import Run


@dataclass(frozen=True)
class EpisodeLog:
    """Summary of one episode."""

    episode: int
    ret: float  # undiscounted return
    length: int
    terminated: bool
    states: tuple[int, ...] = ()  # visited states, for replaying the trajectory


@dataclass(frozen=True)
class IterationLog:
    """Summary of one planning iteration (e.g. one DP sweep)."""

    iteration: int
    delta: float
    eval_return: float | None = None  # mean return of the greedy policy, if evaluated


class Callback:
    """Base class: override only the hooks you need."""

    def on_train_begin(self, run: Run) -> None: ...

    def on_episode_end(self, run: Run, log: EpisodeLog) -> None: ...

    def on_iteration_end(self, run: Run, log: IterationLog) -> None: ...

    def on_train_end(self, run: Run) -> None: ...


class CallbackList(Callback):
    """Fans every hook out to a list of callbacks (composite pattern)."""

    def __init__(self, callbacks: Iterable[Callback] = ()) -> None:
        self.callbacks = list(callbacks)

    def on_train_begin(self, run: Run) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(run)

    def on_episode_end(self, run: Run, log: EpisodeLog) -> None:
        for cb in self.callbacks:
            cb.on_episode_end(run, log)

    def on_iteration_end(self, run: Run, log: IterationLog) -> None:
        for cb in self.callbacks:
            cb.on_iteration_end(run, log)

    def on_train_end(self, run: Run) -> None:
        for cb in self.callbacks:
            cb.on_train_end(run)


@dataclass
class History(Callback):
    """Keeps every log in memory: the simplest way to plot a learning curve."""

    episodes: list[EpisodeLog] = field(default_factory=list)
    iterations: list[IterationLog] = field(default_factory=list)

    def on_episode_end(self, run: Run, log: EpisodeLog) -> None:
        self.episodes.append(log)

    def on_iteration_end(self, run: Run, log: IterationLog) -> None:
        self.iterations.append(log)

    @property
    def returns(self) -> np.ndarray:
        return np.array([e.ret for e in self.episodes])


class PrintLogger(Callback):
    """Prints a moving average every ``every`` episodes / iterations."""

    def __init__(self, every: int = 100, window: int = 100) -> None:
        self.every = every
        self.window = window
        self._returns: list[float] = []

    def on_episode_end(self, run: Run, log: EpisodeLog) -> None:
        self._returns.append(log.ret)
        if (log.episode + 1) % self.every == 0:
            avg = np.mean(self._returns[-self.window :])
            print(f"episode {log.episode + 1:>6}  avg return {avg:9.3f}")

    def on_iteration_end(self, run: Run, log: IterationLog) -> None:
        if (log.iteration + 1) % self.every == 0:
            print(f"iteration {log.iteration + 1:>5}  delta {log.delta:.2e}")


class EarlyStopping(Callback):
    """Stops when the moving-average return reaches ``target``."""

    def __init__(self, target: float, window: int = 100) -> None:
        self.target = target
        self.window = window
        self._returns: list[float] = []

    def on_episode_end(self, run: Run, log: EpisodeLog) -> None:
        self._returns.append(log.ret)
        if (
            len(self._returns) >= self.window
            and np.mean(self._returns[-self.window :]) >= self.target
        ):
            run.stop_requested = True


class LambdaCallback(Callback):
    """Wrap plain functions as a callback, handy in notebooks and the dashboard."""

    def __init__(
        self,
        on_episode_end: Callable[[Run, EpisodeLog], None] | None = None,
        on_iteration_end: Callable[[Run, IterationLog], None] | None = None,
    ) -> None:
        self._on_episode_end = on_episode_end
        self._on_iteration_end = on_iteration_end

    def on_episode_end(self, run: Run, log: EpisodeLog) -> None:
        if self._on_episode_end:
            self._on_episode_end(run, log)

    def on_iteration_end(self, run: Run, log: IterationLog) -> None:
        if self._on_iteration_end:
            self._on_iteration_end(run, log)
