"""One dashboard connection = one ``Session``.

The session owns the environment, the agent and a background training thread.
Training is ordinary synchronous Python (the same ``train`` used in scripts);
a ``StreamCallback`` turns its hooks into JSON messages and hands them to the
asyncio side through a thread-safe ``emit`` function.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from rlx.algorithms import ALGORITHMS, make_agent
from rlx.core.agent import Agent
from rlx.envs import ENVS, GridEnv, make_custom_env, make_env
from rlx.training import Callback, EpisodeLog, IterationLog, Run, run_episode, train

Message = dict[str, Any]

# Without an artificial delay training is far faster than a browser can draw,
# so snapshots are rate limited and episode returns are sent in batches.
MIN_SNAPSHOT_INTERVAL_S = 1 / 30


def catalog() -> Message:
    """Everything the UI needs to build its menus."""
    return {
        "envs": [
            {
                "id": s.id,
                "label": s.label,
                "description": s.description,
                "layout": s.factory().layout,
            }
            for s in ENVS.values()
        ],
        "algorithms": [
            {
                "id": algo_id,
                "name": cls.name,
                "kind": cls.kind,
                "implemented": cls.implemented,
                "reference": cls.reference,
                "params": cls.default_params,
            }
            for algo_id, cls in ALGORITHMS.items()
        ],
    }


def build_env(cfg: Message, seed: int | None = None) -> GridEnv:
    """``cfg`` is either ``{"id": ...}`` or ``{"custom": {...layout and rewards...}}``."""
    if "custom" in cfg:
        return make_custom_env(seed=seed, **cfg["custom"])
    return make_env(cfg["id"], seed=seed)


def snapshot(agent: Agent) -> Message:
    q = agent.q_values()
    return {
        "V": _clean(agent.values()),
        "policy": _clean(agent.policy()),
        "Q": _clean(q) if q is not None else None,
    }


def _clean(a: np.ndarray) -> list:
    # JSON has no inf/nan; round to keep messages small.
    return np.nan_to_num(np.round(np.asarray(a, dtype=float), 4), posinf=0, neginf=0).tolist()


class StreamCallback(Callback):
    def __init__(self, session: Session) -> None:
        self.s = session
        self._last_emit = 0.0
        self._pending: list[tuple[float, int]] = []

    def _should_emit(self) -> bool:
        return self.s.delay_s > 0 or time.monotonic() - self._last_emit >= MIN_SNAPSHOT_INTERVAL_S

    def on_episode_end(self, run: Run, log: EpisodeLog) -> None:
        self._pending.append((log.ret, log.length))
        if self._should_emit():
            self._flush(run, log)
        self._pace(run)

    def on_iteration_end(self, run: Run, log: IterationLog) -> None:
        # Planning runs are short: send every iteration.
        self.s.emit(
            {
                "type": "iteration",
                "iteration": log.iteration,
                "delta": log.delta,
                "eval_return": log.eval_return,
                **snapshot(run.agent),
            }
        )
        self._pace(run, minimum=0.05)

    def on_train_end(self, run: Run) -> None:
        if self._pending:
            self._flush(run, None)

    def _flush(self, run: Run, log: EpisodeLog | None) -> None:
        self.s.emit(
            {
                "type": "episodes",
                "first": run.episodes_done - len(self._pending),
                "returns": [r for r, _ in self._pending],
                "lengths": [n for _, n in self._pending],
                "trajectory": list(log.states) if log else None,
                **snapshot(run.agent),
            }
        )
        self._pending = []
        self._last_emit = time.monotonic()

    def _pace(self, run: Run, minimum: float = 0.0) -> None:
        if self.s.stop_event.is_set():
            run.stop_requested = True
        delay = max(self.s.delay_s, minimum)
        if delay:
            time.sleep(delay)


class Session:
    def __init__(self, emit: Callable[[Message], None]) -> None:
        self.emit = emit
        self.env: GridEnv | None = None
        self.agent: Agent | None = None
        self.play_env: GridEnv | None = None
        self.play_return = 0.0
        self.delay_s = 0.0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    # ------------------------------------------------------------ dispatch
    def handle(self, msg: Message) -> None:
        handlers = {
            "start": self.start,
            "stop": lambda m: self.stop(),
            "speed": self.set_speed,
            "rollout": self.rollout,
            "play_reset": self.play_reset,
            "play_step": self.play_step,
        }
        handler = handlers.get(msg.get("type"))
        if handler is None:
            self.emit({"type": "error", "message": f"Unknown message type {msg.get('type')!r}"})
            return
        try:
            handler(msg)
        except Exception as exc:  # report any bad input to the UI instead of dropping the socket
            self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    @property
    def training(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    # ------------------------------------------------------------ training
    def start(self, msg: Message) -> None:
        self.stop()
        seed = msg.get("seed")
        self.env = build_env(msg["env"], seed=seed)
        self.agent = make_agent(msg["algo"], self.env, seed=seed, **msg.get("params", {}))
        self.set_speed(msg)
        self.stop_event.clear()
        self.emit(
            {
                "type": "started",
                "layout": self.env.layout,
                "kind": self.agent.kind,
                "algo": self.agent.name,
                **snapshot(self.agent),
            }
        )
        n_episodes = int(msg.get("episodes", 500))
        self.thread = threading.Thread(target=self._train, args=(n_episodes, seed), daemon=True)
        self.thread.start()

    def _train(self, n_episodes: int, seed: int | None) -> None:
        assert self.env is not None and self.agent is not None
        try:
            run = train(
                self.env,
                self.agent,
                n_episodes,
                [StreamCallback(self)],
                max_iterations=500,
                tol=1e-6,
                eval_episodes=10,
                seed=seed,
            )
            self.emit(
                {
                    "type": "done",
                    "stopped": run.stop_requested,
                    "episodes": run.episodes_done,
                    "iterations": run.iterations_done,
                }
            )
        except NotImplementedError as exc:
            self.emit({"type": "error", "message": f"Not implemented yet: {exc}"})
        except Exception as exc:
            self.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    def stop(self) -> None:
        if self.training:
            self.stop_event.set()
            assert self.thread is not None
            self.thread.join(timeout=5)

    def set_speed(self, msg: Message) -> None:
        self.delay_s = max(0.0, float(msg.get("delay_ms", 0))) / 1000

    # -------------------------------------------------------- watch / play
    def rollout(self, msg: Message) -> None:
        """One greedy episode of the current agent, for the UI to animate."""
        if self.env is None or self.agent is None:
            raise RuntimeError("Train an agent first.")
        if self.training:
            raise RuntimeError("Wait for training to finish (or stop it).")
        log = run_episode(self.env, self.agent, learn=False)
        self.emit({"type": "rollout", "states": list(log.states), "ret": log.ret})

    def play_reset(self, msg: Message) -> None:
        self.play_env = build_env(msg["env"])
        self.play_return = 0.0
        state = self.play_env.reset()
        self.emit(
            {
                "type": "play",
                "layout": self.play_env.layout,
                "state": state,
                "reward": 0.0,
                "ret": 0.0,
                "done": False,
            }
        )

    def play_step(self, msg: Message) -> None:
        if self.play_env is None:
            raise RuntimeError("Start play mode first.")
        step = self.play_env.step(int(msg["action"]))
        self.play_return += step.reward
        self.emit(
            {
                "type": "play",
                "state": step.state,
                "reward": step.reward,
                "ret": self.play_return,
                "done": step.done,
            }
        )
