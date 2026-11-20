"""Environment registry.

``ENVS`` maps an id to a factory plus metadata. The dashboard lists exactly
what is registered here, so adding an environment is one entry.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rlx.envs.classic import CliffWalking, FrozenLake, GridWorld
from rlx.envs.grid import ACTION_NAMES, GridEnv, GridRewards


@dataclass(frozen=True)
class EnvSpec:
    id: str
    label: str
    description: str
    factory: Callable[..., GridEnv]


ENVS: dict[str, EnvSpec] = {
    spec.id: spec
    for spec in [
        EnvSpec("gridworld", "GridWorld", "Deterministic maze, -1 per step, +10 at G.", GridWorld),
        EnvSpec(
            "frozenlake4",
            "FrozenLake 4x4",
            "Slippery ice: each move succeeds only 1/3 of the time. +1 at G.",
            lambda **kw: FrozenLake(size=4, **kw),
        ),
        EnvSpec(
            "frozenlake8",
            "FrozenLake 8x8",
            "The larger slippery lake.",
            lambda **kw: FrozenLake(size=8, **kw),
        ),
        EnvSpec(
            "cliffwalking",
            "CliffWalking",
            "S&B Example 6.6: -1 per step, -100 for stepping on the cliff.",
            CliffWalking,
        ),
    ]
}


def make_env(env_id: str, **kwargs) -> GridEnv:
    if env_id not in ENVS:
        raise KeyError(f"Unknown env '{env_id}'. Available: {sorted(ENVS)}")
    return ENVS[env_id].factory(**kwargs)


def make_custom_env(
    layout: Sequence[str],
    step: float = -1.0,
    goal: float = 10.0,
    hole: float = -10.0,
    cliff: float = -100.0,
    slip: float = 0.0,
    max_steps: int = 200,
    seed: int | None = None,
) -> GridEnv:
    """Build an env from a user-drawn layout (used by the dashboard editor)."""
    return GridEnv(
        layout,
        GridRewards(step=step, goal=goal, hole=hole, cliff=cliff),
        slip=slip,
        max_steps=max_steps,
        seed=seed,
    )


__all__ = [
    "ACTION_NAMES",
    "ENVS",
    "CliffWalking",
    "EnvSpec",
    "FrozenLake",
    "GridEnv",
    "GridRewards",
    "GridWorld",
    "make_custom_env",
    "make_env",
]
