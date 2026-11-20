"""The classic tabular environments, written from scratch as ``GridEnv`` presets."""

from __future__ import annotations

from rlx.envs.grid import GridEnv, GridRewards

GRIDWORLD_LAYOUT = [
    "S....",
    ".##..",
    "...#.",
    ".#...",
    "...#G",
]

FROZEN_LAKE_4x4 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

FROZEN_LAKE_8x8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]

CLIFF_LAYOUT = [
    "............",
    "............",
    "............",
    "SCCCCCCCCCCG",
]


def _frozen(rows: list[str]) -> list[str]:
    # FrozenLake maps traditionally use 'F' for frozen (free) cells.
    return [r.replace("F", ".") for r in rows]


class GridWorld(GridEnv):
    """Deterministic maze: -1 per step, +10 at the goal. Shortest path wins."""

    def __init__(self, slip: float = 0.0, max_steps: int | None = 100, seed: int | None = None):
        super().__init__(
            GRIDWORLD_LAYOUT,
            GridRewards(step=-1.0, goal=10.0),
            slip=slip,
            max_steps=max_steps,
            seed=seed,
            name="GridWorld",
        )


class FrozenLake(GridEnv):
    """Slippery lake: reach G (+1) without falling in a hole H (0, terminal).

    With ``slippery=True`` each move goes in the intended direction with
    probability 1/3 and in each perpendicular direction with probability 1/3.
    """

    def __init__(
        self,
        size: int = 4,
        slippery: bool = True,
        max_steps: int | None = 100,
        seed: int | None = None,
    ):
        maps = {4: FROZEN_LAKE_4x4, 8: FROZEN_LAKE_8x8}
        if size not in maps:
            raise ValueError("FrozenLake size must be 4 or 8.")
        super().__init__(
            _frozen(maps[size]),
            GridRewards(step=0.0, goal=1.0, hole=0.0),
            slip=2 / 3 if slippery else 0.0,
            max_steps=max_steps,
            seed=seed,
            name=f"FrozenLake {size}x{size}",
        )


class CliffWalking(GridEnv):
    """Sutton & Barto Example 6.6: -1 per step, -100 and back to start on the cliff."""

    def __init__(self, max_steps: int | None = 500, seed: int | None = None):
        super().__init__(
            CLIFF_LAYOUT,
            GridRewards(step=-1.0, goal=-1.0, cliff=-100.0),
            slip=0.0,
            max_steps=max_steps,
            seed=seed,
            name="CliffWalking",
        )
