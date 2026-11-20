"""A generic grid-world MDP built from a text layout.

Every tabular grid environment in RLX (GridWorld, FrozenLake, CliffWalking and
anything drawn in the dashboard editor) is a ``GridEnv`` with a different
layout and reward table. The layout is a list of equal-length strings:

    S  start            G  goal (terminal, reward ``goal``)
    .  free cell        H  hole (terminal, reward ``hole``)
    #  wall (blocked)   C  cliff (reward ``cliff``, teleports back to S)

States are cells numbered row-major: ``s = row * n_cols + col``. Walls are
still states (they are just unreachable) so that ``s`` always maps to a cell,
which keeps plotting trivial.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rlx.core.env import DiscreteEnv, Outcome, TransitionModel

UP, RIGHT, DOWN, LEFT = range(4)
ACTION_NAMES = ("up", "right", "down", "left")
# (d_row, d_col) for each action, indexed by the action id.
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))

TILES = set("SG.#HC")
TERMINAL_TILES = set("GH")


@dataclass(frozen=True)
class GridRewards:
    """Reward received when *entering* a cell of each type."""

    step: float = 0.0
    goal: float = 1.0
    hole: float = 0.0
    cliff: float = -100.0


class GridEnv(DiscreteEnv):
    """A grid MDP with optional slippery dynamics.

    Args:
        layout: rows of tile characters (see module docstring).
        rewards: reward table.
        slip: probability of *not* doing the intended move. The slip mass is
            split evenly between the two perpendicular moves, so with
            ``slip = 2/3`` every one of the three moves has probability 1/3
            (the classic FrozenLake dynamics).
        max_steps: episode step limit (truncation), ``None`` for no limit.
    """

    def __init__(
        self,
        layout: Sequence[str],
        rewards: GridRewards | None = None,
        slip: float = 0.0,
        max_steps: int | None = 200,
        seed: int | None = None,
        name: str = "Custom grid",
    ) -> None:
        self.layout = self._parse(layout)
        self.n_rows = len(self.layout)
        self.n_cols = len(self.layout[0])
        self.rewards = rewards or GridRewards()
        self.slip = slip
        self.name = name

        starts = [self.to_state(r, c) for r, c in self._cells("S")]
        if not starts:
            raise ValueError("Layout needs at least one start cell 'S'.")
        self.start_states = starts
        init = np.zeros(self.n_rows * self.n_cols)
        init[starts] = 1.0 / len(starts)

        super().__init__(self._build_model(), init, max_steps=max_steps, seed=seed)

    # -------------------------------------------------------- coordinates
    def to_state(self, row: int, col: int) -> int:
        return row * self.n_cols + col

    def to_cell(self, state: int) -> tuple[int, int]:
        return divmod(state, self.n_cols)

    def tile(self, state: int) -> str:
        r, c = self.to_cell(state)
        return self.layout[r][c]

    # ------------------------------------------------------ model building
    def _build_model(self) -> TransitionModel:
        P: TransitionModel = []
        for s in range(self.n_rows * self.n_cols):
            P.append([self._outcomes(s, a) for a in range(4)])
        return P

    def _outcomes(self, state: int, action: int) -> list[Outcome]:
        # Terminal (and unreachable wall) states are absorbing with zero reward:
        # this is what makes "V(terminal) = 0" fall out of the Bellman equation.
        if self.tile(state) in TERMINAL_TILES or self.tile(state) == "#":
            return [(1.0, state, 0.0, True)]

        # Intended move with prob 1-slip, each perpendicular move with slip/2.
        moves = [(action, 1.0 - self.slip)]
        if self.slip > 0:
            moves += [((action + 1) % 4, self.slip / 2), ((action - 1) % 4, self.slip / 2)]

        # Merge outcomes that land in the same place (e.g. two moves into walls).
        merged: dict[tuple[int, float, bool], float] = {}
        for a, prob in moves:
            key = self._move(state, a)
            merged[key] = merged.get(key, 0.0) + prob
        return [(p, s2, r, done) for (s2, r, done), p in merged.items() if p > 0]

    def _move(self, state: int, action: int) -> tuple[int, float, bool]:
        """Deterministic effect of a move: (next_state, reward, done)."""
        r, c = self.to_cell(state)
        dr, dc = MOVES[action]
        nr, nc = r + dr, c + dc
        # Bumping into the border or a wall leaves the agent where it is.
        if not (0 <= nr < self.n_rows and 0 <= nc < self.n_cols) or self.layout[nr][nc] == "#":
            nr, nc = r, c

        tile = self.layout[nr][nc]
        if tile == "G":
            return self.to_state(nr, nc), self.rewards.goal, True
        if tile == "H":
            return self.to_state(nr, nc), self.rewards.hole, True
        if tile == "C":
            # Cliff: big penalty and back to the (first) start, episode continues.
            return self.start_states[0], self.rewards.cliff, False
        return self.to_state(nr, nc), self.rewards.step, False

    # ------------------------------------------------------------ parsing
    @staticmethod
    def _parse(layout: Sequence[str]) -> list[str]:
        rows = [row.strip() for row in layout if row.strip()]
        if not rows:
            raise ValueError("Empty layout.")
        if len({len(r) for r in rows}) != 1:
            raise ValueError("All layout rows must have the same length.")
        bad = set("".join(rows)) - TILES
        if bad:
            raise ValueError(f"Unknown tiles {sorted(bad)}; allowed: {sorted(TILES)}.")
        return rows

    def _cells(self, tile: str) -> list[tuple[int, int]]:
        return [
            (r, c) for r, row in enumerate(self.layout) for c, ch in enumerate(row) if ch == tile
        ]

    # ------------------------------------------------------------- render
    def render(self) -> str:
        """ASCII picture with the agent drawn as 'A'."""
        lines = []
        for r, row in enumerate(self.layout):
            chars = list(row)
            if self.state is not None and self.to_cell(self.state)[0] == r:
                chars[self.to_cell(self.state)[1]] = "A"
            lines.append("".join(chars))
        return "\n".join(lines)
