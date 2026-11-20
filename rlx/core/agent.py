"""Agent interfaces.

The training loop talks to every agent through the same small set of hooks:

    begin_episode(state) -> act(state) -> observe(transition) -> ... -> end_episode()

This shape fits every tabular method in Sutton & Barto:

* one-step TD (SARSA, Q-learning) learns inside ``observe``;
* Monte Carlo stores the episode in ``observe`` and learns in ``end_episode``;
* n-step methods keep a sliding buffer across ``observe`` calls;
* Dyna-Q learns in ``observe`` and then runs extra planning updates there.

Planning agents (Dynamic Programming) do not need experience at all: they read
the model of a ``DiscreteEnv`` and improve through ``plan_step``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from rlx.core.env import DiscreteEnv


@dataclass(frozen=True)
class Transition:
    """One step of experience (S_t, A_t, R_{t+1}, S_{t+1})."""

    state: int
    action: int
    reward: float
    next_state: int
    terminated: bool
    truncated: bool = False


class Agent(ABC):
    """Base class for every agent.

    Class attributes describe the algorithm to the registry and dashboard:

    * ``name``: human readable name.
    * ``kind``: ``"online"`` (learns from experience) or ``"planning"`` (uses P).
    * ``implemented``: ``False`` for stubs waiting to be written.
    * ``default_params``: hyperparameters with their defaults; the dashboard
      builds its form from this dict.
    """

    name: ClassVar[str] = "agent"
    kind: ClassVar[str] = "online"
    implemented: ClassVar[bool] = True
    reference: ClassVar[str] = ""
    default_params: ClassVar[dict[str, Any]] = {}

    def __init__(self, n_states: int, n_actions: int, seed: int | None = None) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ acting
    @abstractmethod
    def act(self, state: int, greedy: bool = False) -> int:
        """Choose an action. ``greedy=True`` disables exploration (evaluation)."""

    # ----------------------------------------------------------- learning
    def begin_episode(self, state: int) -> None:  # noqa: B027 - optional hook
        """Called once after ``env.reset()``."""

    def observe(self, transition: Transition) -> None:  # noqa: B027 - optional hook
        """Called after every environment step. Online learning goes here."""

    def end_episode(self) -> None:  # noqa: B027 - optional hook
        """Called when the episode terminates or is truncated."""

    # ------------------------------------------------------- introspection
    @abstractmethod
    def values(self) -> np.ndarray:
        """State-value estimate V(s), shape ``(n_states,)``. Used for plots."""

    @abstractmethod
    def policy(self) -> np.ndarray:
        """Action probabilities pi(a|s), shape ``(n_states, n_actions)``."""

    def q_values(self) -> np.ndarray | None:
        """Action-value estimate Q(s, a) if the agent has one, else ``None``."""
        return None


class TabularAgent(Agent):
    """An agent backed by a Q table with an epsilon-greedy behaviour policy.

    Most S&B control algorithms only differ in *how they update Q*, so all the
    shared plumbing (table, exploration, greedy policy, V = max Q) lives here.
    Subclasses implement ``observe`` and/or ``end_episode``.
    """

    default_params: ClassVar[dict[str, Any]] = {"gamma": 0.99, "alpha": 0.1, "epsilon": 0.1}

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        super().__init__(n_states, n_actions, seed)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def greedy_action(self, state: int) -> int:
        """argmax_a Q(s, a) with random tie-breaking.

        Plain ``np.argmax`` always picks the first maximum, which biases early
        learning (all zeros) toward action 0.
        """
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def act(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy_action(state)

    def values(self) -> np.ndarray:
        return self.Q.max(axis=1)

    def q_values(self) -> np.ndarray:
        return self.Q

    def policy(self) -> np.ndarray:
        """Epsilon-greedy probabilities with respect to Q."""
        pi = np.full((self.n_states, self.n_actions), self.epsilon / self.n_actions)
        best = self.Q.argmax(axis=1)
        pi[np.arange(self.n_states), best] += 1.0 - self.epsilon
        return pi


class PlanningAgent(Agent):
    """An agent that computes its policy from a known model (Dynamic Programming).

    ``plan_step`` performs one unit of work (e.g. one sweep) and returns the
    largest value change, which the training loop uses to detect convergence.
    """

    kind: ClassVar[str] = "planning"

    def __init__(self, env: DiscreteEnv, gamma: float = 0.99, seed: int | None = None) -> None:
        super().__init__(env.n_states, env.n_actions, seed)
        self.env = env
        self.gamma = gamma
        self.V = np.zeros(env.n_states)
        self.pi = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)

    @abstractmethod
    def plan_step(self) -> float:
        """Do one planning iteration and return max |delta V|."""

    def act(self, state: int, greedy: bool = False) -> int:
        probs = self.pi[state]
        if greedy:
            best = np.flatnonzero(probs == probs.max())
            return int(self.rng.choice(best))
        return int(self.rng.choice(self.n_actions, p=probs))

    def values(self) -> np.ndarray:
        return self.V

    def policy(self) -> np.ndarray:
        return self.pi

    def q_values(self) -> np.ndarray:
        return self.q_from_v(self.V)

    # Shared DP helper -------------------------------------------------
    def q_from_v(self, V: np.ndarray) -> np.ndarray:
        """One-step lookahead: Q(s,a) = sum_{s',r} p(s',r|s,a) [r + gamma V(s') (1-done)]."""
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Q[s, a] = sum(
                    p * (r + self.gamma * V[s2] * (not done)) for p, s2, r, done in self.env.P[s][a]
                )
        return Q
