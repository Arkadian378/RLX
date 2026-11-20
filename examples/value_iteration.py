"""Solve CliffWalking with Value Iteration and print the greedy policy.

Run with:  python examples/value_iteration.py
"""

from rlx.algorithms import make_agent
from rlx.envs import CliffWalking
from rlx.training import PrintLogger, evaluate, train

ARROW = "^>v<"

env = CliffWalking(seed=0)
agent = make_agent("value_iteration", env, gamma=1.0)
run = train(env, agent, callbacks=[PrintLogger(every=5)], tol=1e-10)

print(f"\nconverged in {run.iterations_done} sweeps, greedy return {evaluate(env, agent, 1)}\n")
for r in range(env.n_rows):
    row = ""
    for c in range(env.n_cols):
        s = env.to_state(r, c)
        tile = env.tile(s)
        row += tile if tile in "GC#H" else ARROW[agent.policy()[s].argmax()]
    print(row)
