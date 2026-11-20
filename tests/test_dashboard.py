from fastapi.testclient import TestClient

from rlx.dashboard.server import app

client = TestClient(app)


def receive_until(ws, msg_type, limit=2000):
    for _ in range(limit):
        msg = ws.receive_json()
        if msg["type"] in (msg_type, "error"):
            return msg
    raise AssertionError(f"no {msg_type} message")


def test_index_and_catalog():
    assert client.get("/").status_code == 200
    cat = client.get("/api/catalog").json()
    assert {e["id"] for e in cat["envs"]} >= {"gridworld", "frozenlake4", "cliffwalking"}
    vi = next(a for a in cat["algorithms"] if a["id"] == "value_iteration")
    assert vi["implemented"] and vi["kind"] == "planning"


def test_value_iteration_streams_until_done():
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "env": {"id": "gridworld"},
                "algo": "value_iteration",
                "params": {"gamma": 0.9},
                "delay_ms": 0,
            }
        )
        started = ws.receive_json()
        assert started["type"] == "started" and len(started["V"]) == 25
        it = receive_until(ws, "iteration")
        assert len(it["policy"]) == 25 and len(it["Q"][0]) == 4
        done = receive_until(ws, "done")
        assert done["type"] == "done" and done["iterations"] > 0

        ws.send_json({"type": "rollout"})
        roll = receive_until(ws, "rollout")
        assert roll["states"][0] == 0 and roll["ret"] > 0


def test_online_agent_streams_episodes():
    with client.websocket_connect("/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "env": {"id": "gridworld"},
                "algo": "random",
                "episodes": 20,
                "seed": 1,
            }
        )
        receive_until(ws, "started")
        total = 0
        while True:
            msg = ws.receive_json()
            if msg["type"] == "episodes":
                total += len(msg["returns"])
            if msg["type"] == "done":
                break
        assert total == 20


def test_custom_layout_and_play_mode():
    custom = {"custom": {"layout": ["S.#", "..G"], "step": -1, "goal": 5}}
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "play_reset", "env": custom})
        assert receive_until(ws, "play")["state"] == 0
        ws.send_json({"type": "play_step", "action": 2})  # down
        ws.send_json({"type": "play_step", "action": 1})  # right
        ws.send_json({"type": "play_step", "action": 1})  # right -> goal
        msgs = [receive_until(ws, "play") for _ in range(3)]
        assert msgs[-1]["done"] and msgs[-1]["ret"] == 3


def test_errors_are_reported_not_fatal():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "env": {"id": "nope"}, "algo": "random"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "rollout"})
        assert ws.receive_json()["type"] == "error"
