"""FastAPI app: serves the static UI, a JSON catalog and one WebSocket per tab."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rlx.dashboard.session import Message, Session, catalog

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="RLX dashboard")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/catalog")
def get_catalog() -> Message:
    return catalog()


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue[Message] = asyncio.Queue()

    # Called from the training thread: hop onto the event loop safely.
    def emit(msg: Message) -> None:
        loop.call_soon_threadsafe(outbox.put_nowait, msg)

    session = Session(emit)

    async def sender() -> None:
        while True:
            await websocket.send_json(await outbox.get())

    send_task = asyncio.create_task(sender())
    try:
        while True:
            msg = await websocket.receive_json()
            # session.handle may block briefly (stop joins the thread): keep it off the loop.
            await asyncio.to_thread(session.handle, msg)
    except WebSocketDisconnect:
        pass
    finally:
        session.stop()
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await send_task


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the RLX dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"RLX dashboard on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
