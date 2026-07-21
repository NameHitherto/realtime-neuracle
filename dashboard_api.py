from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from telemetry_core import TelemetryBus

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dashboard" / "dist"


class ManualControl(BaseModel):
    control: int = Field(ge=0, le=3)
    duration_ms: int = Field(default=1000, ge=100, le=10000)


def create_app(bus: TelemetryBus, runtime: Any) -> FastAPI:
    app = FastAPI(title="BCI Racing Telemetry", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "telemetry": bus.snapshot().get("health", {})}

    @app.get("/api/status")
    def status() -> dict[str, Any]:
        return bus.snapshot()

    @app.get("/api/events")
    def events(limit: int = 100) -> list[dict[str, Any]]:
        return bus.events(limit)

    @app.get("/api/history")
    def history(limit: int = 120) -> list[dict[str, Any]]:
        return bus.history(limit)

    @app.post("/api/test/control")
    def test_control(payload: ManualControl) -> dict[str, Any]:
        try:
            runtime.manual_control(payload.control, payload.duration_ms)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return bus.snapshot()

    @app.websocket("/ws/telemetry")
    async def telemetry_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4)

        def on_update(snapshot: dict[str, Any]) -> None:
            def enqueue() -> None:
                if queue.full():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(snapshot)

            loop.call_soon_threadsafe(enqueue)

        bus.subscribe(on_update)
        try:
            await websocket.send_json(bus.snapshot())
            while True:
                snapshot = await queue.get()
                await websocket.send_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")))
        except WebSocketDisconnect:
            pass
        finally:
            bus.unsubscribe(on_update)

    @app.get("/media/game.jpg")
    def game_frame() -> Response:
        frame = runtime.capture.latest()
        if not frame:
            raise HTTPException(status_code=404, detail="No game frame captured yet")
        return Response(content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    if DIST.exists():
        app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def frontend(path: str) -> FileResponse:
            candidate = DIST / path
            if path and candidate.exists() and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(DIST / "index.html")

    return app
