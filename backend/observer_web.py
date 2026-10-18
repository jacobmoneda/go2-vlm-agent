"""Camera-only target-following evaluation server. NO MOTOR COMMANDS."""
import asyncio
import json
import os
import threading
import time
from datetime import datetime

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from backend.camera.go2_camera import Go2Camera
from backend.concurrent_vision import ConcurrentVision
from backend.target_observer import parse_target

app = FastAPI()
lock = threading.RLock()
state = {"camera": None, "pipeline": None, "command": None, "started": None,
         "status": "idle", "latest": None, "events": [], "generation": 0}

def log(message):
    line = "[{}] {}".format(datetime.now().strftime("%H:%M:%S"), message)
    print(line, flush=True)
    with lock:
        state["events"].append(line)
        state["events"] = state["events"][-250:]

def stop_tracking():
    with lock:
        started = state["started"]
        state["generation"] += 1
        state["pipeline"] = None
        state["command"] = None
        state["started"] = None
        state["status"] = "stopped"
    if started is not None:
        log("Total Time: {:.2f}s".format(time.monotonic() - started))
    log("Movement: DISABLED; tracking stopped")

def track_loop(generation):
    previous_position = None
    previous_decision = None
    previous_vlm = None
    while True:
        with lock:
            if generation != state["generation"]:
                return
            pipeline = state["pipeline"]
            camera = state["camera"]
            started = state["started"]
        frame = camera.get_frame() if camera else None
        if frame is None:
            time.sleep(0.2)
            continue
        try:
            result = pipeline.process(frame)
            result["elapsed_seconds"] = round(time.monotonic() - started, 2)
            result["movement_executed"] = False
            if result["visible"]:
                position = result["observation"].replace("target ", "").upper()
                if position != previous_position:
                    log("Position: " + position)
                    previous_position = position
                box_size = result.get("relative_box_height_percent")
                if box_size is not None:
                    log("Apparent size: {:.1f}% of frame height (not distance)".format(box_size))
                decision = result["movement_decision"].upper()
                if decision != previous_decision:
                    log("Decision: " + decision)
                    log("Movement: SIMULATED")
                    previous_decision = decision
                log("Centring accuracy: {}%".format(result["centring_accuracy_percent"]))
            else:
                if previous_position != "LOST":
                    log("YOLO: target lost; decision: HOLD")
                    previous_position = "LOST"
            description = result.get("vlm_description")
            if description and description != previous_vlm:
                log("VLM: " + str(description)[:400])
                previous_vlm = description
            with lock:
                if generation != state["generation"]:
                    return
                state["latest"] = result
        except Exception as exc:
            log("Observer error: {}: {}".format(type(exc).__name__, exc))
        time.sleep(0.25)

@app.on_event("startup")
def startup():
    camera = Go2Camera(os.environ.get("GO2_INTERFACE", "eth0"))
    camera.start()
    with lock:
        state["camera"] = camera
    log("Camera starting; physical movement disabled")

@app.on_event("shutdown")
def shutdown():
    stop_tracking()
    with lock:
        camera = state["camera"]
    if camera:
        camera.stop()

@app.get("/health")
def health():
    with lock:
        return {"camera_ready": bool(state["camera"] and state["camera"].is_ready()),
                "status": state["status"], "movement_enabled": False}

@app.get("/camera")
async def camera():
    async def frames():
        while True:
            with lock:
                cam = state["camera"]
            data = cam.get_frame_bytes() if cam else None
            if data:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            await asyncio.sleep(0.1)
    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cursor = 0
    async def sender():
        nonlocal cursor
        while True:
            with lock:
                events = list(state["events"])
                latest = state["latest"]
                started = state["started"]
            if cursor > len(events):
                cursor = 0
            for event in events[cursor:]:
                await ws.send_text(event)
            cursor = len(events)
            await ws.send_text("__METRICS__" + json.dumps({"latest": latest,
                "elapsed_seconds": round(time.monotonic()-started,2) if started else None}))
            await asyncio.sleep(0.5)
    task = asyncio.create_task(sender())
    try:
        while True:
            command = (await ws.receive_text()).strip()
            if command == "__EMERGENCY_STOP__" or command.lower() == "stop":
                stop_tracking()
                continue
            try:
                parse_target(command)
            except Exception as exc:
                log("Unsupported tracking command: {}".format(exc))
                continue
            stop_tracking()
            with lock:
                state["events"].clear()
                state["latest"] = None
                state["command"] = command
                state["started"] = time.monotonic()
                state["status"] = "tracking"
                state["generation"] += 1
                generation = state["generation"]
            log("Command: " + command)
            try:
                pipeline = ConcurrentVision(command, enable_vlm=os.environ.get("GO2_ENABLE_QWEN", "0") == "1")
            except Exception as exc:
                log("Failed to initialize vision: {}".format(exc))
                stop_tracking()
                continue
            with lock:
                if generation != state["generation"]:
                    continue
                state["pipeline"] = pipeline
            threading.Thread(target=track_loop, args=(generation,), daemon=True).start()
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
