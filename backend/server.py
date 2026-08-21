from fastapi import FastAPI, WebSocket
import uvicorn
from fastapi.responses import StreamingResponse
import io
import asyncio

from backend.shared_state import shared_state
from backend.utils.input_processor import preprocess_prompt, InvalidPromptError

app = FastAPI()

@app.get("/")
async def health():
    return {"status": "ok"}

@app.get("/camera")
async def camera_stream():
    async def generate():
        while True:
            if shared_state.camera and shared_state.camera.is_ready():
                frame_bytes = shared_state.camera.get_frame_bytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame_bytes +
                    b"\r\n"
                )
            await asyncio.sleep(0.05)  # ~20fps

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    print("Client connected.")

    while True:

        # Receive prompt from frontend
        prompt = await websocket.receive_text()

        print("Received prompt:", prompt)

        # Pre-process and validate
        try:
            prompt = preprocess_prompt(prompt)
        except InvalidPromptError as e:
            await websocket.send_text(f"Invalid prompt: {e}")
            continue

        # Update shared state
        shared_state.user_prompt = prompt

        # Acknowledge
        await websocket.send_text(
            f"Prompt updated: {shared_state.latest_prompt}"
        )