from fastapi import FastAPI, WebSocket
import uvicorn

from backend.shared_state import shared_state

app = FastAPI()

@app.get("/")
async def health():
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    await websocket.accept()

    print("Client connected.")

    while True:

        # Receive prompt from frontend
        prompt = await websocket.receive_text()

        print("Received prompt:", prompt)

        # Update shared state
        shared_state.user_prompt = prompt

        # Acknowledge
        await websocket.send_text(
            f"Prompt updated: {prompt}"
        )