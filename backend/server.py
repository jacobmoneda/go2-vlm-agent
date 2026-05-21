from fastapi import FastAPI, WebSocket
import uvicorn
from qwen_engine import run_qwen

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected.")

    while True:
        #receive prompt from frontend
        prompt = await websocket.receive_text()

        print("Received prompt:", prompt)

        # Placeholder AI response
        #response = f"Robot received: {prompt}"
        # run qwen engine
        response = run_qwen("view.jpg", prompt)

        # send response back to frontend
        await websocket.send_text(response)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)