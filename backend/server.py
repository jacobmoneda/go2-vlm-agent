from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()

# Track the connected robot client
robot_connection: WebSocket = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global robot_connection
    await websocket.accept()
    print("A client connected.")

    try:
        while True:
            # 1. Receive prompt from your frontend interface
            prompt = await websocket.receive_text()
            print("Received prompt from frontend:", prompt)

            # 2. Check if a robot has registered itself as the endpoint listener
            # If the robot is the one sending the prompt (e.g. unified endpoint), 
            # save it as the active connection.
            if "robot" in prompt.lower():
                robot_connection = websocket
                print("Robot registered via handshake.")
                continue

            # 3. Forward the prompt down to the robot if available
            if robot_connection:
                await robot_connection.send_text(prompt)
                print("Prompt forwarded to robot client.")
                
            # Send immediate visual receipt acknowledgment back to frontend
            await websocket.send_text(f"Backend broadcasted prompt: {prompt}")

    except WebSocketDisconnect:
        if websocket == robot_connection:
            robot_connection = None
            print("Robot client disconnected.")
        else:
            print("Frontend client disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
