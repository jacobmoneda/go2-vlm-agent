import { useEffect, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://192.168.123.18:8000/ws";

function App() {
  const [socket, setSocket] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("Connected to server");
    };

    ws.onmessage = (event) => {
      setResponse(event.data);
    };

    setSocket(ws);

    return () => ws.close();
  }, []);

  const sendPrompt = () => {
    if (socket && prompt) {
      socket.send(prompt);
    }
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>Robot Dog Controller</h1>

      <input
        type="text"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter command..."
        style={{ width: "300px", marginRight: "10px" }}
      />

      <button onClick={sendPrompt}>
        Send
      </button>

      <h2>Robot Response:</h2>
      <p>{response}</p>
    </div>
  );
}

export default App;