import { useEffect, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://192.168.123.18:8000/ws";

function App() {
  const [socket, setSocket] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);

  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("Connected to server");
    };

    ws.onmessage = (event) => {
      setMessages((prevMessages) => [
        ...prevMessages, 
        event.data
      ]);
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

      <h2>Robot Status:</h2>

    <div
      style={{
        backgroundColor: "#111",
        color: "#eee",
        padding: "15px",
        borderRadius: "8px",
        height: "200px",
        overflowY: "auto",
        fontFamily: "monospace",
        marginBottom: "20px"
      }}
    >
      {messages.length === 0 ? (
        <div>Waiting for command...</div>
      ) : (
        messages.map((message, index) => (
          <div key={index}>
            {message}
          </div>
        ))
      )}
    </div>

      <img 
      src="http://192.168.123.18:8000/camera" 
      alt="Robot Camera Feed"
      style={{ width: "100%", borderRadius: "8px" }}
      />
    </div>
  );
}

export default App;