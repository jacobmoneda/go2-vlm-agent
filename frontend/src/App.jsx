import { useEffect, useState } from "react";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://192.168.123.18:8000/ws";

function App() {
  const [socket, setSocket] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [messages, setMessages] = useState([]);
  const [followDecision, setFollowDecision] = useState("");

  useEffect(() => {
    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      console.log("Connected to server");
    };

    ws.onmessage = (event) => {
      const message = event.data;

      if (message.startsWith("[Follow Decision]")) {
          setFollowDecision(message);
      } else {
          setMessages((prevMessages) => [
              ...prevMessages,
              message
          ]);
      }
  };

    setSocket(ws);

    return () => ws.close();
  }, []);

  const sendPrompt = () => {
    if (socket && prompt) {
      setMessages([]);   // clear previous command/results
      setFollowDecision(""); 
      socket.send(prompt);
    }
  };

  const emergencyStop = () => {
    if (socket) {
      socket.send("__EMERGENCY_STOP__");
      setFollowDecision("");
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

      <button
        onClick={emergencyStop}
        style={{
          backgroundColor: "red",
          color: "white",
          fontWeight: "bold",
          fontSize: "16px",
          padding: "8px 20px",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
          marginLeft: "10px"
        }}
      >
        STOP
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
      {messages.length === 0 && !followDecision ? (
        <div>Waiting for command...</div>
      ) : (
        <>
          {messages.map((message, index) => (
            <div key={index}>
              {message}
            </div>
          ))}

          {followDecision && (
            <div>
              {followDecision}
            </div>
          )}
        </>
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
