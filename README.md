# go2-vlm-agent

Vision-Based Autonomous Behaviour Learning for the Unitree Go2

## Deployment Instructions

1. Ensure that you are connected to the Unitree Go2 robot dog. See [docs/go2-setup.md](https://github.com/jacobmoneda/go2-vlm-agent/blob/main/docs/go2-setup.md)

2. Open an ssh terminal

```bash
OLLAMA_LLM_LIBRARY=cpu OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=-1 ollama serve
```

3. Open a second ssh terminal
   
```bash
cd ~/go2-vlm-agent
python3 -m backend.main
```

4. open a local PC terminal

```bash
cd ~/go2-vlm-agent/frontend
npx vite --mode robot
```

5. Open the website url (http://localhost:5173)
