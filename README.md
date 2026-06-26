# go2-vlm-agent

Vision-Based Autonomous Behaviour Learning for the Unitree Go2

## Deployment Instructions

1. Ensure that you are connected to the Unitree Go2 robot dog. See [docs/go2-setup.md](https://github.com/jacobmoneda/go2-vlm-agent/blob/main/docs/go2-setup.md)
2. In the ssh terminal

```bash
cd ~/go2-vlm-agent
python3 -m backend.main
```

3. In your local PC terminal

```bash
npx vite --mode robot
```

4. Open the website url (http://localhost:5173)
