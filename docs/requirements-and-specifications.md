# Product Requirements and Specifications

Go2 VLM Agent provides browser-based natural-language control, live camera viewing, object tracking/following, and scene understanding for a Unitree Go2.

## Hardware and platform

| Component | Specification |
| --- | --- |
| Robot | Unitree Go2 EDU |
| Onboard computer | NVIDIA Jetson Orin NX; RAM/storage capacity not recorded |
| Camera | Go2 front-facing camera through Unitree VideoClient |
| Operating system | Ubuntu 20.04.5 LTS, aarch64 |
| Kernel | 5.10.104-tegra |
| Jetson Linux / L4T | R35.3.1 |
| Operator computer | Ethernet connection, SSH client, Node.js/npm, and a WebSocket/MJPEG-capable browser |

## Software versions and dependencies

Robot versions below come from [the recorded environment](../dog-environment.txt); they are recorded versions, not verified minimum requirements. Frontend versions come from its lockfile.

| Component | Version | Purpose |
| --- | --- | --- |
| Python | 3.8.10 | Backend runtime |
| pip | 25.0.1 | Python package installation |
| CUDA toolkit | 11.4 (V11.4.315) | GPU support for Phi vision |
| PyTorch | 2.0.0a0+8aa34602.nv23.03 | NVIDIA build with CUDA 11.4 |
| Ultralytics | 8.3.228 | YOLO detection and persistent tracking |
| unitree_sdk2py | 1.0.1 | Robot movement and camera access |
| Ollama | 0.3.14 client; server version not recorded | Local command-routing model service |
| React / React DOM | 19.2.6 | Browser interface |
| Vite | 8.0.13 | Frontend development/build tooling |
| ESLint | 10.4.0 | Frontend lint tooling |
| Node.js | Must satisfy Vite and ESLint engine constraints; installed version not recorded | Frontend tooling |

The lockfile specifies Node.js `^20.19.0 || >=22.12.0` for Vite and `^20.19.0 || ^22.13.0 || >=24` for ESLint. Use a version satisfying both.

Additional backend dependencies are **FastAPI, Uvicorn, Requests, Pillow, NumPy, OpenCV (`opencv-python`), Transformers, and CycloneDDS**. Their installed versions are not recorded. Frontend development also uses the React Vite plugin, Babel, React Compiler, React type definitions, and ESLint plugins; exact dependencies are captured in [package.json](../frontend/package.json) and [package-lock.json](../frontend/package-lock.json).

No backend dependency lockfile is present. Compatibility must be maintained with the Jetson Python/CUDA/PyTorch environment.

## AI models

| Model | Role | Runtime / configuration |
| --- | --- | --- |
| YOLO11n (`yolo11n.pt`) | Object detection and tracking | CPU; lazy-loaded; tracking uses `persist=True`, confidence 0.1, installed default tracker |
| Phi-3.5-vision-instruct | Camera-based scene understanding | Transformers; CUDA FP16; eager attention; one image crop; maximum 200 generated tokens |
| `phi3-fast` | Natural-language command routing through Ollama | Temperature 0; context 2048; maximum 90 predicted tokens; exact model revision/quantization not recorded |

Required model locations:

- `/home/unitree/models/yolo11n.pt`
- `/home/unitree/models/Phi-3.5-vision-instruct/`
- Ollama model registered as `phi3-fast`

Checkpoint revisions/hashes are not recorded. Palm detection, Qwen, and SmolVLM models belong to experimental utilities and are not main-product requirements.

## Product capabilities and operating specifications

| Area | Specification |
| --- | --- |
| Commands | Direct movement, posture, gestures, speed selection, object following/approach, and visual questions |
| Input | Text commands up to 200 characters |
| Interface | React UI with command input, status messages, live camera, and software STOP button |
| Backend interfaces | FastAPI HTTP health endpoint, MJPEG camera stream, and text WebSocket commands/status |
| Camera acquisition | Target 10 FPS; control assumes 1920×1080 input |
| Follow selection | Matching class, confidence ≥0.5, assigned track ID; largest box used only for initial selection |
| Persistent following | Retains selected ID; stops when missing; latches stop after 100 consecutive processed missing frames until reset |
| Distance control | Bounding-box height proxy: forward below 540 px, backward above 660 px when centred |
| Turning | Horizontal centre 960 px, dead zone ±60 px |
| Deployment | One robot, one backend process, one shared active command; primarily one operator |

## Network and runtime requirements

- Ethernet access to the Go2; documented robot address `192.168.123.18`.
- Unitree DDS domain 0 and network interface `eth0` in the main application.
- Backend on port **8000**, Vite frontend normally on **5173**, and local Ollama service on **11434**.
- WebSocket defaults to `ws://192.168.123.18:8000/ws`; configurable through `VITE_WS_URL`. Camera URL is fixed to the robot address in the UI.
- Installed model files, compatible SDK/CUDA libraries, and writable `/home/unitree/go2-vlm-agent/images/` directory.
- Supervised operation with an independent stop available. The UI stop is a software request.

Setup and launch instructions: [README](../README.md) and [Go2 setup guide](go2-setup.md).
