# go2-vlm-agent

**Vision-Based Autonomous Behaviour Learning for the Unitree Go2**

`go2-vlm-agent` is an experimental robotics project that combines natural-language command routing, vision-language processing, object detection, and robot control to enable a Unitree Go2 to respond to user commands and perform autonomous behaviours.

The system uses:

- **Phi-3 via Ollama** for natural-language command routing
- **Vision-language processing** for scene understanding
- **YOLO11n** for real-time object detection and target following
- **Unitree SDK2** for robot movement and action execution
- **FastAPI / Uvicorn** for the backend service
- **React / Vite** for the user interface

---

## System Overview

**_to be done_**

The command router determines whether a user request can be executed directly or requires visual perception.

Examples:

```text
"Sit down"            -> Direct robot action
"Wave"                -> Direct robot action
"Detect a person"     -> Vision-based processing
"Follow the person"   -> YOLO detection and follow control
"Stop"                -> Direct robot action
```

---

## Hardware

This project is designed for:

- **Unitree Go2 EDU**
- **NVIDIA Jetson Orin NX**
- Go2 front-facing camera
- Local network connection between the robot and development PC

---

## Deployment Instructions

### 1. Connect to the Unitree Go2

Ensure your development PC is connected to the Unitree Go2 network and that the robot can be accessed over SSH.

For initial setup instructions, see:

[`docs/go2-setup.md`](docs/go2-setup.md)

---

### 2. Start Ollama on the Go2

Open an SSH terminal to the robot and start Ollama in CPU mode:

```bash
OLLAMA_LLM_LIBRARY=cpu \
OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_KEEP_ALIVE=-1 \
ollama serve
```

Keep this terminal running.

> Ollama is currently started in CPU mode to avoid CUDA-runner issues on the Jetson platform.

---

### 3. Start the Robot Backend

Open a **second SSH terminal** to the Go2:

```bash
cd ~/go2-vlm-agent
python3 -m backend.main
```

The backend handles:

- prompt preprocessing
- command routing
- camera access
- vision-language processing
- YOLO object detection
- decision logic
- robot action execution

Keep this terminal running. This is where console debug logs will be printed.

---

### 4. Start the Frontend

On your **local development PC**, open a terminal:

```bash
cd ~/go2-vlm-agent/frontend
npx vite --mode robot
```

---

### 5. Open the Web Interface

Open the following address in your browser:

[http://localhost:5173](http://localhost:5173)

You can now send natural-language commands to the Unitree Go2 through the web interface.

---

## Required Terminals

During normal deployment, three terminals are used:

| Terminal | Device         | Purpose                          |
| -------- | -------------- | -------------------------------- |
| 1        | Unitree Go2    | Ollama LLM server                |
| 2        | Unitree Go2    | Python backend and robot control |
| 3        | Development PC | Vite frontend                    |

---

## Target Following

YOLO11n is used for real-time target detection and following.

For a detected person, the system can:

1. Detect the person in the camera frame.
2. Calculate the horizontal offset from the image centre.
3. Turn left or right to centre the target.
4. Move forward if the target is too far away.
5. Move backward if the target is too close.
6. Stop when the target is centred and within the desired distance.

The current following system uses bounding-box position and size as visual control inputs.

---

## Project Structure

```text
go2-vlm-agent/
├── backend/
│   ├── camera/
│   │   └── go2_camera.py
│   ├── objectDetection/
│   │   └── yolo_engine.py
│   ├── robotControl/
│   │   └── robot_control.py
│   ├── utils/
│   │   ├── command_router.py
│   │   └── input_processor.py
│   ├── vlm/
│   │   └── phi_engine.py
│   ├── decision_logic.py
│   ├── shared_state.py
│   ├── server.py
│   └── main.py
├── frontend/
├── docs/
├── testing/
└── README.md
```

---

## Technologies

### Robotics

- Unitree Go2 EDU
- Unitree SDK2

### AI and Computer Vision

- Phi-3
- Ollama
- YOLO11n
- Ultralytics
- PyTorch
- Pillow
- NumPy

### Backend

- Python
- FastAPI
- Uvicorn

### Frontend

- React
- Vite

---

## Development and Testing

The project has been evaluated across several areas, including:

- YOLO inference time per frame
- Ollama command-routing latency
- end-to-end command latency
- target-centering accuracy
- target-following behaviour
- invalid and unsupported command handling
- VLM response latency
- cold-start versus warm-model performance
- robot action execution
- camera-stream reliability

These tests are used to evaluate both real-time performance and system reliability on the Jetson-based robot platform.

---

## Environment and Dependencies

To record the Python dependencies used on the robot:

```bash
pip3 freeze > requirements.txt
```

Useful system information can be recorded with:

```bash
python3 --version
pip3 --version
ollama --version
cat /etc/nv_tegra_release
nvcc --version
```

For the Unitree SDK:

```bash
pip3 show unitree_sdk2py
```

If the SDK was installed from source, record its Git commit:

```bash
git rev-parse HEAD
```

---

## Notes

- Ollama should currently be started in **CPU mode** on the Jetson.
- YOLO11n is also currently run on the CPU.
- The system is under active development and some behaviours may still require tuning.
- Always test movement commands in a safe, open area with an emergency stop available.

---

## Documentation

Additional setup documentation is available in:

```text
docs/go2-setup.md
```

---

## Authors

Developed as part of a research and development project exploring vision-based autonomous behaviour learning for the Unitree Go2.
