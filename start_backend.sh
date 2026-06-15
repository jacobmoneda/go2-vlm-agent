#!/bin/bash
# Run on the robot dog via SSH
cd ~/go2-vlm-agent
python3 -m uvicorn backend.server:app --host 0.0.0.0 --port 8000
