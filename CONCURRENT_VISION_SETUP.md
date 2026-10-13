# Concurrent Qwen + YOLO observation (no robot movement)

Run from repository root with the Go2 SDK, ultralytics, OpenCV, Pillow, PyTorch, transformers, accelerate, and qwen-vl-utils installed:

```bash
python -m backend.concurrent_vision --interface eth0 --command 'track the blue bottle'
```

For tracking-only testing (no model download):

```bash
python -m backend.concurrent_vision --interface eth0 --command 'track the blue bottle' --no-vlm
```

The fast loop uses YOLO tracking and colour heuristics, while a separate background thread runs Qwen occasionally. Outputs are JSON records with target location, tracking confidence, centring accuracy, VLM observations and **movement_executed: false**. No movement is performed. The VLM description is informational and does not select a bounding box or confirm identity. This is not a navigation solution.

Accuracy is centred visible frames / all visible target frames. Target loss is excluded; evaluate detection recall and lost-frame rate separately. The simple colour threshold and track IDs can fail when objects overlap or lighting changes. Qwen 3B may be too large or slow for the onboard processor; run on a compatible companion computer if necessary. Never treat image bounding-box size as calibrated distance.

For physical movement, independently validate obstacle clearance, distance measurement, stale-frame handling, stop behaviour, speed limits, and manufacturer safety procedures with qualified supervision.
