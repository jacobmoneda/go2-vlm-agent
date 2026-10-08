#!/usr/bin/env bash

echo "Starting Ollama in GPU mode..."

export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=-1

ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!

echo "Ollama PID: $OLLAMA_PID"
echo "Waiting for Ollama..."

# Wait until Ollama API is actually ready
for i in {1..60}; do
    if curl -sf http://127.0.0.1:11434/api/tags > /dev/null; then
        echo "Ollama is ready."
        break
    fi

    sleep 1
done

echo "Pre-loading phi3-fast..."

curl -fsS \
    --max-time 300 \
    http://127.0.0.1:11434/api/generate \
    -d '{
        "model": "phi3-fast",
        "prompt": "",
        "stream": false,
        "keep_alive": -1
    }'

echo
echo "phi3-fast loaded and kept in memory."
echo "Ollama is running on PID $OLLAMA_PID"
echo "READY :-)"

# Keep this script/terminal alive
wait "$OLLAMA_PID"