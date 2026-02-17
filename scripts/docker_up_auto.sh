#!/usr/bin/env bash
set -euo pipefail

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "GPU detected. Trying GPU service first..."
  if docker compose --profile gpu up --build rag-app-gpu; then
    exit 0
  fi
  echo "GPU service failed. Falling back to CPU service."
fi

echo "Starting CPU service..."
docker compose up --build rag-app-cpu
