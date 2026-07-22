#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo ".venv is missing. Run ./scripts/install_camera_splitter.sh first."
  exit 1
fi

if [ ! -f .env.camera ]; then
  cp .env.camera.example .env.camera
  echo "Created .env.camera. Set the two USB camera indexes, then run this command again."
  exit 1
fi

if [ ! -f web_app/.env ]; then
  cp web_app/.env.example web_app/.env
  echo "Created web_app/.env from its example. Check the Gemma endpoint, then run this command again."
  exit 1
fi

mkdir -p logs

if ! curl --silent --fail http://127.0.0.1:8090/snapshot/all >/dev/null; then
  nohup ./scripts/start_camera_splitter.sh >logs/camera-splitter.log 2>&1 &
fi

for _ in $(seq 1 30); do
  if curl --silent --fail http://127.0.0.1:8090/snapshot/all >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl --silent --fail http://127.0.0.1:8090/snapshot/all >/dev/null; then
  echo "Camera splitter did not produce frames. Read logs/camera-splitter.log and check .env.camera indexes."
  exit 1
fi

source .venv/bin/activate
echo "Open http://100.64.0.1:8000 in the browser."
echo "Camera preview: http://100.64.0.1:8090/preview"
exec python web_app/main.py
