#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "$#" -eq 0 ]; then
  echo "Usage: ./scripts/start_pipeline.sh \"move red cube to white bowl\" [supervisor options]"
  exit 2
fi

if [ ! -d .venv ]; then
  echo ".venv is missing. Run ./scripts/install_camera_splitter.sh first."
  exit 1
fi

for env_file in .env.camera .env.vlm .env.smolvla; do
  if [ ! -f "$env_file" ]; then
    echo "$env_file is missing. Copy its .example file and configure it before launch."
    exit 1
  fi
done

mkdir -p logs

wait_for_http() {
  local url="$1"
  local name="$2"
  for _ in $(seq 1 45); do
    if curl --silent --fail "$url" >/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "$name did not become reachable: $url"
  return 1
}

if ! curl --silent --fail http://127.0.0.1:8090/health >/dev/null; then
  nohup ./scripts/start_camera_splitter.sh >logs/camera-splitter.log 2>&1 &
fi
wait_for_http http://127.0.0.1:8090/health "Camera splitter"

source .venv/bin/activate
python verify_camera_splitter.py --base-url http://127.0.0.1:8090 --zmq-address tcp://127.0.0.1:5555 --cameras "$(grep '^SMOLVLA_CAMERA_NAMES=' .env.smolvla | cut -d= -f2)"

if ! curl --silent --fail http://127.0.0.1:8091/health >/dev/null; then
  nohup ./scripts/start_smolvla_runner.sh >logs/smolvla-runner.log 2>&1 &
fi
wait_for_http http://127.0.0.1:8091/health "SmolVLA runner"
python verify_smolvla_runner.py
python verify_gemma.py

exec ./scripts/start_supervisor.sh "$@"
