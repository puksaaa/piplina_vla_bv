#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
WEB_APP_DIR="${WEB_APP_DIR:-laptop_app}"

if [ ! -d .venv ]; then
  echo ".venv is missing. Run ./scripts/install_camera_splitter.sh first."
  exit 1
fi

for env_file in .env.camera .env.vlm .env.smolvla .env.orchestrator "$WEB_APP_DIR/.env"; do
  if [ ! -f "$env_file" ]; then
    echo "$env_file is missing. Copy its .example file and configure it first."
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

if ! curl --silent --fail http://127.0.0.1:8090/snapshot/all >/dev/null; then
  nohup ./scripts/start_camera_splitter.sh >logs/camera-splitter.log 2>&1 &
fi
wait_for_http http://127.0.0.1:8090/snapshot/all "Camera splitter"

if ! curl --silent --fail http://127.0.0.1:8091/health >/dev/null; then
  nohup ./scripts/start_smolvla_runner.sh >logs/smolvla-runner.log 2>&1 &
fi
wait_for_http http://127.0.0.1:8091/health "SmolVLA runner"

if ! curl --silent --fail http://127.0.0.1:8092/health >/dev/null; then
  nohup ./scripts/start_orchestrator.sh >logs/orchestrator.log 2>&1 &
fi
wait_for_http http://127.0.0.1:8092/health "Orchestrator"

source .venv/bin/activate
export ORCHESTRATOR_ENABLED=1
export ORCHESTRATOR_URL=http://127.0.0.1:8092
echo "Open http://100.64.0.1:8000"
exec python "$WEB_APP_DIR/main.py"
