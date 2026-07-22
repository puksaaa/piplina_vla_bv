#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo ".venv is missing. Run ./scripts/install_camera_splitter.sh first."
  exit 1
fi

for env_file in .env.vlm .env.smolvla .env.orchestrator; do
  if [ ! -f "$env_file" ]; then
    cp "${env_file}.example" "$env_file"
    echo "Created $env_file. Configure it, then run this command again."
    exit 1
  fi
done

source .venv/bin/activate
exec python orchestrator.py
