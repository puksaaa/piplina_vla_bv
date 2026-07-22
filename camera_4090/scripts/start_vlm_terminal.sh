#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo ".venv is missing. Run ./scripts/install_camera_splitter.sh first."
  exit 1
fi

if [ ! -f .env.vlm ]; then
  cp .env.vlm.example .env.vlm
  echo "Created .env.vlm from .env.vlm.example"
fi

source .venv/bin/activate
exec python vlm_terminal.py "$@"
