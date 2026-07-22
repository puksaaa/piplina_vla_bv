#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f .env.camera ]; then
  cp .env.camera.example .env.camera
  echo "Created .env.camera from .env.camera.example"
fi

echo "Camera splitter dependencies are installed."
echo "Edit .env.camera if camera indexes are not 0 and 1."
echo "Then run: ./scripts/start_camera_splitter.sh"
