#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo ".venv is missing. Run ./scripts/install_camera_splitter.sh first."
  exit 1
fi

if [ -f .env.camera ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.camera
  set +a
fi

source .venv/bin/activate

ARGS=(
  --front-device "${CAMERA_FRONT_DEVICE:-0}"
  --wrist-device "${CAMERA_WRIST_DEVICE:-1}"
  --front-name "${CAMERA_FRONT_NAME:-front}"
  --wrist-name "${CAMERA_WRIST_NAME:-wrist}"
  --width "${CAMERA_WIDTH:-640}"
  --height "${CAMERA_HEIGHT:-480}"
  --fps "${CAMERA_FPS:-15}"
  --jpeg-quality "${CAMERA_JPEG_QUALITY:-82}"
  --http-host "${CAMERA_HTTP_HOST:-0.0.0.0}"
  --http-port "${CAMERA_HTTP_PORT:-8090}"
  --zmq-bind "${CAMERA_ZMQ_BIND:-tcp://0.0.0.0:5555}"
  --zmq-fps "${CAMERA_ZMQ_FPS:-15}"
)

if [ "${CAMERA_MOCK:-0}" = "1" ]; then
  ARGS+=(--mock)
fi

if [ "${CAMERA_ENABLE_ZMQ:-1}" = "1" ]; then
  ARGS+=(--enable-zmq)
fi

exec python camera_splitter.py "${ARGS[@]}"
