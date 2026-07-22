#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env.smolvla ]; then
  cp .env.smolvla.example .env.smolvla
  echo "Created .env.smolvla from .env.smolvla.example"
fi

set -a
source .env.smolvla
set +a

if [ -z "${SMOLVLA_POLICY_PATH:-}" ] || [ "${SMOLVLA_POLICY_PATH:-}" = "/absolute/path/to/checkpoint/pretrained_model" ]; then
  echo "Set SMOLVLA_POLICY_PATH in .env.smolvla before starting the runner."
  exit 1
fi

if [[ "$SMOLVLA_POLICY_PATH" == /* || "$SMOLVLA_POLICY_PATH" == ./* || "$SMOLVLA_POLICY_PATH" == ../* ]]; then
  if [ ! -d "$SMOLVLA_POLICY_PATH" ]; then
    echo "SMOLVLA_POLICY_PATH directory does not exist: $SMOLVLA_POLICY_PATH"
    exit 1
  fi
  if [ ! -f "$SMOLVLA_POLICY_PATH/config.json" ]; then
    echo "Checkpoint has no config.json: $SMOLVLA_POLICY_PATH"
    echo "Point SMOLVLA_POLICY_PATH at the pretrained_model checkpoint directory."
    exit 1
  fi
else
  echo "Using SMOLVLA_POLICY_PATH as a Hugging Face model ID: $SMOLVLA_POLICY_PATH"
fi

if [ -n "${SMOLVLA_PYTHON:-}" ] && [ "${SMOLVLA_PYTHON}" != "python" ]; then
  PYTHON_BIN="$SMOLVLA_PYTHON"
else
  PYTHON_BIN="$(pwd)/.venv/bin/python"
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python for SmolVLA is not executable: $PYTHON_BIN"
  echo "Set SMOLVLA_PYTHON to the Python executable in the environment with LeRobot and your fine-tuned SmolVLA dependencies."
  exit 1
fi

echo "SmolVLA checkpoint: $SMOLVLA_POLICY_PATH"
echo "SmolVLA Python: $PYTHON_BIN"
exec "$PYTHON_BIN" smolvla_runner.py
