#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo ".venv is missing. Run ./scripts/install_camera_splitter.sh first."
  exit 1
fi

if [ ! -f .env.camera ]; then
  cp .env.camera.example .env.camera
  echo "Created .env.camera. Set USB camera indexes, then run again."
  exit 1
fi

if [ ! -f .env.lerobot_rollout ]; then
  cp .env.lerobot_rollout.example .env.lerobot_rollout
  echo "Created .env.lerobot_rollout. Set robot, policy, and camera keys, then run again."
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env.camera
# shellcheck disable=SC1091
source .env.lerobot_rollout
set +a

mkdir -p logs

if ! curl --silent --fail "http://127.0.0.1:${CAMERA_HTTP_PORT:-8090}/snapshot/all" >/dev/null; then
  nohup ./scripts/start_camera_splitter.sh >logs/camera-splitter.log 2>&1 &
fi

for _ in $(seq 1 30); do
  if curl --silent --fail "http://127.0.0.1:${CAMERA_HTTP_PORT:-8090}/snapshot/all" >/dev/null; then
    break
  fi
  sleep 1
done

if ! curl --silent --fail "http://127.0.0.1:${CAMERA_HTTP_PORT:-8090}/snapshot/all" >/dev/null; then
  echo "Camera splitter did not produce frames. Read logs/camera-splitter.log and check .env.camera indexes."
  exit 1
fi

TASK="${1:-${LEROBOT_DEFAULT_TASK:-}}"
if [ -z "$TASK" ]; then
  echo "Usage: ./scripts/start_lerobot_rollout_zmq.sh \"task for the policy\""
  exit 2
fi

camera_front="${LEROBOT_CAMERA_FRONT_KEY:-front}: {type: zmq, server_address: '${LEROBOT_ZMQ_SERVER_ADDRESS:-127.0.0.1}', port: ${LEROBOT_ZMQ_PORT:-5555}, camera_name: '${LEROBOT_CAMERA_FRONT_NAME:-camera1}', width: ${LEROBOT_CAMERA_WIDTH:-640}, height: ${LEROBOT_CAMERA_HEIGHT:-480}, fps: ${LEROBOT_CAMERA_FPS:-15}}"
camera_config="{${camera_front}}"

if [ "${LEROBOT_CAMERA_WRIST_ENABLED:-0}" = "1" ]; then
  camera_wrist="${LEROBOT_CAMERA_WRIST_KEY:-wrist}: {type: zmq, server_address: '${LEROBOT_ZMQ_SERVER_ADDRESS:-127.0.0.1}', port: ${LEROBOT_ZMQ_PORT:-5555}, camera_name: '${LEROBOT_CAMERA_WRIST_NAME:-camera2}', width: ${LEROBOT_CAMERA_WIDTH:-640}, height: ${LEROBOT_CAMERA_HEIGHT:-480}, fps: ${LEROBOT_CAMERA_FPS:-15}}"
  camera_config="{${camera_front}, ${camera_wrist}}"
fi

args=(
  "--strategy.type=${LEROBOT_STRATEGY_TYPE:-base}"
  "--robot.type=${LEROBOT_ROBOT_TYPE:-so101_follower}"
  "--robot.port=${LEROBOT_ROBOT_PORT:-/dev/ttyACM0}"
  "--robot.id=${LEROBOT_ROBOT_ID:-my_follower_arm}"
  "--robot.cameras=${camera_config}"
  "--policy.path=${LEROBOT_POLICY_PATH:-outputs/train/my_smolvla/checkpoints/last/pretrained_model}"
  "--policy.device=${LEROBOT_POLICY_DEVICE:-cuda}"
  "--task=${TASK}"
  "--duration=${LEROBOT_DURATION:-60}"
  "--display_data=${LEROBOT_DISPLAY_DATA:-true}"
)

if [ "${LEROBOT_INFERENCE_TYPE:-sync}" != "sync" ]; then
  args+=(
    "--inference.type=${LEROBOT_INFERENCE_TYPE}"
    "--inference.rtc.execution_horizon=${LEROBOT_RTC_EXECUTION_HORIZON:-10}"
    "--inference.rtc.max_guidance_weight=${LEROBOT_RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
  )
fi

echo "Camera splitter: http://127.0.0.1:${CAMERA_HTTP_PORT:-8090}/preview"
echo "LeRobot cameras: ${camera_config}"
echo "Task: ${TASK}"
exec "${LEROBOT_ROLLOUT_BIN:-lerobot-rollout}" "${args[@]}"
