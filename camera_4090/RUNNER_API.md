# Resident SmolVLA Inference API

`smolvla_runner.py` is a long-lived inference service. It loads the checkpoint
once, keeps it in CUDA memory, reads ZMQ camera frames, and changes the current
short language task without restarting the process.

Actuation is opt-in via `SMOLVLA_ACTUATION_ENABLED`. When off (default) the
runner is inference-only: it never opens a serial port and expects robot state at
`POST /v1/state`. When on, the runner owns the SO-101 serial port through
`robot_actuator.RobotActuator`, reads joint state itself, and sends each policy
action to the motors. `GET /health` reports the active `execution_mode`
(`inference_only`, `robot_state_readonly`, or `actuation`).

## Lifecycle

1. Start the camera splitter.
2. Start `./scripts/start_smolvla_runner.sh`.
3. State source depends on the mode:
   - inference-only: a protected robot-side reader supplies the current state
     vector to `POST /v1/state` at the control rate. The vector length must equal
     `policy_state_dimension` in `GET /health`.
   - actuation / robot_state_readonly: the runner reads joint state from the arm
     itself each control tick, so no external state pusher is required.
4. The supervisor starts exactly one short action with `POST /v1/actions/start`.
5. The runner resets its old SmolVLA action cache and begins inference using the
   new action and fresh ZMQ frames.
6. The supervisor calls `finish` or `cancel`, then may submit the next action.

## Endpoints

`GET /health`

Shows whether the checkpoint is loaded, the expected state-vector length, the
active task, latest inference time, and whether state data is fresh.

`POST /v1/state`

```json
{"state": [0.0, 0.0], "timestamp": 0}
```

The example length is illustrative only. Send the live state vector from the
robot-side reader with exactly `policy_state_dimension` float values.

`POST /v1/actions/start`

```json
{
  "run_id": "session-1",
  "revision": 0,
  "step": 1,
  "action": "grasp red cube",
  "deadline_seconds": 8
}
```

`POST /v1/actions/finish` and `POST /v1/actions/cancel` use the matching
`run_id`, `revision`, and `step` fields. `GET /events` reports
`action_accepted`, `inference_waiting_for_state`, `action_chunk_generated`, and
`inference_error` events.

The next language task is sent only after the VLM verifier completes or
cancels the current task. Do not submit multiple active tasks at once.

## One-command web stack

`./scripts/start_orchestrated_web.sh` starts the splitter, the CUDA-resident
runner, the VLM orchestrator, and `laptop_app`, the web UI. The browser sends a high-level
task to the orchestrator. The orchestrator creates a VLM contract once and
submits only one contract step at a time to the runner.

The runner refuses a step while the state stream is stale. This prevents the
UI from presenting a task as running when the policy has no current robot
state.

The time window is controlled in `.env.orchestrator`: `grasp`, `move`, and
`place` have independent base durations, and each receives
`SUPERVISOR_ACTION_GRACE_SECONDS`. At the end of a window the VLM always
receives fresh splitter frames and returns `completed`, `not_completed`,
`failed`, or `uncertain`. Only `completed` advances to the next step; every
other result cancels the active revision and creates a new remaining-work
contract.
