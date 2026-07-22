"""Hardware smoke test for the SO-101 actuation boundary.

Run this on the 4090 with the arm connected BEFORE trusting the closed loop.
It uses the same ``RobotActuator`` the runner uses, so a pass here means the
runner's connect / read_state / send_action path works against your LeRobot
version and wiring.

Stages (each is opt-in and increasingly physical):

  1. default: connect, print the resolved motor order and one live state read.
     Never moves the arm.
  2. --send-hold: additionally command the arm to its *current* position. This
     exercises send_action end to end but should produce no visible motion.
  3. --nudge DEG: additionally move the last joint (usually the gripper) by a
     small relative amount, then command it back. Keep DEG tiny (a few degrees).

Examples:
  python verify_actuation.py --port /dev/ttyACM0
  python verify_actuation.py --port /dev/ttyACM0 --send-hold
  python verify_actuation.py --port /dev/ttyACM0 --nudge 3 --max-relative-target 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from dotenv import load_dotenv

from robot_actuator import RobotActuator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env.smolvla"))


def log(event: str, **data: object) -> None:
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the SO-101 actuation boundary.")
    parser.add_argument("--robot-type", default=os.getenv("SMOLVLA_ROBOT_TYPE", "so101_follower"))
    parser.add_argument("--port", default=os.getenv("SMOLVLA_ROBOT_PORT", ""))
    parser.add_argument("--robot-id", default=os.getenv("SMOLVLA_ROBOT_ID", ""))
    parser.add_argument("--use-degrees", type=int, default=int(os.getenv("SMOLVLA_ROBOT_USE_DEGREES", "1")))
    _mrt = os.getenv("SMOLVLA_MAX_RELATIVE_TARGET", "").strip()
    parser.add_argument("--max-relative-target", type=float, default=float(_mrt) if _mrt else None)
    parser.add_argument("--send-hold", action="store_true", help="Command current position (no motion).")
    parser.add_argument("--nudge", type=float, default=0.0, help="Move the last joint by this much, then back.")
    args = parser.parse_args()

    if not args.port:
        log("verify_failed", reason="no_port", detail="Pass --port or set SMOLVLA_ROBOT_PORT.")
        return 2

    # Physical stages need actuation enabled; the read-only stage does not.
    actuation_enabled = args.send_hold or args.nudge != 0.0
    actuator = RobotActuator(
        robot_type=args.robot_type,
        port=args.port,
        robot_id=args.robot_id,
        max_relative_target=args.max_relative_target,
        use_degrees=bool(args.use_degrees),
        actuation_enabled=actuation_enabled,
    )

    try:
        boundary = actuator.connect()
    except Exception as exc:
        log("verify_failed", stage="connect", error=str(exc))
        return 2
    log("actuator_connected", **boundary)

    try:
        state = actuator.read_state()
        log("state_read", dimension=len(state), state=[round(value, 3) for value in state])

        if args.send_hold or args.nudge != 0.0:
            # Command the current position: exercises send_action without motion.
            sent = actuator.send_action(state)
            log("hold_sent", sent_action=sent)
            time.sleep(0.5)

        if args.nudge != 0.0:
            target = list(state)
            target[-1] = target[-1] + args.nudge
            log("nudge_start", joint=actuator.motor_pos_names[-1], delta=args.nudge)
            actuator.send_action(target)
            time.sleep(1.0)
            after = actuator.read_state()
            log("nudge_after", state=[round(value, 3) for value in after])
            # Return to the original position.
            actuator.send_action(state)
            time.sleep(1.0)
            log("nudge_returned", state=[round(value, 3) for value in actuator.read_state()])
    except Exception as exc:
        log("verify_failed", stage="motion", error=str(exc))
        return 2
    finally:
        actuator.disconnect()

    log("verify_ok", actuation_enabled=actuation_enabled)
    return 0


if __name__ == "__main__":
    sys.exit(main())
