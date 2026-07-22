"""Fail-fast readiness check for the resident fine-tuned SmolVLA runner."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.smolvla")


def verify(base_url: str, timeout: float, expected_names: list[str]) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=timeout) as http:
            response = http.get(f"{base_url.rstrip('/')}/health")
            response.raise_for_status()
            health = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RuntimeError(f"SmolVLA runner is unavailable: {exc}") from exc

    if health.get("state") != "ready" or health.get("policy_loaded") is not True:
        raise RuntimeError(f"SmolVLA runner is not ready: {health}")
    if health.get("execution_mode") not in {"inference_only", "robot_state_readonly", "actuation"}:
        raise RuntimeError(f"Unexpected SmolVLA runner execution mode: {health.get('execution_mode')}")

    actual_names = {item.get("name") for item in health.get("policy_cameras", []) if isinstance(item, dict)}
    if actual_names != set(expected_names):
        raise RuntimeError(f"Policy cameras={sorted(actual_names)}, expected cameras={sorted(expected_names)}")
    if health.get("configured_cameras") != expected_names:
        raise RuntimeError(f"Runner cameras={health.get('configured_cameras')}, expected cameras={expected_names}")
    return health


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the resident SmolVLA runner before a task starts.")
    parser.add_argument("--url", default=os.getenv("SMOLVLA_RUNNER_URL", "http://127.0.0.1:8091"))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("SMOLVLA_RUNNER_TIMEOUT", "10")))
    parser.add_argument(
        "--cameras",
        default=os.getenv("SMOLVLA_CAMERA_NAMES", "camera1,camera2"),
        help="Comma-separated camera names that must match policy metadata.",
    )
    args = parser.parse_args()
    expected_names = [name.strip() for name in args.cameras.split(",") if name.strip()]
    if not expected_names:
        parser.error("At least one camera name is required.")
    try:
        health = verify(args.url, args.timeout, expected_names)
    except RuntimeError as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ready": True, "runner": health}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
