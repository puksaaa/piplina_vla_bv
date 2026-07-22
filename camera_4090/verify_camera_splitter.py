import argparse
import sys
import time
from typing import Any

import httpx
import zmq


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def ok(message: str) -> None:
    print(f"OK: {message}")


def check_http(base_url: str, timeout: float, expected: list[str], allow_partial: bool) -> list[str]:
    with httpx.Client(timeout=timeout) as http:
        health = http.get(f"{base_url}/health")
        health.raise_for_status()
        health_data = health.json()

        ready_names: list[str] = []
        cameras = health_data.get("cameras", {})
        for name in expected:
            camera = cameras.get(name)
            if not camera:
                if not allow_partial:
                    fail(f"camera {name!r} is missing from /health")
                continue
            if not camera.get("ready"):
                if allow_partial:
                    print(f"WARN: {name} is not ready: {camera.get('error') or 'no error detail'}")
                    continue
                fail(f"camera {name!r} is not ready: {camera.get('error') or 'no error detail'}")
            ready_names.append(name)
            ok(f"{name} ready, age={camera.get('age_ms')} ms")

        snapshot = http.get(f"{base_url}/snapshot/all")
        snapshot.raise_for_status()
        snapshot_data = snapshot.json()
        frames: dict[str, Any] = snapshot_data.get("frames", {})
        names_to_check = ready_names if allow_partial else expected
        if not names_to_check:
            fail("no cameras are ready")

        for name in names_to_check:
            frame = frames.get(name)
            if not frame:
                fail(f"camera {name!r} is missing from /snapshot/all")
            image = frame.get("image", "")
            if not image.startswith("data:image/jpeg;base64,"):
                fail(f"camera {name!r} did not return JPEG data URL")
            ok(f"{name} snapshot {frame.get('width')}x{frame.get('height')}")

        return names_to_check


def check_zmq(address: str, timeout: float, expected: list[str], allow_partial: bool) -> None:
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.connect(address)

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    deadline = time.time() + timeout

    try:
        while time.time() < deadline:
            events = dict(poller.poll(500))
            if socket not in events:
                continue
            payload = socket.recv_json()
            images = payload.get("images", {})
            timestamps = payload.get("timestamps", {})
            available = [name for name in expected if name in images and name in timestamps]
            missing = [name for name in expected if name not in available]
            if missing and not allow_partial:
                fail(f"ZMQ payload missing cameras: {', '.join(missing)}")
            if not available:
                fail("ZMQ payload did not contain any expected cameras")
            for name in available:
                if not isinstance(images[name], str) or not images[name].startswith("/9j/"):
                    fail(f"ZMQ camera {name!r} does not look like JPEG base64")
            if missing:
                print(f"WARN: ZMQ missing cameras: {', '.join(missing)}")
            ok(f"ZMQ frame received from {address}: {', '.join(available)}")
            return
        fail(f"no ZMQ frame received from {address} within {timeout}s")
    finally:
        socket.close(linger=0)
        context.term()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify camera_splitter HTTP snapshots and optional ZMQ publishing.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8090")
    parser.add_argument("--timeout", type=float, default=10)
    parser.add_argument("--cameras", default="front,wrist")
    parser.add_argument("--zmq-address", default="")
    parser.add_argument("--allow-partial", action="store_true", help="Pass if at least one expected camera is ready.")
    args = parser.parse_args()

    expected = [item.strip() for item in args.cameras.split(",") if item.strip()]
    if not expected:
        fail("at least one camera name is required")

    try:
        ready_names = check_http(args.base_url.rstrip("/"), args.timeout, expected, args.allow_partial)
        if args.zmq_address:
            check_zmq(args.zmq_address, args.timeout, ready_names, args.allow_partial)
    except httpx.HTTPError as exc:
        fail(f"HTTP check failed: {exc}")
    except KeyboardInterrupt:
        print("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
