"""Persistent inference service for a fine-tuned SmolVLA.

The process loads a policy into CUDA once, reads ZMQ camera frames, and swaps
the language task without reloading the model. It deliberately never opens a
robot serial port or sends motor commands: action chunks are emitted as events
for an existing protected actuator boundary to consume.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.smolvla")

POLICY_PATH = os.getenv("SMOLVLA_POLICY_PATH", "").strip()
DEVICE = os.getenv("SMOLVLA_DEVICE", "cuda").strip()
USE_TORCH_COMPILE = os.getenv("SMOLVLA_USE_TORCH_COMPILE", "0") == "1"
ROBOT_TYPE = os.getenv("SMOLVLA_ROBOT_TYPE", "").strip()
ROBOT_PORT = os.getenv("SMOLVLA_ROBOT_PORT", "").strip()
ROBOT_ID = os.getenv("SMOLVLA_ROBOT_ID", "").strip()
RUNNER_FPS = int(os.getenv("SMOLVLA_RUNNER_FPS", "15"))
ACTION_CHUNK_SIZE = int(os.getenv("SMOLVLA_ACTION_CHUNK_SIZE", "0"))
ACTION_QUEUE_SIZE = int(os.getenv("SMOLVLA_ACTION_QUEUE_SIZE", "1"))
ACTION_TIMEOUT_SECONDS = float(os.getenv("SMOLVLA_ACTION_TIMEOUT_SECONDS", "15"))
INFERENCE_ENABLED = os.getenv("SMOLVLA_INFERENCE_ENABLED", "1") == "1"
INFERENCE_FPS = float(os.getenv("SMOLVLA_INFERENCE_FPS", str(RUNNER_FPS)))
STATE_TIMEOUT_MS = int(os.getenv("SMOLVLA_STATE_TIMEOUT_MS", "1000"))
ACTION_EVENT_INTERVAL = max(1, int(os.getenv("SMOLVLA_ACTION_EVENT_INTERVAL", "15")))
CAMERA_HOST = os.getenv("SMOLVLA_CAMERA_HOST", "127.0.0.1").strip()
CAMERA_PORT = int(os.getenv("SMOLVLA_CAMERA_PORT", "5555"))
CAMERA_NAMES = [name.strip() for name in os.getenv("SMOLVLA_CAMERA_NAMES", "camera1,camera2").split(",") if name.strip()]
CAMERA_WIDTH = int(os.getenv("SMOLVLA_CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("SMOLVLA_CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.getenv("SMOLVLA_CAMERA_FPS", "15"))
CAMERA_TIMEOUT_MS = int(os.getenv("SMOLVLA_CAMERA_TIMEOUT_MS", "5000"))
EVENT_LIMIT = int(os.getenv("SMOLVLA_EVENT_LIMIT", "200"))

ALLOWED_ACTION_PREFIXES = ("move ", "grasp ", "place to ")


@dataclass
class CameraExpectation:
    name: str
    width: int | None
    height: int | None


@dataclass
class StateExpectation:
    dimension: int | None


class ActionStartRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=128)
    revision: int = Field(..., ge=0)
    step: int = Field(..., ge=1)
    action: str = Field(..., min_length=1, max_length=500)
    deadline_seconds: float = Field(..., ge=0)


class ActionRefRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=128)
    revision: int = Field(..., ge=0)
    step: int = Field(..., ge=1)
    reason: str = Field(default="", max_length=200)


class StateUpdateRequest(BaseModel):
    """Read-only robot state supplied by the secured robot-side service."""

    state: list[float] = Field(..., min_length=1, max_length=256)
    timestamp: float | None = None


def normalize_action(value: str) -> str:
    return " ".join(value.strip().lower().split())


def is_allowed_action(action: str) -> bool:
    return action == "stop" or action.startswith(ALLOWED_ACTION_PREFIXES)


def extract_camera_expectations(config: Any) -> list[CameraExpectation]:
    """Find `observation.images.<camera>` feature keys in varied LeRobot configs."""

    found: dict[str, CameraExpectation] = {}

    def add_feature(feature_name: Any, feature: Any) -> None:
        if not isinstance(feature_name, str) or not feature_name.startswith("observation.images."):
            return
        camera_name = feature_name.removeprefix("observation.images.")
        shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
        height = width = None
        if isinstance(shape, (list, tuple)) and len(shape) >= 3:
            height, width = int(shape[-2]), int(shape[-1])
        found[camera_name] = CameraExpectation(camera_name, width, height)

    # Current LeRobot policy configs expose input_features as an attribute.
    # Reading it first avoids depending on a particular config serialization API.
    input_features = getattr(config, "input_features", None)
    if isinstance(input_features, dict):
        for name, feature in input_features.items():
            add_feature(name, feature)

    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        return list(found.values())

    def visit(value: Any, key_hint: str = "") -> None:
        if isinstance(value, dict):
            feature_name = value.get("name") if isinstance(value.get("name"), str) else key_hint
            add_feature(feature_name, value)
            for key, child in value.items():
                add_feature(key, child)
                visit(child, key if isinstance(key, str) else "")
        elif hasattr(value, "to_dict"):
            visit(value.to_dict(), key_hint)

    visit(config)
    return list(found.values())


def validate_camera_expectations(
    expected: list[CameraExpectation], configured_names: list[str], width: int, height: int
) -> list[str]:
    errors: list[str] = []
    if not expected:
        return ["Policy config did not expose observation.images camera keys."]
    expected_names = {item.name for item in expected}
    actual_names = set(configured_names)
    if expected_names != actual_names:
        errors.append(f"Policy cameras={sorted(expected_names)}, configured cameras={sorted(actual_names)}")
    for item in expected:
        if item.width is not None and item.width != width:
            errors.append(f"{item.name} expects width {item.width}, configured width is {width}")
        if item.height is not None and item.height != height:
            errors.append(f"{item.name} expects height {item.height}, configured height is {height}")
    return errors


def extract_state_expectation(config: Any) -> StateExpectation:
    """Read the dimensionality of ``observation.state`` from policy metadata."""

    input_features = getattr(config, "input_features", None)
    if isinstance(input_features, dict):
        feature = input_features.get("observation.state")
        shape = feature.get("shape") if isinstance(feature, dict) else getattr(feature, "shape", None)
        if isinstance(shape, (list, tuple)) and shape:
            return StateExpectation(int(shape[-1]))

    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if isinstance(config, dict):
        feature = config.get("input_features", {}).get("observation.state")
        if isinstance(feature, dict):
            shape = feature.get("shape")
            if isinstance(shape, (list, tuple)) and shape:
                return StateExpectation(int(shape[-1]))
    return StateExpectation(None)


def build_dataset_features(
    camera_expectations: list[CameraExpectation], state_expectation: StateExpectation
) -> dict[str, dict[str, Any]]:
    """Create the small feature map required by LeRobot's inference helpers."""

    features: dict[str, dict[str, Any]] = {}
    for camera in camera_expectations:
        if camera.width is None or camera.height is None:
            raise RuntimeError(f"Policy did not expose a complete image shape for {camera.name}.")
        features[f"observation.images.{camera.name}"] = {
            "dtype": "image",
            "shape": (camera.height, camera.width, 3),
            "names": ["height", "width", "channels"],
        }
    if state_expectation.dimension is not None:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (state_expectation.dimension,),
            "names": [f"state_{index}" for index in range(state_expectation.dimension)],
        }
    return features


class RunnerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self.worker_stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.state = "idle"
        self.error: str | None = None
        self.policy: Any | None = None
        self.preprocess: Any | None = None
        self.postprocess: Any | None = None
        self.dataset_features: dict[str, dict[str, Any]] = {}
        self.policy_expectations: list[CameraExpectation] = []
        self.state_expectation = StateExpectation(None)
        self.cameras: dict[str, Any] = {}
        self.active: dict[str, Any] | None = None
        self.task_generation = 0
        self.latest_state: list[float] | None = None
        self.latest_state_at: float | None = None
        self.inference_count = 0
        self.last_inference_at: float | None = None
        self.last_inference_error: str | None = None
        self.last_action_dimension: int | None = None
        self.last_action_at: float | None = None
        self.waiting_for_state_generation: int | None = None
        self.events: deque[dict[str, Any]] = deque(maxlen=EVENT_LIMIT)
        self.event_id = 0
        self.loaded_at: float | None = None

    def emit(self, event: str, **data: Any) -> dict[str, Any]:
        with self.lock:
            self.event_id += 1
            payload = {"id": self.event_id, "event": event, "timestamp": time.time(), **data}
            self.events.append(payload)
            return payload

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "state": self.state,
                "error": self.error,
                "policy_loaded": self.policy is not None,
                "policy_path": POLICY_PATH,
                "device": DEVICE,
                "execution_mode": "inference_only",
                "runner_fps": RUNNER_FPS,
                "inference_enabled": INFERENCE_ENABLED,
                "inference_fps": INFERENCE_FPS,
                "action_chunk_size": ACTION_CHUNK_SIZE,
                "action_queue_size": ACTION_QUEUE_SIZE,
                "action_timeout_seconds": ACTION_TIMEOUT_SECONDS,
                "state_timeout_ms": STATE_TIMEOUT_MS,
                "robot_boundary": {
                    "type": ROBOT_TYPE or None,
                    "port_configured": bool(ROBOT_PORT),
                    "id": ROBOT_ID or None,
                    "actuator_connected": False,
                },
                "loaded_at": self.loaded_at,
                "policy_cameras": [asdict(item) for item in self.policy_expectations],
                "policy_state_dimension": self.state_expectation.dimension,
                "configured_cameras": CAMERA_NAMES,
                "active": self.active,
                "task_generation": self.task_generation,
                "state_available": self.latest_state is not None,
                "state_age_ms": None
                if self.latest_state_at is None
                else int((time.time() - self.latest_state_at) * 1000),
                "inference_count": self.inference_count,
                "last_inference_at": self.last_inference_at,
                "last_inference_error": self.last_inference_error,
                "last_action_dimension": self.last_action_dimension,
                "last_action_at": self.last_action_at,
                "event_id": self.event_id,
            }


runner = RunnerState()
app = FastAPI(title="Fine-tuned SmolVLA Runner")


def load_policy_and_cameras() -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required in SMOLVLA_PYTHON.") from exc
    if not POLICY_PATH:
        raise RuntimeError("SMOLVLA_POLICY_PATH is required.")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("SMOLVLA_DEVICE=cuda but CUDA is not available.")

    try:
        from lerobot.cameras.zmq import ZMQCamera, ZMQCameraConfig
        from lerobot.policies import make_pre_post_processors
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError:
            from lerobot.policies.smolvla import SmolVLAPolicy
    except ImportError as exc:
        raise RuntimeError("LeRobot with SmolVLA and pyzmq extras is required in SMOLVLA_PYTHON.") from exc

    policy = SmolVLAPolicy.from_pretrained(POLICY_PATH)
    policy.eval()
    policy.to(DEVICE)
    expectations = extract_camera_expectations(getattr(policy, "config", None))
    state_expectation = extract_state_expectation(getattr(policy, "config", None))
    errors = validate_camera_expectations(expectations, CAMERA_NAMES, CAMERA_WIDTH, CAMERA_HEIGHT)
    if errors:
        raise RuntimeError("; ".join(errors))
    print(
        json.dumps(
            {
                "event": "smolvla_startup_validation",
                "policy_path": POLICY_PATH,
                "device": DEVICE,
                "expected_observation_images": [asdict(item) for item in expectations],
                "splitter_cameras": CAMERA_NAMES,
                "splitter_resolution": [CAMERA_WIDTH, CAMERA_HEIGHT],
                "compatible": True,
            }
        ),
        flush=True,
    )
    try:
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            POLICY_PATH,
            preprocessor_overrides={"device_processor": {"device": DEVICE}},
        )
    except Exception as exc:
        raise RuntimeError(f"Could not create LeRobot inference preprocessors: {exc}") from exc
    if USE_TORCH_COMPILE:
        policy = torch.compile(policy)

    cameras: dict[str, Any] = {}
    try:
        for name in CAMERA_NAMES:
            camera = ZMQCamera(
                ZMQCameraConfig(
                    server_address=CAMERA_HOST,
                    port=CAMERA_PORT,
                    camera_name=name,
                    width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT,
                    fps=CAMERA_FPS,
                    timeout_ms=CAMERA_TIMEOUT_MS,
                    warmup_s=2,
                )
            )
            camera.connect()
            cameras[name] = camera
    except Exception:
        for camera in cameras.values():
            try:
                camera.disconnect()
            except Exception:
                pass
        raise

    with runner.lock:
        runner.policy = policy
        runner.preprocess = preprocess
        runner.postprocess = postprocess
        runner.dataset_features = build_dataset_features(expectations, state_expectation)
        runner.policy_expectations = expectations
        runner.state_expectation = state_expectation
        runner.cameras = cameras
        runner.loaded_at = time.time()
        runner.state = "ready"
        runner.error = None
    # A connected socket without a frame is not a usable camera. Probe now so
    # startup fails before Gemma can create an executable-looking contract.
    frame_shapes = verify_latest_camera_frames()
    health = runner.snapshot()
    runner.emit("runner_ready", health=health, frame_shapes=frame_shapes)
    print(json.dumps({"event": "smolvla_runner_ready", "health": health, "frame_shapes": frame_shapes}), flush=True)
    if INFERENCE_ENABLED:
        runner.worker_stop.clear()
        runner.worker = threading.Thread(target=inference_loop, name="smolvla-inference", daemon=True)
        runner.worker.start()


def verify_latest_camera_frames() -> dict[str, list[int]]:
    frames: dict[str, list[int]] = {}
    with runner.lock:
        cameras = dict(runner.cameras)
    for name, camera in cameras.items():
        if hasattr(camera, "read_latest"):
            try:
                frame = camera.read_latest(max_age_ms=CAMERA_TIMEOUT_MS)
            except TypeError:
                frame = camera.read_latest()
        else:
            frame = camera.read()
        frames[name] = list(frame.shape)
    return frames


def read_latest_camera_frames() -> dict[str, Any]:
    """Read one fresh RGB frame per policy camera from the splitter."""

    frames: dict[str, Any] = {}
    with runner.lock:
        cameras = dict(runner.cameras)
    for name, camera in cameras.items():
        if hasattr(camera, "read_latest"):
            try:
                frame = camera.read_latest(max_age_ms=CAMERA_TIMEOUT_MS)
            except TypeError:
                frame = camera.read_latest()
        else:
            frame = camera.read()
        if not hasattr(frame, "shape") or len(frame.shape) != 3:
            raise RuntimeError(f"Camera {name} returned an invalid frame.")
        frames[name] = frame
    return frames


def state_is_fresh(updated_at: float | None) -> bool:
    return updated_at is not None and (time.time() - updated_at) * 1000 <= STATE_TIMEOUT_MS


def reset_policy_action_queue() -> None:
    """Discard a SmolVLA action chunk whenever the language task changes."""

    with runner.lock:
        policy = runner.policy
    if policy is None:
        return
    with runner.inference_lock:
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()


def build_policy_input(frames: dict[str, Any], task: str, state: list[float]) -> Any:
    """Use LeRobot's own frame builder so language tokenization matches training."""

    try:
        from lerobot.policies.utils import build_inference_frame
    except ImportError as exc:
        raise RuntimeError("Installed LeRobot does not provide build_inference_frame.") from exc

    with runner.lock:
        dataset_features = dict(runner.dataset_features)
        expected_dimension = runner.state_expectation.dimension
    if expected_dimension is not None and len(state) != expected_dimension:
        raise RuntimeError(f"State dimension is {len(state)}, policy expects {expected_dimension}.")

    values: dict[str, Any] = dict(frames)
    if expected_dimension is not None:
        values.update({f"state_{index}": value for index, value in enumerate(state)})
    return build_inference_frame(
        observation=values,
        ds_features=dataset_features,
        device=DEVICE,
        task=task,
        robot_type=ROBOT_TYPE or None,
    )


def action_dimension(value: Any) -> int | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            size = 1
            for item in shape:
                size *= int(item)
            return size
        except (TypeError, ValueError):
            pass
    try:
        return len(value)
    except TypeError:
        return None


def inference_loop() -> None:
    """Continuously infer only while a supervisor-owned action is active.

    The loop intentionally stops before any actuator call. Its role is to prove
    that task replacement, state injection, image preprocessing, and policy
    inference all work without reloading the CUDA-resident model.
    """

    interval = 1 / max(INFERENCE_FPS, 0.1)
    while not runner.worker_stop.wait(interval):
        with runner.lock:
            active = dict(runner.active) if runner.state == "running" and runner.active else None
            generation = runner.task_generation
            state = list(runner.latest_state) if runner.latest_state is not None else None
            state_at = runner.latest_state_at
            policy = runner.policy
            preprocess = runner.preprocess
            postprocess = runner.postprocess

        if active is None or policy is None or preprocess is None or postprocess is None:
            continue
        if not state_is_fresh(state_at):
            with runner.lock:
                already_reported = runner.waiting_for_state_generation == generation
                runner.waiting_for_state_generation = generation
            if not already_reported:
                runner.emit("inference_waiting_for_state", active=active, state_timeout_ms=STATE_TIMEOUT_MS)
            continue

        try:
            frames = read_latest_camera_frames()
            policy_input = build_policy_input(frames, active["action"], state or [])
            with runner.inference_lock:
                processed_input = preprocess(policy_input)
                action = policy.select_action(processed_input)
                action = postprocess(action)
            dimension = action_dimension(action)
        except Exception as exc:
            message = str(exc)
            with runner.lock:
                runner.last_inference_error = message
            runner.emit("inference_error", active=active, error=message)
            continue

        with runner.lock:
            still_current = (
                runner.state == "running"
                and runner.active is not None
                and runner.task_generation == generation
                and runner.active["run_id"] == active["run_id"]
                and runner.active["revision"] == active["revision"]
                and runner.active["step"] == active["step"]
            )
            if still_current:
                runner.inference_count += 1
                runner.last_inference_at = time.time()
                runner.last_inference_error = None
                runner.last_action_dimension = dimension
                runner.last_action_at = time.time()
                count = runner.inference_count
                runner.active["inference_count"] = int(runner.active.get("inference_count", 0)) + 1
                action_inference_count = runner.active["inference_count"]
            else:
                count = 0
                action_inference_count = 0
        if still_current and (action_inference_count == 1 or count % ACTION_EVENT_INTERVAL == 0):
            runner.emit(
                "action_chunk_generated",
                active=active,
                action_dimension=dimension,
                frame_names=sorted(frames),
                action_inference_count=action_inference_count,
            )


def require_matching_active(request: ActionRefRequest) -> None:
    active = runner.active
    if active is None:
        raise HTTPException(status_code=409, detail="No active action.")
    expected = (active["run_id"], active["revision"], active["step"])
    received = (request.run_id, request.revision, request.step)
    if received != expected:
        raise HTTPException(status_code=409, detail={"error": "stale_action_reference", "active": active})


@app.get("/health")
def health() -> dict[str, Any]:
    return runner.snapshot()


@app.get("/events")
def events(after: int = Query(default=0, ge=0)) -> dict[str, Any]:
    with runner.lock:
        return {"events": [event for event in runner.events if event["id"] > after], "last_event_id": runner.event_id}


@app.post("/v1/state")
def update_state(request: StateUpdateRequest) -> dict[str, Any]:
    """Receive current joint/state values from the protected robot-side reader."""

    state = [float(value) for value in request.state]
    with runner.lock:
        expected_dimension = runner.state_expectation.dimension
    if expected_dimension is not None and len(state) != expected_dimension:
        raise HTTPException(
            status_code=400,
            detail=f"State dimension is {len(state)}, policy expects {expected_dimension}.",
        )
    now = time.time()
    with runner.lock:
        runner.latest_state = state
        runner.latest_state_at = request.timestamp if request.timestamp is not None else now
        runner.waiting_for_state_generation = None
    return {"accepted": True, "dimension": len(state), "received_at": now}


@app.post("/v1/actions/start")
def start_action(request: ActionStartRequest) -> dict[str, Any]:
    action = normalize_action(request.action)
    if not is_allowed_action(action):
        raise HTTPException(status_code=400, detail="Action is outside the SmolVLA runner contract.")
    with runner.lock:
        if runner.state != "ready":
            raise HTTPException(status_code=503, detail={"error": "runner_not_ready", "state": runner.state, "detail": runner.error})
        if runner.active is not None:
            raise HTTPException(status_code=409, detail={"error": "action_already_active", "active": runner.active})
        if runner.state_expectation.dimension is not None and not state_is_fresh(runner.latest_state_at):
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "state_unavailable",
                    "expected_dimension": runner.state_expectation.dimension,
                    "state_timeout_ms": STATE_TIMEOUT_MS,
                },
            )
        if request.deadline_seconds > ACTION_TIMEOUT_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=f"Action deadline exceeds SMOLVLA_ACTION_TIMEOUT_SECONDS={ACTION_TIMEOUT_SECONDS}.",
            )
    try:
        frame_shapes = verify_latest_camera_frames()
    except Exception as exc:
        runner.emit("camera_unavailable", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Fresh ZMQ camera frames are unavailable: {exc}") from exc

    try:
        reset_policy_action_queue()
    except Exception as exc:
        runner.emit("policy_reset_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=f"Could not reset the previous policy action chunk: {exc}") from exc

    active = {
        "run_id": request.run_id,
        "revision": request.revision,
        "step": request.step,
        "action": action,
        "deadline_seconds": request.deadline_seconds,
        "accepted_at": time.time(),
        "inference_count": 0,
    }
    with runner.lock:
        if runner.active is not None:
            raise HTTPException(status_code=409, detail={"error": "action_already_active", "active": runner.active})
        runner.active = active
        runner.state = "running"
        runner.task_generation += 1
        active["generation"] = runner.task_generation
        runner.waiting_for_state_generation = None
        runner.last_inference_error = None
        runner.last_action_dimension = None
        runner.last_action_at = None
    event = runner.emit("action_accepted", active=active, frame_shapes=frame_shapes)
    return {"accepted": True, "active": active, "event": event}


@app.post("/v1/actions/finish")
def finish_action(request: ActionRefRequest) -> dict[str, Any]:
    with runner.lock:
        require_matching_active(request)
        active = runner.active
        runner.active = None
        runner.state = "ready"
    event = runner.emit("action_finished", active=active, reason=request.reason or "completed")
    return {"finished": True, "event": event}


@app.post("/v1/actions/cancel")
def cancel_action(request: ActionRefRequest) -> dict[str, Any]:
    with runner.lock:
        if runner.state not in {"running", "paused"}:
            raise HTTPException(status_code=409, detail={"error": "action_is_not_running", "state": runner.state})
        require_matching_active(request)
        active = runner.active
        runner.active = None
        runner.state = "ready"
    event = runner.emit("action_cancelled", active=active, reason=request.reason or "cancelled")
    return {"cancelled": True, "event": event}


@app.post("/v1/actions/pause")
def pause_action(request: ActionRefRequest) -> dict[str, Any]:
    with runner.lock:
        if runner.state != "running":
            raise HTTPException(status_code=409, detail={"error": "action_is_not_running", "state": runner.state})
        require_matching_active(request)
        active = runner.active
        runner.state = "paused"
    event = runner.emit("action_paused", active=active, reason=request.reason or "paused")
    return {"paused": True, "event": event}


@app.post("/v1/actions/resume")
def resume_action(request: ActionRefRequest) -> dict[str, Any]:
    with runner.lock:
        if runner.state != "paused":
            raise HTTPException(status_code=409, detail={"error": "action_is_not_paused", "state": runner.state})
        require_matching_active(request)
        active = runner.active
        runner.state = "running"
    event = runner.emit("action_resumed", active=active, reason=request.reason or "resumed")
    return {"resumed": True, "event": event}


@app.post("/v1/stop")
def stop_all() -> dict[str, Any]:
    with runner.lock:
        active = runner.active
        runner.active = None
        if runner.state != "error":
            runner.state = "stopped"
    event = runner.emit("runner_stopped", active=active)
    return {"stopped": True, "event": event}


@app.on_event("startup")
def startup() -> None:
    try:
        load_policy_and_cameras()
    except Exception as exc:
        with runner.lock:
            runner.state = "error"
            runner.error = str(exc)
        runner.emit("runner_error", error=str(exc))


@app.on_event("shutdown")
def shutdown() -> None:
    runner.worker_stop.set()
    if runner.worker is not None:
        runner.worker.join(timeout=3)
    with runner.lock:
        cameras = list(runner.cameras.values())
        runner.cameras = {}
        runner.active = None
        runner.policy = None
        runner.preprocess = None
        runner.postprocess = None
        runner.state = "stopped"
    for camera in cameras:
        try:
            camera.disconnect()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent fine-tuned SmolVLA runner.")
    parser.add_argument("--host", default=os.getenv("SMOLVLA_RUNNER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SMOLVLA_RUNNER_PORT", "8091")))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
