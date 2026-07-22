import os
import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

load_dotenv(BASE_DIR / ".env")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
REQUEST_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "300"))
CAMERA_BASE_URL = os.getenv("CAMERA_BASE_URL", "http://127.0.0.1:8090").rstrip("/")
CAMERA_TIMEOUT = float(os.getenv("CAMERA_TIMEOUT", "10"))
CAMERA_MAX_FRAME_AGE_MS = int(os.getenv("CAMERA_MAX_FRAME_AGE_MS", "2000"))
ROBOT_VLM_MODEL = os.getenv("ROBOT_VLM_MODEL", OPENAI_MODEL)
ROBOT_VLM_MAX_TOKENS = int(os.getenv("ROBOT_VLM_MAX_TOKENS", "512"))
ROBOT_VLM_THINK_PREFILL = os.getenv("ROBOT_VLM_THINK_PREFILL", "<think>\n\n</think>\n\n")
MODEL_KEEPALIVE_ENABLED = os.getenv("MODEL_KEEPALIVE_ENABLED", "1") == "1"
MODEL_KEEPALIVE_ON_STARTUP = os.getenv("MODEL_KEEPALIVE_ON_STARTUP", "1") == "1"
MODEL_KEEPALIVE_INTERVAL = float(os.getenv("MODEL_KEEPALIVE_INTERVAL", "30"))
MODEL_KEEPALIVE_TIMEOUT = float(os.getenv("MODEL_KEEPALIVE_TIMEOUT", "300"))
MODEL_KEEPALIVE_MAX_TOKENS = int(os.getenv("MODEL_KEEPALIVE_MAX_TOKENS", "1"))
MODEL_KEEPALIVE_PROMPT = os.getenv("MODEL_KEEPALIVE_PROMPT", "ok")
VLA_ENABLED = os.getenv("VLA_ENABLED", "0") == "1"
VLA_COMMAND_URL = os.getenv("VLA_COMMAND_URL", "").strip()
VLA_TIMEOUT = float(os.getenv("VLA_TIMEOUT", "120"))
VLA_STEP_DELAY = float(os.getenv("VLA_STEP_DELAY", "0.2"))
VLA_STOP_ON_ERROR = os.getenv("VLA_STOP_ON_ERROR", "1") == "1"
ORCHESTRATOR_ENABLED = os.getenv("ORCHESTRATOR_ENABLED", "0") == "1"
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://127.0.0.1:8092").rstrip("/")
ORCHESTRATOR_TIMEOUT = float(os.getenv("ORCHESTRATOR_TIMEOUT", "330"))
WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
LOG_LEVEL = os.getenv("PLANNER_LOG_LEVEL", "INFO").upper()
_log_file = Path(os.getenv("PLANNER_LOG_FILE", "logs/planner.log"))
LOG_FILE = _log_file if _log_file.is_absolute() else BASE_DIR / _log_file

CONTRACT_VERSION = "robot_action_contract.v1"
ALLOWED_ACTION_PATTERNS = [
    r"^move\s+.+\s+to\s+.+$",
    r"^grasp\s+.+$",
    r"^place\s+to\s+.+$",
    r"^stop$",
]
PREDICATE_FIELDS = {
    "object_visible": ("object",),
    "target_visible": ("target",),
    "object_not_visible": ("object",),
    "target_not_visible": ("target",),
    "object_held": ("object",),
    "object_not_held": ("object",),
    "object_in_target": ("object", "target"),
    "object_not_in_target": ("object", "target"),
    "object_on_target": ("object", "target"),
    "object_not_on_target": ("object", "target"),
    "object_touching_target": ("object", "target"),
    "object_not_touching_target": ("object", "target"),
    "object_dropped": ("object",),
    "gripper_open": (),
    "gripper_closed": (),
}
VISIBILITY_FAILURE_PREDICATES = {"object_not_visible", "target_not_visible"}
GRASP_FAILURE_PREDICATES = {"object_not_held", "object_dropped"}
TARGET_FAILURE_PREDICATES = {
    "object_not_in_target",
    "object_not_on_target",
    "object_not_touching_target",
    "object_dropped",
}
TARGET_SUCCESS_PREDICATES = {"object_in_target", "object_on_target", "object_touching_target"}
TARGET_SUCCESS_TO_FAILURE = {
    "object_in_target": "object_not_in_target",
    "object_on_target": "object_not_on_target",
    "object_touching_target": "object_not_touching_target",
}

if not OPENAI_BASE_URL:
    raise RuntimeError("OPENAI_BASE_URL is not set. Copy .env.example to .env and check settings.")

if not OPENAI_MODEL:
    raise RuntimeError("OPENAI_MODEL is not set. Copy .env.example to .env and check settings.")


def configure_logger() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    planner_logger = logging.getLogger("robot_planner")
    planner_logger.setLevel(LOG_LEVEL)
    planner_logger.propagate = False
    if planner_logger.handlers:
        return planner_logger

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")):
        handler.setFormatter(formatter)
        planner_logger.addHandler(handler)
    return planner_logger


logger = configure_logger()

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
    timeout=REQUEST_TIMEOUT,
)

app = FastAPI(title="Local LLM Chat")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

keepalive_stop = threading.Event()
keepalive_lock = threading.Lock()
keepalive_status: dict[str, Any] = {
    "enabled": MODEL_KEEPALIVE_ENABLED,
    "running": False,
    "last_ok_at": None,
    "last_error_at": None,
    "last_error": None,
    "last_latency_ms": None,
    "attempts": 0,
    "successes": 0,
}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class RobotPlanRequest(BaseModel):
    command: str = Field(..., min_length=1)
    images: list[str] = Field(default_factory=list)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "base_url": OPENAI_BASE_URL,
        "model": OPENAI_MODEL,
        "camera_base_url": CAMERA_BASE_URL,
        "camera_max_frame_age_ms": str(CAMERA_MAX_FRAME_AGE_MS),
        "keepalive_enabled": str(MODEL_KEEPALIVE_ENABLED),
        "vla_enabled": str(VLA_ENABLED),
        "vla_command_url": VLA_COMMAND_URL,
        "orchestrator_enabled": str(ORCHESTRATOR_ENABLED),
        "orchestrator_url": ORCHESTRATOR_URL,
    }


@app.on_event("startup")
def start_model_keepalive() -> None:
    if not MODEL_KEEPALIVE_ENABLED:
        return
    keepalive_stop.clear()
    thread = threading.Thread(target=model_keepalive_loop, name="model-keepalive", daemon=True)
    thread.start()


@app.on_event("shutdown")
def stop_model_keepalive() -> None:
    keepalive_stop.set()


@app.get("/api/warmup")
def warmup_status() -> dict[str, Any]:
    with keepalive_lock:
        return {
            **keepalive_status,
            "model": ROBOT_VLM_MODEL,
            "base_url": OPENAI_BASE_URL,
            "interval_seconds": MODEL_KEEPALIVE_INTERVAL,
        }


@app.post("/api/warmup")
def warmup_now() -> dict[str, Any]:
    result = warm_model_once("manual")
    if not result["ok"]:
        raise HTTPException(status_code=503, detail=result["error"])
    return result


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    try:
        models = client.models.list()
        return models.model_dump()
    except (APIConnectionError, APITimeoutError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Endpoint is unavailable or timed out: {exc}",
        ) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Model endpoint error: {exc}") from exc


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[message.model_dump() for message in payload.messages],
            temperature=0.7,
        )
        answer = completion.choices[0].message.content or ""
        return {
            "answer": answer,
            "raw": completion.model_dump(),
        }
    except (APIConnectionError, APITimeoutError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Endpoint is unavailable or timed out: {exc}",
        ) from exc
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"Model endpoint error: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc


@app.get("/api/camera/health")
def camera_health() -> dict[str, Any]:
    try:
        with httpx.Client(timeout=CAMERA_TIMEOUT) as http:
            response = http.get(f"{CAMERA_BASE_URL}/health")
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail=f"Camera splitter timed out: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Camera splitter is unavailable: {exc}") from exc


@app.post("/api/robot/plan")
def robot_plan(payload: RobotPlanRequest) -> dict[str, Any]:
    return build_robot_plan(payload, uuid.uuid4().hex[:12])


@app.post("/api/robot/run")
def robot_run(payload: RobotPlanRequest) -> dict[str, Any]:
    if ORCHESTRATOR_ENABLED:
        if payload.images:
            raise HTTPException(
                status_code=400,
                detail="Orchestrated execution requires live camera frames. Remove uploaded images and try again.",
            )
        return start_orchestrated_run(payload.command)

    result = build_robot_plan(payload, uuid.uuid4().hex[:12])
    plan = result.get("plan") if isinstance(result.get("plan"), list) else []

    if not VLA_ENABLED:
        return {
            **result,
            "mode": "plan_ready",
            "vla_called": False,
            "vla_enabled": False,
            "vla_error": "VLA_ENABLED=0. Set VLA_ENABLED=1 and VLA_COMMAND_URL to execute.",
            "vla_results": [],
        }

    if not VLA_COMMAND_URL:
        raise HTTPException(status_code=500, detail="VLA_COMMAND_URL is not set.")

    if not result.get("task_feasible") or any(step.get("action") == "stop" for step in plan if isinstance(step, dict)):
        return {
            **result,
            "mode": "not_executed",
            "vla_called": False,
            "vla_enabled": True,
            "vla_error": result.get("failure_reason") or "Planner returned stop.",
            "vla_results": [],
        }

    vla_results = dispatch_plan_to_vla(plan)
    return {
        **result,
        "mode": "executed",
        "vla_called": True,
        "vla_enabled": True,
        "vla_command_url": VLA_COMMAND_URL,
        "vla_results": vla_results,
    }


@app.get("/api/robot/runs/{run_id}")
def robot_run_status(run_id: str) -> dict[str, Any]:
    if not ORCHESTRATOR_ENABLED:
        raise HTTPException(status_code=404, detail="Orchestrator mode is disabled.")
    return orchestrator_request("GET", f"/v1/runs/{run_id}")


@app.post("/api/robot/runs/{run_id}/cancel")
def cancel_robot_run(run_id: str) -> dict[str, Any]:
    if not ORCHESTRATOR_ENABLED:
        raise HTTPException(status_code=404, detail="Orchestrator mode is disabled.")
    return orchestrator_request("POST", f"/v1/runs/{run_id}/cancel")


def start_orchestrated_run(command: str) -> dict[str, Any]:
    request_id = uuid.uuid4().hex[:12]
    logger.info("event=orchestrated_run_requested request_id=%s command=%r", request_id, command)
    result = orchestrator_request("POST", "/v1/runs", {"task": command})
    contract = result.get("contract")
    if not isinstance(contract, dict):
        raise HTTPException(status_code=502, detail="Orchestrator returned no VLM contract.")
    logger.info(
        "event=orchestrated_run_created request_id=%s run_id=%s status=%s",
        request_id,
        result.get("run_id"),
        result.get("status"),
    )
    return {
        **contract,
        "mode": "orchestrating",
        "run_id": result.get("run_id"),
        "run_status": result.get("status"),
        "vla_called": False,
        "vla_enabled": False,
        "image_source": "camera_splitter",
    }


def orchestrator_request(method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=ORCHESTRATOR_TIMEOUT) as http:
            response = http.request(method, f"{ORCHESTRATOR_URL}{path}", json=payload)
            response.raise_for_status()
            body = response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail=f"Orchestrator timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:1000]
        raise HTTPException(status_code=exc.response.status_code, detail=f"Orchestrator error: {detail}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Orchestrator is unavailable: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=502, detail="Orchestrator returned malformed JSON.")
    return body


def build_robot_plan(payload: RobotPlanRequest, request_id: str) -> dict[str, Any]:
    started_at = time.perf_counter()
    logger.info(
        "event=plan_started request_id=%s source=%s upload_count=%s command=%r",
        request_id,
        "upload" if payload.images else "camera_splitter",
        len(payload.images),
        payload.command,
    )
    if payload.images:
        frames = build_uploaded_frames(payload.images)
        source = "upload"
    else:
        snapshots = fetch_camera_snapshots()
        frames = snapshots.get("frames", {})
        source = "camera_splitter"

    image_parts = build_image_parts(frames)
    if not image_parts:
        raise HTTPException(status_code=502, detail="No images were available for VLM planning.")

    logger.info(
        "event=images_ready request_id=%s source=%s frame_names=%s image_count=%s",
        request_id,
        source,
        ",".join(frames.keys()),
        len(frames),
    )

    prompt = build_robot_planner_prompt(payload.command)

    try:
        logger.info(
            "event=vlm_request_started request_id=%s model=%s timeout_seconds=%s",
            request_id,
            ROBOT_VLM_MODEL,
            REQUEST_TIMEOUT,
        )
        completion = client.chat.completions.create(
            model=ROBOT_VLM_MODEL,
            temperature=0,
            max_tokens=ROBOT_VLM_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": build_robot_planner_system_prompt(),
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, *image_parts],
                },
                # Qwen-compatible prefill: force completion after internal reasoning.
                {"role": "assistant", "content": ROBOT_VLM_THINK_PREFILL},
            ],
        )
        answer = completion.choices[0].message.content or ""
        logger.info(
            "event=vlm_raw_response request_id=%s response=%s",
            request_id,
            json.dumps(answer, ensure_ascii=False),
        )
        parsed = validate_planner_output(parse_json_answer(answer))
        normalized_answer = json.dumps(parsed, ensure_ascii=False)
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        logger.info(
            "event=vlm_request_finished request_id=%s elapsed_ms=%s feasible=%s steps=%s failure_reason=%s",
            request_id,
            elapsed_ms,
            parsed.get("task_feasible"),
            len(parsed.get("plan") or []),
            parsed.get("failure_reason"),
        )
        return {
            "request_id": request_id,
            "elapsed_ms": elapsed_ms,
            "command": payload.command,
            "mode": "plan_only",
            "vla_called": False,
            "vla_enabled": VLA_ENABLED,
            "image_source": source,
            "plan": parsed.get("plan"),
            "contract": parsed,
            "task_feasible": parsed.get("task_feasible"),
            "failure_reason": parsed.get("failure_reason"),
            # Never expose free-form model text as the planner result.
            "answer": normalized_answer,
            "camera": summarize_frames(frames),
        }
    except (APIConnectionError, APITimeoutError) as exc:
        logger.warning(
            "event=vlm_request_unavailable request_id=%s elapsed_ms=%s error=%s",
            request_id,
            int((time.perf_counter() - started_at) * 1000),
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail=f"VLM endpoint is unavailable or timed out: {exc}",
        ) from exc
    except OpenAIError as exc:
        logger.warning(
            "event=vlm_request_error request_id=%s elapsed_ms=%s error=%s",
            request_id,
            int((time.perf_counter() - started_at) * 1000),
            exc,
        )
        raise HTTPException(status_code=502, detail=f"VLM endpoint error: {exc}") from exc
    except Exception as exc:
        logger.exception(
            "event=plan_unexpected_error request_id=%s elapsed_ms=%s",
            request_id,
            int((time.perf_counter() - started_at) * 1000),
        )
        raise HTTPException(status_code=500, detail=f"Unexpected robot planner error: {exc}") from exc


def fetch_camera_snapshots() -> dict[str, Any]:
    try:
        with httpx.Client(timeout=CAMERA_TIMEOUT) as http:
            response = http.get(f"{CAMERA_BASE_URL}/snapshot/all")
            response.raise_for_status()
            snapshots = response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=503, detail=f"Camera splitter timed out: {exc}") from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500]
        raise HTTPException(status_code=503, detail=f"Camera splitter returned HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Camera splitter is unavailable: {exc}") from exc

    frames = snapshots.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise HTTPException(status_code=503, detail="Camera splitter returned no frames.")
    fresh_frames = {
        name: frame
        for name, frame in frames.items()
        if isinstance(frame, dict)
        and isinstance(frame.get("age_ms"), (int, float))
        and frame["age_ms"] <= CAMERA_MAX_FRAME_AGE_MS
    }
    if not fresh_frames:
        raise HTTPException(
            status_code=503,
            detail=f"Camera frames are older than {CAMERA_MAX_FRAME_AGE_MS} ms.",
        )
    return {**snapshots, "frames": fresh_frames}


def dispatch_plan_to_vla(plan: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=VLA_TIMEOUT) as http:
        total = len(plan)
        for index, step in enumerate(plan, start=1):
            if not isinstance(step, dict):
                continue
            action = step.get("action")
            if not isinstance(action, str) or action == "stop":
                continue

            sent_at = time.time()
            payload = {
                "command": action,
                "step": index,
                "total_steps": total,
                "source": "vlm_planner",
            }

            try:
                response = http.post(VLA_COMMAND_URL, json=payload)
                body = parse_vla_response(response)
                result = {
                    "step": index,
                    "command": action,
                    "ok": response.is_success,
                    "status_code": response.status_code,
                    "latency_ms": int((time.time() - sent_at) * 1000),
                    "response": body,
                }
                results.append(result)
                if not response.is_success and VLA_STOP_ON_ERROR:
                    break
            except httpx.HTTPError as exc:
                results.append(
                    {
                        "step": index,
                        "command": action,
                        "ok": False,
                        "status_code": None,
                        "latency_ms": int((time.time() - sent_at) * 1000),
                        "error": str(exc),
                    }
                )
                if VLA_STOP_ON_ERROR:
                    break

            if VLA_STEP_DELAY > 0 and index < total:
                time.sleep(VLA_STEP_DELAY)

    return results


def parse_vla_response(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text[:500]


def model_keepalive_loop() -> None:
    with keepalive_lock:
        keepalive_status["running"] = True

    try:
        if MODEL_KEEPALIVE_ON_STARTUP:
            warm_model_once("startup")

        while not keepalive_stop.wait(MODEL_KEEPALIVE_INTERVAL):
            warm_model_once("interval")
    finally:
        with keepalive_lock:
            keepalive_status["running"] = False


def warm_model_once(reason: str) -> dict[str, Any]:
    started_at = time.time()
    with keepalive_lock:
        keepalive_status["attempts"] += 1

    try:
        warm_client = client.with_options(timeout=MODEL_KEEPALIVE_TIMEOUT)
        warm_client.chat.completions.create(
            model=ROBOT_VLM_MODEL,
            temperature=0,
            max_tokens=MODEL_KEEPALIVE_MAX_TOKENS,
            messages=[
                {
                    "role": "system",
                    "content": "Return exactly one short token. No explanation.",
                },
                {
                    "role": "user",
                    "content": MODEL_KEEPALIVE_PROMPT,
                },
            ],
        )
        latency_ms = int((time.time() - started_at) * 1000)
        with keepalive_lock:
            keepalive_status.update(
                {
                    "last_ok_at": time.time(),
                    "last_error": None,
                    "last_latency_ms": latency_ms,
                    "successes": keepalive_status["successes"] + 1,
                }
            )
        logger.info("event=keepalive_ok reason=%s latency_ms=%s", reason, latency_ms)
        return {"ok": True, "reason": reason, "latency_ms": latency_ms}
    except Exception as exc:
        latency_ms = int((time.time() - started_at) * 1000)
        error = str(exc)
        with keepalive_lock:
            keepalive_status.update(
                {
                    "last_error_at": time.time(),
                    "last_error": error,
                    "last_latency_ms": latency_ms,
                }
            )
        logger.warning("event=keepalive_failed reason=%s latency_ms=%s error=%s", reason, latency_ms, error)
        return {"ok": False, "reason": reason, "latency_ms": latency_ms, "error": error}


def build_image_parts(frames: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for name, frame in frames.items():
        image = frame.get("image") if isinstance(frame, dict) else None
        if not image:
            continue
        if not image.startswith("data:image/"):
            image = f"data:image/jpeg;base64,{image}"
        parts.append({"type": "text", "text": f"Camera: {name}"})
        parts.append({"type": "image_url", "image_url": {"url": image}})
    return parts


def build_uploaded_frames(images: list[str]) -> dict[str, Any]:
    frames: dict[str, Any] = {}
    for index, image in enumerate(images, start=1):
        if not image.startswith("data:image/"):
            raise HTTPException(status_code=400, detail=f"Uploaded image {index} is not a data:image URL.")
        if ";base64," not in image:
            raise HTTPException(status_code=400, detail=f"Uploaded image {index} must be base64 encoded.")
        frames[f"upload_{index}"] = {
            "image": image,
            "timestamp": None,
            "age_ms": None,
            "width": None,
            "height": None,
        }
    return frames


def summarize_frames(frames: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, frame in frames.items():
        if not isinstance(frame, dict):
            continue
        summary[name] = {
            "timestamp": frame.get("timestamp"),
            "age_ms": frame.get("age_ms"),
            "width": frame.get("width"),
            "height": frame.get("height"),
        }
    return summary


def build_robot_planner_system_prompt() -> str:
    return """
ROLE: deterministic vision-language planner for a VLA.

You already completed all private reasoning. Do not output reasoning, analysis, explanation, markdown, or prose.
Use only visible information in the supplied RGB camera images. A partially visible or partly occluded object still counts as visible. Do not reject a task just because part of a requested object is outside the frame or covered.

Return exactly one valid JSON object and nothing before or after it. The object is a robot action contract.

The plan may contain ONLY these lowercase action strings:
1. "move <visible object> to <visible object or visible destination>"
2. "grasp <visible object>"
3. "place to <visible object or visible destination>"
4. "stop"

No other action verbs, command forms, synonyms, parentheses, function calls, or punctuation commands are allowed.
Forbidden words in every action: left, right, up, down, above, below, forward, backward, front, back, behind, coordinate, pixel, angle, distance.
Never use directions, coordinates, trajectories, measurements, or inferred objects.
Describe every object only by visible attributes, for example "red cube", "white plate", or "black marker beside notebook".

Steps are ordered, shortest first-to-last VLA instructions. Each step contains exactly one action string.
Do not create extra steps. Do not explain a step.

Every non-stop step MUST include a machine-readable verification object:
{
  "required_visible": ["visible object names"],
  "success": [{"predicate": "one allowed predicate", "object": "when required", "target": "when required"}],
  "failure": [{"predicate": "one allowed predicate", "object": "when required", "target": "when required"}],
  "on_uncertain": "stop"
}

Allowed verification predicates and fields:
- object_visible(object)
- target_visible(target)
- object_not_visible(object)
- target_not_visible(target)
- object_held(object)
- object_not_held(object)
- object_in_target(object, target)
- object_not_in_target(object, target)
- object_on_target(object, target)
- object_not_on_target(object, target)
- object_touching_target(object, target)
- object_not_touching_target(object, target)
- object_dropped(object)
- gripper_open()
- gripper_closed()

Use only those predicate strings and fields. Do not put explanation text inside verification.
`required_visible` is only the list of things the verifier should look for before judging the step.
Never put object_not_visible or target_not_visible inside `failure`. If the verifier cannot see the required object or target after execution, that is `uncertain`, not `failed`.
`failure` means a visibly wrong result after the VLA tried the action: the object is not held, is not in/on/touching the target, or was dropped/fell.
For `grasp`, success must include object_held and failure must include object_not_held or object_dropped.
For `move` and `place`, success must include object_in_target, object_on_target, or object_touching_target. Failure must include object_not_in_target, object_not_on_target, object_not_touching_target, or object_dropped.
`on_uncertain` is always exactly `stop`.

For a feasible task, use exactly this shape:
{
  "contract_version": "robot_action_contract.v1",
  "task_feasible": true,
  "plan": [{
    "step": 1,
    "action": "move red cube to white plate",
    "verification": {
      "required_visible": ["red cube", "white plate"],
      "success": [{"predicate": "object_on_target", "object": "red cube", "target": "white plate"}],
      "failure": [{"predicate": "object_not_on_target", "object": "red cube", "target": "white plate"}],
      "on_uncertain": "stop"
    }
  }],
  "failure_reason": null
}

If the task is truly impossible after a best-effort reading of the image, use exactly this shape:
{
  "contract_version": "robot_action_contract.v1",
  "task_feasible": false,
  "plan": [{"step": 1, "action": "stop", "verification": null}],
  "failure_reason": "not_visible_or_ambiguous"
}

JSON is the complete answer. Never add any text outside JSON.
""".strip()


def build_robot_planner_prompt(command: str) -> str:
    return f"""
TASK:
{command}

Return ONLY this JSON shape:
{{
  "contract_version": "robot_action_contract.v1",
  "task_feasible": true,
  "plan": [
    {{
      "step": 1,
      "action": "move red cube to white plate",
      "verification": {{
        "required_visible": ["red cube", "white plate"],
        "success": [{{"predicate": "object_on_target", "object": "red cube", "target": "white plate"}}],
        "failure": [{{"predicate": "object_not_on_target", "object": "red cube", "target": "white plate"}}],
        "on_uncertain": "stop"
      }}
    }}
  ],
  "failure_reason": null
}}

For a request about "all" visible objects, create an ordered sequence for every clearly visible object that matches the requested class. Give each object a visible description. Do not include `stop` in a feasible plan.

If impossible or ambiguous, return exactly one stop JSON with `task_feasible` set to false.
""".strip()


def parse_json_answer(answer: str) -> dict[str, Any]:
    cleaned = answer.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    elif not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else stop_plan("VLM did not return a JSON object.")
    except json.JSONDecodeError:
        return stop_plan("VLM did not return one valid JSON object.")


def validate_planner_output(parsed: dict[str, Any]) -> dict[str, Any]:
    if parsed.get("contract_version") != CONTRACT_VERSION:
        return stop_plan("Planner returned an unsupported contract version.")

    task_feasible = parsed.get("task_feasible")
    if not isinstance(task_feasible, bool):
        return stop_plan("Planner returned an invalid feasibility flag.")
    if not task_feasible:
        reason = parsed.get("failure_reason")
        return stop_plan(reason if isinstance(reason, str) and reason else "not_visible_or_ambiguous")

    plan = parsed.get("plan")
    if not isinstance(plan, list) or not plan:
        return stop_plan("Planner returned no valid plan.")

    normalized_plan: list[dict[str, Any]] = []
    for index, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            return stop_plan("Planner returned a malformed step.")
        action = step.get("action")
        if not isinstance(action, str):
            return stop_plan("Planner returned a step without an action.")
        action = normalize_plain_action(action)
        if contains_forbidden_direction(action):
            return stop_plan(f"Planner returned forbidden direction word: {action}")
        if not any(re.match(pattern, action) for pattern in ALLOWED_ACTION_PATTERNS):
            return stop_plan(f"Planner returned forbidden action: {action}")
        if action == "stop":
            return stop_plan("Planner returned stop in a feasible plan.")
        verification = validate_verification(step.get("verification"), action)
        if verification is None:
            return stop_plan("Planner returned invalid verification criteria.")
        normalized_plan.append({"step": index, "action": action, "verification": verification})

    return {
        "contract_version": CONTRACT_VERSION,
        "task_feasible": True,
        "plan": normalized_plan,
        "failure_reason": None,
    }


def stop_plan(reason: str) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "task_feasible": False,
        "plan": [{"step": 1, "action": "stop", "verification": None}],
        "failure_reason": reason,
    }


def validate_verification(value: Any, action: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("on_uncertain") != "stop":
        return None

    required_visible = value.get("required_visible")
    success = value.get("success")
    failure = value.get("failure")
    if not isinstance(required_visible, list) or not required_visible:
        return None
    normalized_visible = [normalize_reference(item) for item in required_visible]
    if any(item is None for item in normalized_visible):
        return None
    normalized_success = validate_conditions(success)
    normalized_failure = validate_conditions(failure)
    if not normalized_success or not normalized_failure:
        return None
    normalized_failure = [
        condition for condition in normalized_failure if condition["predicate"] not in VISIBILITY_FAILURE_PREDICATES
    ]
    if not normalized_failure:
        normalized_failure = derive_failure_conditions(action, normalized_success)
    if not normalized_failure:
        return None

    success_predicates = {condition["predicate"] for condition in normalized_success}
    failure_predicates = {condition["predicate"] for condition in normalized_failure}
    if VISIBILITY_FAILURE_PREDICATES & failure_predicates:
        return None
    if action.startswith("grasp ") and "object_held" not in success_predicates:
        return None
    if action.startswith("grasp ") and not (GRASP_FAILURE_PREDICATES & failure_predicates):
        return None
    if action.startswith(("move ", "place to ")) and not (TARGET_SUCCESS_PREDICATES & success_predicates):
        return None
    if action.startswith(("move ", "place to ")) and not (TARGET_FAILURE_PREDICATES & failure_predicates):
        return None

    return {
        "required_visible": normalized_visible,
        "success": normalized_success,
        "failure": normalized_failure,
        "on_uncertain": "stop",
    }


def derive_failure_conditions(action: str, success_conditions: list[dict[str, str]]) -> list[dict[str, str]]:
    if action.startswith("grasp "):
        return [{"predicate": "object_not_held", "object": action.removeprefix("grasp ").strip()}]

    for condition in success_conditions:
        predicate = condition["predicate"]
        if predicate in TARGET_SUCCESS_TO_FAILURE:
            return [
                {
                    "predicate": TARGET_SUCCESS_TO_FAILURE[predicate],
                    "object": condition["object"],
                    "target": condition["target"],
                }
            ]

    match = re.match(r"^move\s+(.+)\s+to\s+(.+)$", action)
    if match:
        return [{"predicate": "object_not_on_target", "object": match.group(1), "target": match.group(2)}]
    return []


def validate_conditions(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list) or not value:
        return None
    normalized: list[dict[str, str]] = []
    for condition in value:
        if not isinstance(condition, dict):
            return None
        predicate = condition.get("predicate")
        if predicate not in PREDICATE_FIELDS:
            return None
        expected_fields = {"predicate", *PREDICATE_FIELDS[predicate]}
        supplied_fields = set(condition)
        if not expected_fields.issubset(supplied_fields) or not supplied_fields.issubset({"predicate", "object", "target"}):
            return None
        normalized_condition = {"predicate": predicate}
        for field in PREDICATE_FIELDS[predicate]:
            reference = normalize_reference(condition.get(field))
            if reference is None:
                return None
            normalized_condition[field] = reference
        normalized.append(normalized_condition)
    return normalized


def normalize_reference(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    reference = " ".join(value.strip().lower().split())
    if not reference or contains_forbidden_direction(reference):
        return None
    return reference


def normalize_plain_action(action: str) -> str:
    action = action.strip().lower()
    action = re.sub(r"\s+", " ", action)
    action = action.rstrip(".;")
    return action


def contains_forbidden_direction(action: str) -> bool:
    forbidden_words = {
        "left",
        "right",
        "up",
        "down",
        "above",
        "below",
        "forward",
        "backward",
        "front",
        "back",
        "behind",
        "coordinate",
        "coordinates",
        "pixel",
        "pixels",
        "angle",
        "angles",
        "лево",
        "право",
        "вверх",
        "вниз",
        "слева",
        "справа",
        "сверху",
        "снизу",
    }
    words = set(re.findall(r"[a-zа-яё]+", action, flags=re.IGNORECASE))
    return bool(words & forbidden_words)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=WEB_HOST, port=WEB_PORT, reload=False)
