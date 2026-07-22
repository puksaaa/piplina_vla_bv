"""Timed visual verification and replanning for robot action contracts.

The process is deliberately an observer/planner. It emits ready actions and
verification events but never opens a robot port or transmits motor commands.
"""

import argparse
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from vlm_terminal import (
    CONTRACT_VERSION,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    ROBOT_VLM_MAX_TOKENS,
    ROBOT_VLM_THINK_PREFILL,
    load_image_parts,
    normalize_reference,
    stop_plan,
    system_prompt,
    validate_conditions,
    validate_plan,
)


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.smolvla")
load_dotenv(BASE_DIR / ".env.orchestrator")

ASSESSMENT_VERSION = "robot_step_assessment.v1"
ASSESSMENT_STATUSES = {"completed", "not_completed", "failed", "uncertain"}
REASON_CODES = {
    "success_criteria_observed",
    "success_criteria_not_observed",
    "failure_criteria_observed",
    "uncertain_visual_evidence",
    "required_object_not_visible",
    "invalid_assessment",
}
KEEPALIVE_ENABLED = os.getenv("SUPERVISOR_KEEPALIVE_ENABLED", "1") == "1"
KEEPALIVE_INTERVAL = float(os.getenv("SUPERVISOR_KEEPALIVE_INTERVAL", "20"))
KEEPALIVE_TIMEOUT = float(os.getenv("SUPERVISOR_KEEPALIVE_TIMEOUT", "30"))
RUNNER_URL = os.getenv("SMOLVLA_RUNNER_URL", "http://127.0.0.1:8091").rstrip("/")
RUNNER_TIMEOUT = float(os.getenv("SMOLVLA_RUNNER_TIMEOUT", "10"))

# The supervisor owns execution windows. The policy receives a deterministic
# per-action deadline, while the VLM verifies only after that window expires.
ACTION_GRACE_SECONDS = float(os.getenv("SUPERVISOR_ACTION_GRACE_SECONDS", "3"))
GRASP_TIMEOUT_SECONDS = float(os.getenv("SUPERVISOR_GRASP_TIMEOUT_SECONDS", "6"))
MOVE_TIMEOUT_SECONDS = float(os.getenv("SUPERVISOR_MOVE_TIMEOUT_SECONDS", "10"))
PLACE_TIMEOUT_SECONDS = float(os.getenv("SUPERVISOR_PLACE_TIMEOUT_SECONDS", "6"))
RUNNER_POLL_INTERVAL_SECONDS = float(os.getenv("SUPERVISOR_RUNNER_POLL_INTERVAL_SECONDS", "0.5"))
MAX_ACTION_TIMEOUT_SECONDS = float(os.getenv("SUPERVISOR_MAX_ACTION_TIMEOUT_SECONDS", "15"))


def assessment_prompt() -> str:
    return """
ROLE: deterministic visual action verifier for a robot VLA.

You already completed all private reasoning. Return exactly one valid JSON object and nothing else.
You receive one planned action, its allowed visual verification contract, and fresh camera images after the action deadline.
Use only visible evidence in the fresh images. Never invent an object or an observation.

Return exactly this schema:
{
  "assessment_version": "robot_step_assessment.v1",
  "step": 1,
  "status": "completed | not_completed | failed | uncertain",
  "observed_predicates": [
    {"predicate": "one predicate copied exactly from the supplied verification contract"}
  ],
  "replan_required": true,
  "reason_code": "one allowed reason code"
}

Allowed status rules:
- completed: at least one supplied success predicate is visibly true; replan_required is false; reason_code is success_criteria_observed.
- failed: at least one supplied failure predicate is visibly true; replan_required is true; reason_code is failure_criteria_observed.
- not_completed: neither success nor failure is visible by the deadline; replan_required is true; reason_code is success_criteria_not_observed.
- uncertain: evidence is insufficient or contradictory, including when a required object or target is not visible; replan_required is true; reason_code is uncertain_visual_evidence or required_object_not_visible.

observed_predicates may contain only exact predicate objects that appear in the supplied success or failure arrays. Do not add explanations, directions, coordinates, or extra fields.
""".strip()


def planner_prompt(task: str, completed_actions: list[str], assessment: dict[str, Any] | None) -> str:
    context = {
        "completed_actions": completed_actions,
        "last_assessment": assessment,
    }
    return (
        f"TASK:\n{task}\n\n"
        f"EXECUTION_CONTEXT:\n{json.dumps(context, ensure_ascii=False)}\n\n"
        "Return a new robot_action_contract.v1 for only the remaining work. "
        "Do not repeat completed actions. Return the required JSON object only."
    )


def assessment_stop(step: int, reason_code: str = "invalid_assessment") -> dict[str, Any]:
    return {
        "assessment_version": ASSESSMENT_VERSION,
        "step": step,
        "status": "uncertain",
        "observed_predicates": [],
        "replan_required": True,
        "reason_code": reason_code,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def validate_assessment(raw: str, step: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return assessment_stop(step["step"])

    expected_fields = {
        "assessment_version",
        "step",
        "status",
        "observed_predicates",
        "replan_required",
        "reason_code",
    }
    if not isinstance(parsed, dict) or set(parsed) != expected_fields:
        return assessment_stop(step["step"])
    if parsed.get("assessment_version") != ASSESSMENT_VERSION or parsed.get("step") != step["step"]:
        return assessment_stop(step["step"])
    status = parsed.get("status")
    reason = parsed.get("reason_code")
    if status not in ASSESSMENT_STATUSES or reason not in REASON_CODES:
        return assessment_stop(step["step"])
    if not isinstance(parsed.get("replan_required"), bool):
        return assessment_stop(step["step"])

    verification = step.get("verification")
    if not isinstance(verification, dict):
        return assessment_stop(step["step"])
    observed = validate_conditions(parsed.get("observed_predicates"))
    if observed is None and parsed.get("observed_predicates") != []:
        return assessment_stop(step["step"])
    observed = observed or []
    allowed_observed = {
        canonical_json(item) for item in [*verification.get("success", []), *verification.get("failure", [])]
    }
    if any(canonical_json(item) not in allowed_observed for item in observed):
        return assessment_stop(step["step"])

    success = {canonical_json(item) for item in verification.get("success", [])}
    failure = {canonical_json(item) for item in verification.get("failure", [])}
    observed_set = {canonical_json(item) for item in observed}
    observed_success = bool(observed_set & success)
    observed_failure = bool(observed_set & failure)

    valid_by_status = {
        "completed": observed_success and not observed_failure and not parsed["replan_required"] and reason == "success_criteria_observed",
        "failed": observed_failure and parsed["replan_required"] and reason == "failure_criteria_observed",
        "not_completed": not observed_success and not observed_failure and parsed["replan_required"] and reason == "success_criteria_not_observed",
        "uncertain": (
            not observed_success
            and not observed_failure
            and parsed["replan_required"]
            and reason in {"uncertain_visual_evidence", "required_object_not_visible"}
        ),
    }
    if not valid_by_status[status]:
        return assessment_stop(step["step"])
    return {
        "assessment_version": ASSESSMENT_VERSION,
        "step": step["step"],
        "status": status,
        "observed_predicates": observed,
        "replan_required": parsed["replan_required"],
        "reason_code": reason,
    }


class ResidentVLM:
    """One persistent client plus a lightweight keepalive to avoid idle unloads."""

    def __init__(self) -> None:
        self.client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.status: dict[str, Any] = {
            "attempts": 0,
            "successes": 0,
            "last_error": None,
            "last_ok_at": None,
            "last_latency_ms": None,
        }
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.warmup()
        if KEEPALIVE_ENABLED:
            self.thread = threading.Thread(target=self._keepalive_loop, name="gemma-keepalive", daemon=True)
            self.thread.start()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def warmup(self) -> bool:
        self.status["attempts"] += 1
        started_at = time.perf_counter()
        try:
            with self.lock:
                self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": "ready"}],
                    max_tokens=1,
                    temperature=0,
                    timeout=KEEPALIVE_TIMEOUT,
                )
            self.status["successes"] += 1
            self.status["last_ok_at"] = time.time()
            self.status["last_error"] = None
            self.status["last_latency_ms"] = int((time.perf_counter() - started_at) * 1000)
            return True
        except Exception as exc:
            self.status["last_error"] = str(exc)
            self.status["last_latency_ms"] = int((time.perf_counter() - started_at) * 1000)
            return False

    def _keepalive_loop(self) -> None:
        while not self.stop_event.wait(max(KEEPALIVE_INTERVAL, 1)):
            self.warmup()

    def create_contract(
        self,
        task: str,
        completed_actions: list[str],
        assessment: dict[str, Any] | None,
    ) -> dict[str, Any]:
        try:
            image_parts = load_image_parts()
        except RuntimeError as exc:
            return stop_plan(f"camera_unavailable: {exc}")
        try:
            with self.lock:
                completion = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0,
                    max_tokens=ROBOT_VLM_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": system_prompt()},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": planner_prompt(task, completed_actions, assessment)}, *image_parts],
                        },
                        {"role": "assistant", "content": ROBOT_VLM_THINK_PREFILL},
                    ],
                )
        except Exception as exc:
            return stop_plan(f"vlm_unavailable_or_timed_out: {exc}")
        return validate_plan(completion.choices[0].message.content or "")

    def assess_step(self, step: dict[str, Any]) -> dict[str, Any]:
        try:
            image_parts = load_image_parts()
        except RuntimeError:
            return assessment_stop(step["step"], "uncertain_visual_evidence")
        payload = {
            "action": step["action"],
            "step": step["step"],
            "verification": step["verification"],
        }
        try:
            with self.lock:
                completion = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    temperature=0,
                    max_tokens=ROBOT_VLM_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": assessment_prompt()},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"STEP_CONTRACT:\n{json.dumps(payload, ensure_ascii=False)}"},
                                *image_parts,
                            ],
                        },
                        {"role": "assistant", "content": ROBOT_VLM_THINK_PREFILL},
                    ],
                )
        except Exception:
            return assessment_stop(step["step"], "uncertain_visual_evidence")
        return validate_assessment(completion.choices[0].message.content or "", step)


class RunnerClient:
    """Local runner protocol; stale revisions are rejected by both sides."""

    def __init__(self, base_url: str = RUNNER_URL, timeout: float = RUNNER_TIMEOUT) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.last_event_id = 0

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.timeout) as http:
                response = http.request(method, f"{self.base_url}{path}", json=payload)
                response.raise_for_status()
                return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"runner_request_failed: {exc}") from exc

    def ensure_ready(self) -> dict[str, Any]:
        health = self._request("GET", "/health")
        if health.get("state") != "ready" or not health.get("policy_loaded"):
            raise RuntimeError(f"runner_not_ready: {health}")
        return health

    def start(self, run_id: str, revision: int, step: dict[str, Any], deadline_seconds: float) -> dict[str, Any]:
        payload = {
            "run_id": run_id,
            "revision": revision,
            "step": step["step"],
            "action": step["action"],
            "deadline_seconds": deadline_seconds,
        }
        response = self._request("POST", "/v1/actions/start", payload)
        active = response.get("active")
        expected = (run_id, revision, step["step"], step["action"])
        received = (
            active.get("run_id") if isinstance(active, dict) else None,
            active.get("revision") if isinstance(active, dict) else None,
            active.get("step") if isinstance(active, dict) else None,
            active.get("action") if isinstance(active, dict) else None,
        )
        if response.get("accepted") is not True or received != expected:
            raise RuntimeError(f"runner_ack_mismatch: expected={expected}, received={received}")
        return response

    def finish(self, run_id: str, revision: int, step: int) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/actions/finish",
            {"run_id": run_id, "revision": revision, "step": step, "reason": "visual_completion"},
        )

    def cancel(self, run_id: str, revision: int, step: int, reason: str) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/actions/cancel",
            {"run_id": run_id, "revision": revision, "step": step, "reason": reason},
        )

    def events(self) -> list[dict[str, Any]]:
        response = self._request("GET", f"/events?after={self.last_event_id}")
        events = response.get("events")
        if not isinstance(events, list):
            raise RuntimeError("runner returned malformed event stream")
        self.last_event_id = int(response.get("last_event_id", self.last_event_id))
        return [event for event in events if isinstance(event, dict)]


@dataclass
class RunState:
    task: str
    step_timeout_s: float
    max_replans: int
    completed_actions: list[str] = field(default_factory=list)
    revision: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    history: list[dict[str, Any]] = field(default_factory=list)
    initial_contract: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    history_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class Supervisor:
    def __init__(self, vlm: ResidentVLM, runner: RunnerClient, state: RunState, log_path: Path | None) -> None:
        self.vlm = vlm
        self.runner = runner
        self.state = state
        self.log_path = log_path

    def emit(self, event: str, **data: Any) -> None:
        payload = {"event": event, "timestamp": time.time(), **data}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        with self.state.history_lock:
            self.state.history.append(payload)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def execution_timeout(self, action: str) -> dict[str, float | str]:
        """Return a deterministic action window with an explicit safety margin."""

        if self.state.step_timeout_s <= 0:
            return {
                "action_kind": "immediate_verification",
                "base_seconds": 0.0,
                "grace_seconds": 0.0,
                "deadline_seconds": 0.0,
            }

        if action.startswith("grasp "):
            base_seconds = GRASP_TIMEOUT_SECONDS
            action_kind = "grasp"
        elif action.startswith("move "):
            base_seconds = MOVE_TIMEOUT_SECONDS
            action_kind = "move"
        elif action.startswith("place to "):
            base_seconds = PLACE_TIMEOUT_SECONDS
            action_kind = "place"
        else:
            base_seconds = self.state.step_timeout_s
            action_kind = "fallback"

        requested_seconds = max(base_seconds + ACTION_GRACE_SECONDS, self.state.step_timeout_s)
        deadline_seconds = min(requested_seconds, MAX_ACTION_TIMEOUT_SECONDS)
        return {
            "action_kind": action_kind,
            "base_seconds": base_seconds,
            "grace_seconds": ACTION_GRACE_SECONDS,
            "deadline_seconds": deadline_seconds,
        }

    def wait_for_execution_window(self, revision: int, step: int, deadline_seconds: float) -> str:
        """Wait for the full action window while surfacing runner failures early."""

        deadline = time.monotonic() + deadline_seconds
        while True:
            if self.state.cancel_event.is_set():
                return "cancelled"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return "deadline"
            time.sleep(min(max(RUNNER_POLL_INTERVAL_SECONDS, 0.1), remaining))
            try:
                events = self.emit_runner_events(revision, step)
            except RuntimeError as exc:
                self.emit("runner_event_poll_failed", detail=str(exc), revision=revision, step=step)
                return "runner_unavailable"
            if any(event.get("event") == "inference_error" for event in events):
                return "inference_error"
            if any(event.get("event") == "inference_waiting_for_state" for event in events):
                return "state_stale"

    def run(self) -> int:
        if not self.vlm.status["successes"]:
            self.emit("run_stopped", reason="model_not_warm", keepalive=self.vlm.status)
            return 2
        try:
            runner_health = self.runner.ensure_ready()
        except RuntimeError as exc:
            self.emit("run_stopped", reason="smolvla_runner_not_ready", detail=str(exc))
            return 2
        contract = self.state.initial_contract or self.vlm.create_contract(self.state.task, [], None)
        if not contract.get("task_feasible"):
            self.emit("contract_rejected", contract=contract, keepalive=self.vlm.status)
            return 2
        self.emit(
            "contract_ready",
            run_id=self.state.run_id,
            revision=self.state.revision,
            contract=contract,
            keepalive=self.vlm.status,
            runner=runner_health,
        )

        while True:
            for step in contract["plan"]:
                if self.state.cancel_event.is_set():
                    self.emit("run_cancelled", run_id=self.state.run_id, phase="before_action")
                    return 1
                timing = self.execution_timeout(step["action"])
                deadline_seconds = float(timing["deadline_seconds"])
                try:
                    ack = self.runner.start(self.state.run_id, self.state.revision, step, deadline_seconds)
                    self.emit_runner_events(self.state.revision, step["step"])
                except RuntimeError as exc:
                    self.emit("run_stopped", reason="smolvla_runner_rejected_action", detail=str(exc), step=step)
                    return 2
                action_started_at = time.time()
                step_started_at = time.perf_counter()
                self.emit(
                    "action_ready",
                    run_id=self.state.run_id,
                    revision=self.state.revision,
                    step=step["step"],
                    action=step["action"],
                    timing=timing,
                    deadline_seconds=deadline_seconds,
                    started_at=action_started_at,
                    deadline_at=action_started_at + deadline_seconds,
                    runner_ack=ack,
                )
                execution_outcome = self.wait_for_execution_window(
                    self.state.revision,
                    step["step"],
                    deadline_seconds,
                )
                if execution_outcome == "cancelled":
                    try:
                        self.runner.cancel(
                            self.state.run_id,
                            self.state.revision,
                            step["step"],
                            "operator_cancelled",
                        )
                    except RuntimeError as exc:
                        self.emit("runner_cancel_after_operator_request_failed", detail=str(exc), step=step)
                    else:
                        self.emit_runner_events(self.state.revision, step["step"])
                    self.emit("run_cancelled", run_id=self.state.run_id, phase="during_action")
                    return 1

                self.emit(
                    "action_window_finished",
                    run_id=self.state.run_id,
                    revision=self.state.revision,
                    step=step["step"],
                    action=step["action"],
                    outcome=execution_outcome,
                    deadline_seconds=deadline_seconds,
                )

                self.emit(
                    "visual_verification_started",
                    run_id=self.state.run_id,
                    revision=self.state.revision,
                    step=step["step"],
                    action=step["action"],
                )
                assessment = self.vlm.assess_step(step)
                self.emit(
                    "step_assessment",
                    revision=self.state.revision,
                    assessment=assessment,
                    execution_outcome=execution_outcome,
                    elapsed_ms=int((time.perf_counter() - step_started_at) * 1000),
                )
                if assessment["status"] == "completed":
                    try:
                        self.runner.finish(self.state.run_id, self.state.revision, step["step"])
                        self.emit_runner_events(self.state.revision, step["step"])
                    except RuntimeError as exc:
                        self.emit("run_stopped", reason="smolvla_runner_finish_failed", detail=str(exc), step=step)
                        return 2
                    self.state.completed_actions.append(step["action"])
                    continue

                try:
                    self.runner.cancel(
                        self.state.run_id,
                        self.state.revision,
                        step["step"],
                        assessment["reason_code"],
                    )
                    self.emit_runner_events(self.state.revision, step["step"])
                except RuntimeError as exc:
                    self.emit("run_stopped", reason="smolvla_runner_cancel_failed", detail=str(exc), step=step)
                    return 2

                if self.state.revision >= self.state.max_replans:
                    self.emit("run_stopped", reason="max_replans_reached", assessment=assessment)
                    return 2
                self.state.revision += 1
                contract = self.vlm.create_contract(self.state.task, self.state.completed_actions, assessment)
                if not contract.get("task_feasible"):
                    self.emit(
                        "replan_rejected",
                        run_id=self.state.run_id,
                        revision=self.state.revision,
                        contract=contract,
                        assessment=assessment,
                    )
                    return 2
                self.emit(
                    "contract_replanned",
                    run_id=self.state.run_id,
                    revision=self.state.revision,
                    contract=contract,
                    assessment=assessment,
                )
                break
            else:
                self.emit("run_completed", run_id=self.state.run_id, completed_actions=self.state.completed_actions)
                return 0

    def emit_runner_events(self, revision: int, step: int) -> list[dict[str, Any]]:
        matched_events: list[dict[str, Any]] = []
        for event in self.runner.events():
            active = event.get("active")
            if isinstance(active, dict):
                event_ref = (active.get("run_id"), active.get("revision"), active.get("step"))
                current_ref = (self.state.run_id, revision, step)
                if event_ref != current_ref:
                    self.emit("runner_event_ignored", event_payload=event, expected=current_ref)
                    continue
            self.emit("runner_event", event_payload=event)
            matched_events.append(event)
        return matched_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Timed visual verifier and replanner for robot action contracts.")
    parser.add_argument("task", help="High-level task to plan and monitor.")
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=float(os.getenv("SUPERVISOR_STEP_TIMEOUT", "8")),
        help="Seconds allowed for each external VLA action.",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=int(os.getenv("SUPERVISOR_MAX_REPLANS", "3")),
        help="Stop after this many unsuccessful replans.",
    )
    parser.add_argument("--log", default="", help="Optional JSONL event log path.")
    args = parser.parse_args()
    if args.step_timeout < 0 or args.max_replans < 0:
        parser.error("--step-timeout and --max-replans must be non-negative")

    vlm = ResidentVLM()
    vlm.start()
    try:
        return Supervisor(
            vlm,
            RunnerClient(),
            RunState(task=args.task, step_timeout_s=args.step_timeout, max_replans=args.max_replans),
            Path(args.log) if args.log else None,
        ).run()
    finally:
        vlm.close()


if __name__ == "__main__":
    raise SystemExit(main())
