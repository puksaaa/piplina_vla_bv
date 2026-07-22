"""HTTP orchestrator for VLM contracts and the resident SmolVLA inference runner.

This service owns one run at a time. It asks the VLM for a contract, submits a
single step to the resident runner, waits for visual verification, and only
then advances or replans. It never opens a robot port or emits motor commands.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env.vlm")
load_dotenv(BASE_DIR / ".env.smolvla")
load_dotenv(BASE_DIR / ".env.orchestrator")

from supervisor import ResidentVLM, RunnerClient, RunState, Supervisor

LOG_DIR = BASE_DIR / "logs" / "orchestrator"
APPROVAL_TIMEOUT_SECONDS = float(os.getenv("ORCHESTRATOR_APPROVAL_TIMEOUT_SECONDS", "300"))


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=500)


@dataclass
class RunRecord:
    state: RunState
    status: str = "planning"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    error: str | None = None
    contract: dict[str, Any] | None = None
    approved_at: float | None = None
    approval_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)

    def snapshot(self, include_events: bool = False) -> dict[str, Any]:
        with self.state.history_lock:
            events = list(self.state.history)
        result: dict[str, Any] = {
            "run_id": self.state.run_id,
            "task": self.state.task,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "approved_at": self.approved_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "revision": self.state.revision,
            "completed_actions": list(self.state.completed_actions),
            "contract": self.contract,
            "event_count": len(events),
        }
        if include_events:
            result["events"] = events
        return result


class OrchestratorService:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.vlm = ResidentVLM()
        self.records: dict[str, RunRecord] = {}
        self.active_run_id: str | None = None

    def start(self) -> None:
        self.vlm.start()

    def close(self) -> None:
        self.vlm.close()

    def health(self) -> dict[str, Any]:
        try:
            runner = RunnerClient().ensure_ready()
        except RuntimeError as exc:
            runner = {"ready": False, "error": str(exc)}
        with self.lock:
            active = self.active_run_id
        return {
            "status": "ok",
            "active_run_id": active,
            "vlm_keepalive": dict(self.vlm.status),
            "runner": runner,
            "execution_mode": "inference_only",
        }

    def create_run(self, task: str) -> RunRecord:
        task = " ".join(task.strip().split())
        with self.lock:
            if self.active_run_id is not None:
                active = self.records.get(self.active_run_id)
                if active is not None and active.status in {
                    "planning",
                    "awaiting_approval",
                    "starting",
                    "running",
                    "cancelling",
                }:
                    raise RuntimeError(f"run_already_active:{active.state.run_id}")
            state = RunState(
                task=task,
                step_timeout_s=float(os.getenv("SUPERVISOR_STEP_TIMEOUT", "8")),
                max_replans=int(os.getenv("SUPERVISOR_MAX_REPLANS", "3")),
                run_id=uuid.uuid4().hex,
            )
            record = RunRecord(state=state)
            self.records[state.run_id] = record
            self.active_run_id = state.run_id

        if not self.vlm.status["successes"]:
            self._finish(record, status="failed", exit_code=2, error="vlm_not_warm")
            return record

        record.started_at = time.time()
        record.thread = threading.Thread(
            target=self._plan_and_run,
            args=(record,),
            name=f"orchestrator-{state.run_id[:8]}",
            daemon=True,
        )
        record.thread.start()
        return record

    def get_run(self, run_id: str) -> RunRecord:
        with self.lock:
            record = self.records.get(run_id)
        if record is None:
            raise KeyError(run_id)
        return record

    def cancel_run(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        if record.status not in {"planning", "awaiting_approval", "starting", "running", "cancelling"}:
            return record
        record.status = "cancelling"
        record.state.cancel_event.set()
        record.approval_event.set()
        return record

    def approve_run(self, run_id: str) -> RunRecord:
        record = self.get_run(run_id)
        with self.lock:
            if record.status in {"starting", "running"}:
                return record
            if record.status != "awaiting_approval":
                raise RuntimeError(f"run_not_awaiting_approval:{record.status}")
            record.approved_at = time.time()
            record.status = "starting"
            record.approval_event.set()
        return record

    def _wait_for_approval(self, record: RunRecord) -> bool:
        timeout = APPROVAL_TIMEOUT_SECONDS if APPROVAL_TIMEOUT_SECONDS > 0 else None
        started = time.monotonic()
        while True:
            if record.state.cancel_event.is_set():
                return False
            remaining = None if timeout is None else timeout - (time.monotonic() - started)
            if remaining is not None and remaining <= 0:
                self._finish(record, status="failed", exit_code=2, error="operator_approval_timeout")
                return False
            if record.approval_event.wait(timeout=0.2 if remaining is None else min(0.2, remaining)):
                return not record.state.cancel_event.is_set()

    def _plan_and_run(self, record: RunRecord) -> None:
        try:
            if record.state.cancel_event.is_set():
                self._finish(record, status="cancelled", exit_code=1)
                return

            contract = self.vlm.create_contract(record.state.task, [], None)
            record.contract = contract
            if record.state.cancel_event.is_set():
                self._finish(record, status="cancelled", exit_code=1)
                return
            if not contract.get("task_feasible"):
                self._finish(record, status="rejected", exit_code=2, error=contract.get("failure_reason"))
                return

            record.state.initial_contract = contract
            record.status = "awaiting_approval"
            if not self._wait_for_approval(record):
                if record.finished_at is None:
                    self._finish(record, status="cancelled", exit_code=1)
                return
            record.status = "running"
            exit_code = Supervisor(
                self.vlm,
                RunnerClient(),
                record.state,
                LOG_DIR / f"{record.state.run_id}.jsonl",
            ).run()
            status = "completed" if exit_code == 0 else "cancelled" if record.state.cancel_event.is_set() else "failed"
            self._finish(record, status=status, exit_code=exit_code)
        except Exception as exc:
            self._finish(record, status="failed", exit_code=2, error=str(exc))

    def _finish(self, record: RunRecord, status: str, exit_code: int, error: str | None = None) -> None:
        record.status = status
        record.exit_code = exit_code
        record.error = error
        record.finished_at = time.time()
        with self.lock:
            if self.active_run_id == record.state.run_id:
                self.active_run_id = None


service = OrchestratorService()
app = FastAPI(title="VLM to SmolVLA Orchestrator")


@app.get("/health")
def health() -> dict[str, Any]:
    return service.health()


@app.post("/v1/runs")
def create_run(request: RunRequest) -> dict[str, Any]:
    try:
        record = service.create_run(request.task)
    except RuntimeError as exc:
        if str(exc).startswith("run_already_active:"):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return record.snapshot(include_events=True)


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return service.get_run(run_id).snapshot(include_events=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


@app.post("/v1/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    try:
        return service.cancel_run(run_id).snapshot(include_events=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


@app.post("/v1/runs/{run_id}/approve")
def approve_run(run_id: str) -> dict[str, Any]:
    try:
        return service.approve_run(run_id).snapshot(include_events=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.on_event("startup")
def startup() -> None:
    service.start()


@app.on_event("shutdown")
def shutdown() -> None:
    service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM to resident SmolVLA orchestrator.")
    parser.add_argument("--host", default=os.getenv("ORCHESTRATOR_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ORCHESTRATOR_PORT", "8092")))
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
