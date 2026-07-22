import contextlib
import io
import threading
import time
import unittest

import orchestrator as orchestrator_module
from smolvla_runner import (
    ActionRefRequest,
    ActionStartRequest,
    CameraExpectation,
    StateExpectation,
    build_dataset_features,
    cancel_action,
    extract_camera_expectations,
    extract_state_expectation,
    pause_action,
    resume_action,
    runner,
    start_action,
    update_state,
    validate_camera_expectations,
    StateUpdateRequest,
)
from supervisor import RunState, Supervisor
from orchestrator import OrchestratorService


def contract(action: str) -> dict:
    return {
        "task_feasible": True,
        "plan": [
            {
                "step": 1,
                "action": action,
                "verification": {
                    "required_visible": ["red cube"],
                    "success": [{"predicate": "object_held", "object": "red cube"}],
                    "failure": [{"predicate": "object_not_held", "object": "red cube"}],
                    "on_uncertain": "stop",
                },
            }
        ],
    }


class FakeVLM:
    def __init__(self) -> None:
        self.status = {"successes": 1}
        self.contracts = [contract("grasp red cube"), contract("grasp red cube")]
        self.assessments = [
            {
                "status": "not_completed",
                "reason_code": "success_criteria_not_observed",
                "observed_predicates": [],
                "replan_required": True,
            },
            {
                "status": "completed",
                "reason_code": "success_criteria_observed",
                "observed_predicates": [{"predicate": "object_held", "object": "red cube"}],
                "replan_required": False,
            },
        ]

    def create_contract(self, *_args):
        return self.contracts.pop(0)

    def assess_step(self, _step):
        return self.assessments.pop(0)


class FakeRunner:
    def __init__(self) -> None:
        self.starts = []
        self.cancels = []
        self.finishes = []
        self.event_calls = 0

    def ensure_ready(self):
        return {"state": "ready", "policy_loaded": True}

    def start(self, run_id, revision, step, _deadline):
        self.starts.append((run_id, revision, step["step"]))
        return {"accepted": True, "active": {"run_id": run_id, "revision": revision, "step": step["step"], "action": step["action"]}}

    def cancel(self, run_id, revision, step, reason):
        self.cancels.append((run_id, revision, step, reason))
        return {"cancelled": True}

    def finish(self, run_id, revision, step):
        self.finishes.append((run_id, revision, step))
        return {"finished": True}

    def events(self):
        self.event_calls += 1
        if self.event_calls == 1:
            return [{"event": "action_accepted", "active": {"run_id": "old", "revision": 99, "step": 1}}]
        return []


class SmolVLAIntegrationTests(unittest.TestCase):
    def test_runner_waits_for_operator_approval(self):
        runner_started = threading.Event()

        class ImmediateVLM:
            status = {"successes": 1}

            def create_contract(self, *_args):
                return contract("grasp red cube")

        class FakeSupervisor:
            def __init__(self, *_args):
                pass

            def run(self):
                runner_started.set()
                return 0

        original_supervisor = orchestrator_module.Supervisor
        original_runner_client = orchestrator_module.RunnerClient
        orchestrator_module.Supervisor = FakeSupervisor
        orchestrator_module.RunnerClient = lambda: object()
        try:
            service = OrchestratorService()
            service.vlm = ImmediateVLM()
            record = service.create_run("grasp red cube")
            deadline = time.monotonic() + 1
            while record.status == "planning" and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertEqual("awaiting_approval", record.status)
            self.assertFalse(runner_started.is_set())
            service.approve_run(record.state.run_id)
            record.thread.join(timeout=2)
            self.assertTrue(runner_started.is_set())
            self.assertEqual("completed", record.status)
        finally:
            orchestrator_module.Supervisor = original_supervisor
            orchestrator_module.RunnerClient = original_runner_client

    def test_operator_cancel_during_vlm_planning_prevents_runner_start(self):
        class BlockingVLM:
            def __init__(self):
                self.status = {"successes": 1}
                self.started = threading.Event()
                self.release = threading.Event()

            def create_contract(self, *_args):
                self.started.set()
                self.release.wait(timeout=2)
                return contract("grasp red cube")

        service = OrchestratorService()
        vlm = BlockingVLM()
        service.vlm = vlm

        record = service.create_run("grasp red cube")
        self.assertEqual("planning", record.status)
        self.assertTrue(vlm.started.wait(timeout=1))
        service.cancel_run(record.state.run_id)
        vlm.release.set()
        record.thread.join(timeout=2)

        self.assertEqual("cancelled", record.status)
        self.assertEqual(1, record.exit_code)
        self.assertEqual([], record.state.history)

    def test_policy_camera_metadata_is_authoritative(self):
        config = type(
            "Config",
            (),
            {
                "input_features": {
                    "observation.images.camera1": {"shape": [3, 480, 640]},
                    "observation.images.camera2": {"shape": [3, 480, 640]},
                }
            },
        )()
        expected = extract_camera_expectations(config)
        self.assertEqual([], validate_camera_expectations(expected, ["camera1", "camera2"], 640, 480))
        self.assertTrue(validate_camera_expectations(expected, ["camera1", "wrist"], 640, 480))
        self.assertTrue(validate_camera_expectations(expected, ["camera1", "camera2"], 1280, 480))

    def test_runner_builds_policy_input_schema_from_metadata(self):
        config = type(
            "Config",
            (),
            {
                "input_features": {
                    "observation.images.camera1": {"shape": [3, 480, 640]},
                    "observation.state": {"shape": [6]},
                }
            },
        )()
        state = extract_state_expectation(config)
        self.assertEqual(6, state.dimension)
        features = build_dataset_features(extract_camera_expectations(config), state)
        self.assertEqual((480, 640, 3), features["observation.images.camera1"]["shape"])
        self.assertEqual(["state_0", "state_1", "state_2", "state_3", "state_4", "state_5"], features["observation.state"]["names"])
        self.assertEqual({}, build_dataset_features([], StateExpectation(None)))

    def test_failed_step_creates_one_new_revision_and_ignores_stale_event(self):
        runner = FakeRunner()
        state = RunState(task="grasp red cube", step_timeout_s=0, max_replans=2, run_id="run-1")
        supervisor = Supervisor(FakeVLM(), runner, state, None)

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(0, supervisor.run())
        self.assertEqual([("run-1", 0, 1), ("run-1", 1, 1)], runner.starts)
        self.assertEqual([("run-1", 0, 1, "success_criteria_not_observed")], runner.cancels)
        self.assertEqual([("run-1", 1, 1)], runner.finishes)
        self.assertIn("runner_event_ignored", [event["event"] for event in state.history])

    def test_cancelled_run_does_not_submit_the_next_vla_step(self):
        runner = FakeRunner()
        state = RunState(
            task="grasp red cube",
            step_timeout_s=0,
            max_replans=2,
            run_id="run-cancelled",
            initial_contract=contract("grasp red cube"),
        )
        state.cancel_event.set()
        supervisor = Supervisor(FakeVLM(), runner, state, None)

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(1, supervisor.run())
        self.assertEqual([], runner.starts)
        self.assertIn("run_cancelled", [event["event"] for event in state.history])

    def test_action_windows_include_grace_and_keep_move_longer_than_grasp(self):
        state = RunState(task="test", step_timeout_s=8, max_replans=1)
        supervisor = Supervisor(FakeVLM(), FakeRunner(), state, None)
        grasp = supervisor.execution_timeout("grasp red cube")
        move = supervisor.execution_timeout("move red cube to white plate")
        place = supervisor.execution_timeout("place to white plate")
        self.assertGreater(float(grasp["deadline_seconds"]), float(grasp["base_seconds"]))
        self.assertGreater(float(move["deadline_seconds"]), float(grasp["deadline_seconds"]))
        self.assertGreaterEqual(float(place["deadline_seconds"]), 8)

    def test_runner_supports_pause_resume_and_matching_cancel(self):
        class Frame:
            shape = (480, 640, 3)

        class Camera:
            def read_latest(self, **_kwargs):
                return Frame()

        with runner.lock:
            runner.state = "ready"
            runner.error = None
            runner.policy = object()
            runner.cameras = {"camera1": Camera()}
            runner.active = None
            runner.task_generation = 0
            runner.latest_state = None
            runner.latest_state_at = None
            runner.events.clear()
            runner.event_id = 0
            runner.state_expectation = StateExpectation(2)
        start = ActionStartRequest(
            run_id="run-1", revision=0, step=1, action="grasp red cube", deadline_seconds=1
        )
        self.assertTrue(update_state(StateUpdateRequest(state=[0.1, 0.2]))["accepted"])
        self.assertTrue(start_action(start)["accepted"])
        self.assertEqual(1, runner.snapshot()["task_generation"])
        self.assertTrue(runner.snapshot()["state_available"])
        ref = ActionRefRequest(run_id="run-1", revision=0, step=1)
        self.assertTrue(pause_action(ref)["paused"])
        self.assertEqual("paused", runner.snapshot()["state"])
        self.assertTrue(resume_action(ref)["resumed"])
        self.assertEqual("running", runner.snapshot()["state"])
        self.assertTrue(cancel_action(ref)["cancelled"])
        self.assertEqual("ready", runner.snapshot()["state"])


if __name__ == "__main__":
    unittest.main()
