"""Terminal-only VLM planner for the local camera splitter.

This program deliberately plans and validates only. It never opens a robot port,
loads a VLA policy, or sends motor actions.
"""

import argparse
import json
import os
import re
import sys
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, OpenAIError


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env.vlm"))

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://100.64.0.4:1234/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemma-4-31b")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "120"))
CAMERA_BASE_URL = os.getenv("CAMERA_BASE_URL", "http://127.0.0.1:8090").rstrip("/")
CAMERA_TIMEOUT = float(os.getenv("CAMERA_TIMEOUT", "10"))
ROBOT_VLM_MAX_TOKENS = int(os.getenv("ROBOT_VLM_MAX_TOKENS", "512"))
ROBOT_VLM_THINK_PREFILL = os.getenv("ROBOT_VLM_THINK_PREFILL", "<think>\n\n</think>\n\n")

FORBIDDEN_WORDS = {
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
    "distance",
    "distances",
}
CONTRACT_VERSION = "robot_action_contract.v1"
ALLOWED_PATTERNS = (
    r"move\s+.+\s+to\s+.+",
    r"grasp\s+.+",
    r"place\s+to\s+.+",
    r"stop",
)
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


def system_prompt() -> str:
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

Every non-stop step MUST include:
"verification": {
  "required_visible": ["visible object names"],
  "success": [{"predicate": "allowed predicate", "object": "when required", "target": "when required"}],
  "failure": [{"predicate": "allowed predicate", "object": "when required", "target": "when required"}],
  "on_uncertain": "stop"
}

Allowed predicates: object_visible(object), target_visible(target), object_not_visible(object), target_not_visible(target), object_held(object), object_not_held(object), object_in_target(object,target), object_not_in_target(object,target), object_on_target(object,target), object_not_on_target(object,target), object_touching_target(object,target), object_not_touching_target(object,target), object_dropped(object), gripper_open(), gripper_closed().
`required_visible` is only the list of things the verifier should look for before judging the step.
Never put object_not_visible or target_not_visible inside `failure`. If the verifier cannot see the required object or target after execution, that is `uncertain`, not `failed`.
`failure` means a visibly wrong result after the VLA tried the action: the object is not held, is not in/on/touching the target, or was dropped/fell.
For grasp, success includes object_held and failure includes object_not_held or object_dropped. For move and place, success includes object_in_target, object_on_target, or object_touching_target; failure includes object_not_in_target, object_not_on_target, object_not_touching_target, or object_dropped. Never add fields or predicates.

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

If the task is ambiguous, impossible, or a needed object is not visible, use exactly this shape:
{
  "contract_version": "robot_action_contract.v1",
  "task_feasible": false,
  "plan": [{"step": 1, "action": "stop", "verification": null}],
  "failure_reason": "not_visible_or_ambiguous"
}

JSON is the complete answer. Never add any text outside JSON.
""".strip()


def user_prompt(task: str) -> str:
    return f"TASK:\n{task}\n\nReturn the required JSON object only."


def stop_plan(reason: str) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "task_feasible": False,
        "plan": [{"step": 1, "action": "stop", "verification": None}],
        "failure_reason": reason,
    }


def normalize_action(action: str) -> str:
    return " ".join(action.strip().lower().split())


def contains_forbidden_word(action: str) -> bool:
    words = set(re.findall(r"[a-z]+", action.lower()))
    return bool(words & FORBIDDEN_WORDS)


def validate_plan(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
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
    except json.JSONDecodeError:
        return stop_plan("invalid_json")

    if (
        not isinstance(parsed, dict)
        or parsed.get("contract_version") != CONTRACT_VERSION
        or not isinstance(parsed.get("plan"), list)
        or not parsed["plan"]
    ):
        return stop_plan("invalid_schema")

    feasible = parsed.get("task_feasible")
    if not isinstance(feasible, bool):
        return stop_plan("invalid_schema")
    if not feasible:
        reason = parsed.get("failure_reason")
        return stop_plan(reason if isinstance(reason, str) and reason else "not_visible_or_ambiguous")

    normalized_steps: list[dict[str, Any]] = []
    for number, step in enumerate(parsed["plan"], start=1):
        if not isinstance(step, dict) or not isinstance(step.get("action"), str):
            return stop_plan("invalid_step")
        action = normalize_action(step["action"])
        if contains_forbidden_word(action) or not any(re.fullmatch(pattern, action) for pattern in ALLOWED_PATTERNS):
            return stop_plan("forbidden_action")
        if action == "stop":
            return stop_plan("stop_in_feasible_plan")
        verification = validate_verification(step.get("verification"), action)
        if verification is None:
            return stop_plan("invalid_verification")
        normalized_steps.append({"step": number, "action": action, "verification": verification})

    return {
        "contract_version": CONTRACT_VERSION,
        "task_feasible": True,
        "plan": normalized_steps,
        "failure_reason": None,
    }


def normalize_reference(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    reference = normalize_action(value)
    if not reference or contains_forbidden_word(reference):
        return None
    return reference


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
        fields = {"predicate", *PREDICATE_FIELDS[predicate]}
        supplied_fields = set(condition)
        if not fields.issubset(supplied_fields) or not supplied_fields.issubset({"predicate", "object", "target"}):
            return None
        item = {"predicate": predicate}
        for field in PREDICATE_FIELDS[predicate]:
            reference = normalize_reference(condition.get(field))
            if reference is None:
                return None
            item[field] = reference
        normalized.append(item)
    return normalized


def validate_verification(value: Any, action: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("on_uncertain") != "stop":
        return None
    visible = value.get("required_visible")
    if not isinstance(visible, list) or not visible:
        return None
    required_visible = [normalize_reference(item) for item in visible]
    if any(item is None for item in required_visible):
        return None
    success = validate_conditions(value.get("success"))
    failure = validate_conditions(value.get("failure"))
    if not success or not failure:
        return None
    failure = [item for item in failure if item["predicate"] not in VISIBILITY_FAILURE_PREDICATES]
    if not failure:
        failure = derive_failure_conditions(action, success)
    if not failure:
        return None
    success_predicates = {item["predicate"] for item in success}
    failure_predicates = {item["predicate"] for item in failure}
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
        "required_visible": required_visible,
        "success": success,
        "failure": failure,
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


def load_image_parts() -> list[dict[str, Any]]:
    try:
        with httpx.Client(timeout=CAMERA_TIMEOUT) as http:
            response = http.get(f"{CAMERA_BASE_URL}/snapshot/all")
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"camera_splitter_unavailable: {exc}") from exc

    frames = payload.get("frames")
    if not isinstance(frames, dict) or not frames:
        raise RuntimeError("camera_splitter_returned_no_frames")

    image_parts: list[dict[str, Any]] = []
    for name, frame in frames.items():
        if not isinstance(frame, dict) or not isinstance(frame.get("image"), str):
            continue
        image_parts.append({"type": "text", "text": f"camera: {name}"})
        image_parts.append({"type": "image_url", "image_url": {"url": frame["image"]}})
    if not image_parts:
        raise RuntimeError("camera_splitter_returned_invalid_frames")
    return image_parts


def make_plan(task: str) -> dict[str, Any]:
    image_parts = load_image_parts()
    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY, timeout=OPENAI_TIMEOUT)
    try:
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            max_tokens=ROBOT_VLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": system_prompt()},
                {"role": "user", "content": [{"type": "text", "text": user_prompt(task)}, *image_parts]},
                {"role": "assistant", "content": ROBOT_VLM_THINK_PREFILL},
            ],
        )
    except (APIConnectionError, APITimeoutError) as exc:
        raise RuntimeError(f"qwen_unavailable_or_timed_out: {exc}") from exc
    except OpenAIError as exc:
        raise RuntimeError(f"qwen_error: {exc}") from exc

    answer = completion.choices[0].message.content or ""
    return validate_plan(answer)


def run_one(task: str) -> int:
    result = make_plan(task)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["task_feasible"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Strict VLM planner using local camera splitter frames.")
    parser.add_argument("task", nargs="*", help="One robot task. Omit for interactive terminal mode.")
    args = parser.parse_args()

    if args.task:
        return run_one(" ".join(args.task))

    while True:
        try:
            task = input("task> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if task.lower() in {"exit", "quit"}:
            return 0
        if task:
            try:
                run_one(task)
            except RuntimeError as exc:
                print(json.dumps(stop_plan(str(exc)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RuntimeError as exc:
        print(json.dumps(stop_plan(str(exc)), ensure_ascii=False, indent=2))
        sys.exit(2)
