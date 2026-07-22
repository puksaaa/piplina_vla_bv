# Robot Action Contract v1

The VLM returns one JSON object. The VLA consumes only `plan[].action`; a later visual verifier uses `plan[].verification` with fresh camera frames.

```json
{
  "contract_version": "robot_action_contract.v1",
  "task_feasible": true,
  "plan": [
    {
      "step": 1,
      "action": "move red cube to white bowl",
      "verification": {
        "required_visible": ["red cube", "white bowl"],
        "success": [
          {"predicate": "object_in_target", "object": "red cube", "target": "white bowl"}
        ],
        "failure": [
          {"predicate": "object_not_in_target", "object": "red cube", "target": "white bowl"}
        ],
        "on_uncertain": "stop"
      }
    }
  ],
  "failure_reason": null
}
```

Allowed VLA actions:

- `move <visible object> to <visible object or destination>`
- `grasp <visible object>`
- `place to <visible object or destination>`
- `stop`

Allowed visual predicates:

- `object_visible(object)`
- `target_visible(target)`
- `object_not_visible(object)`
- `target_not_visible(target)`
- `object_held(object)`
- `object_not_held(object)`
- `object_in_target(object, target)`
- `object_not_in_target(object, target)`
- `object_on_target(object, target)`
- `object_not_on_target(object, target)`
- `object_touching_target(object, target)`
- `object_not_touching_target(object, target)`
- `object_dropped(object)`
- `gripper_open()`
- `gripper_closed()`

The verifier uses these meanings:

- all names in `required_visible` are what the verifier should try to see before evaluating a step;
- at least one `success` predicate must be true to mark the step complete;
- any `failure` predicate marks the step failed;
- missing or contradictory visual evidence is `uncertain`, which always means `stop`.

`object_not_visible(object)` and `target_not_visible(target)` are allowed only as visual observations for uncertainty or diagnostics. They must not be used in `verification.failure`. A failure means the action visibly produced the wrong result: the object is not held, is not in/on/touching the target, or was dropped/fell.

`grasp` requires `object_held` as a success predicate and `object_not_held` or `object_dropped` as a failure predicate. `move` and `place` require one of `object_in_target`, `object_on_target`, or `object_touching_target` for success and one of `object_not_in_target`, `object_not_on_target`, `object_not_touching_target`, or `object_dropped` for failure.

## Timed Step Assessment

After the configured action deadline, the supervisor sends fresh camera frames to the VLM and accepts only this assessment shape:

```json
{
  "assessment_version": "robot_step_assessment.v1",
  "step": 1,
  "status": "completed",
  "observed_predicates": [
    {"predicate": "object_in_target", "object": "red cube", "target": "white bowl"}
  ],
  "replan_required": false,
  "reason_code": "success_criteria_observed"
}
```

`observed_predicates` can only repeat predicate objects already present in the current step's `success` or `failure` lists. Any malformed assessment becomes `uncertain` and requests a replan. Non-completed, failed, or uncertain steps all trigger a new contract for the remaining task.
