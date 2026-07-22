"""Physical actuation boundary for the resident SmolVLA runner.

This module is the ONLY place that opens the SO-101 serial port. It wraps the
LeRobot follower so the runner can:

  * read the real joint state each control tick (``read_state``), which replaces
    the previous dependency on an external ``POST /v1/state`` pusher, and
  * send a policy action to the motors (``send_action``), clamped by LeRobot's
    own ``max_relative_target`` safety limit.

Cameras are intentionally left empty here: image frames come from the ZMQ
splitter, and letting the robot open USB cameras would fight the splitter for
the devices. So the follower is created motors-only.

Everything is opt-in. If ``SMOLVLA_ACTUATION_ENABLED`` is not set the runner
never constructs this object and no serial port is ever opened.

HARDWARE VALIDATION NOTES (must be checked on the real arm before trusting):
  * ``_resolve_motor_pos_names`` derives the ordered ``<motor>.pos`` keys from
    the LeRobot robot object. The order MUST match the order the policy was
    trained on (dataset ``observation.state`` / ``action`` feature order). The
    startup log prints the resolved order loudly so it can be eyeballed.
  * LeRobot import paths and the follower config field names drift between
    versions; the version-tolerant fallbacks below cover the layouts we know of.
"""

from __future__ import annotations

from typing import Any

# Canonical SO-101 joint order, used only as a last-resort fallback when the
# installed LeRobot does not expose ordered action/observation features.
SO101_MOTOR_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _import_follower() -> tuple[Any, Any]:
    """Import ``SO101Follower`` / ``SO101FollowerConfig`` across LeRobot layouts."""

    errors: list[str] = []
    for module_path in (
        "lerobot.robots.so101_follower",
        "lerobot.common.robots.so101_follower",
    ):
        try:
            module = __import__(module_path, fromlist=["SO101Follower", "SO101FollowerConfig"])
            return module.SO101Follower, module.SO101FollowerConfig
        except Exception as exc:  # ImportError or attribute errors on partial installs
            errors.append(f"{module_path}: {exc}")
    raise RuntimeError(
        "Could not import SO101Follower from LeRobot. Tried:\n  " + "\n  ".join(errors)
    )


def _pos_names_from_features(features: Any) -> list[str]:
    """Extract ordered ``<motor>.pos`` keys from a LeRobot features mapping."""

    if not isinstance(features, dict):
        return []
    return [name for name in features if isinstance(name, str) and name.endswith(".pos")]


class RobotActuator:
    """Owns the SO-101 serial bus. All bus access happens on the runner's single
    control thread; this class is not internally locked and must not be shared
    across threads."""

    def __init__(
        self,
        *,
        robot_type: str,
        port: str,
        robot_id: str,
        max_relative_target: float | None,
        use_degrees: bool,
        actuation_enabled: bool,
    ) -> None:
        self.robot_type = robot_type
        self.port = port
        self.robot_id = robot_id
        self.max_relative_target = max_relative_target
        self.use_degrees = use_degrees
        # When False the object still connects and reads state, but never moves
        # the motors. This makes it safe to bring the full loop up and watch the
        # inference/state path before enabling motion.
        self.actuation_enabled = actuation_enabled

        self.robot: Any | None = None
        self.motor_pos_names: list[str] = []
        self.connected = False
        self.last_error: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> dict[str, Any]:
        if self.robot_type != "so101_follower":
            raise RuntimeError(
                f"Only so101_follower actuation is implemented; got SMOLVLA_ROBOT_TYPE={self.robot_type!r}."
            )
        if not self.port:
            raise RuntimeError("SMOLVLA_ROBOT_PORT is required when SMOLVLA_ACTUATION_ENABLED=1.")

        follower_cls, config_cls = _import_follower()

        # Build the config defensively: field names differ slightly across
        # LeRobot versions, so pass only the ones the installed config accepts.
        config_kwargs: dict[str, Any] = {"port": self.port, "cameras": {}}
        if self.robot_id:
            config_kwargs["id"] = self.robot_id
        if self.max_relative_target is not None:
            config_kwargs["max_relative_target"] = self.max_relative_target
        config_kwargs["use_degrees"] = self.use_degrees
        config = self._build_config(config_cls, config_kwargs)

        robot = follower_cls(config)
        robot.connect()
        self.robot = robot
        self.connected = True
        self.motor_pos_names = self._resolve_motor_pos_names(robot)
        return {
            "robot_type": self.robot_type,
            "port": self.port,
            "id": self.robot_id or None,
            "actuation_enabled": self.actuation_enabled,
            "max_relative_target": self.max_relative_target,
            "use_degrees": self.use_degrees,
            "motor_pos_names": list(self.motor_pos_names),
            "action_dimension": len(self.motor_pos_names),
        }

    @staticmethod
    def _build_config(config_cls: Any, desired: dict[str, Any]) -> Any:
        """Instantiate the follower config, dropping kwargs it does not accept."""

        try:
            return config_cls(**desired)
        except TypeError:
            pass
        # Retry with only the fields the dataclass actually declares.
        allowed: set[str] = set()
        fields = getattr(config_cls, "__dataclass_fields__", None)
        if isinstance(fields, dict):
            allowed = set(fields)
        filtered = {key: value for key, value in desired.items() if key in allowed}
        if "port" not in filtered:
            filtered["port"] = desired["port"]
        return config_cls(**filtered)

    def _resolve_motor_pos_names(self, robot: Any) -> list[str]:
        """Ordered ``<motor>.pos`` keys defining both the state and action vectors."""

        for attr in ("action_features", "observation_features"):
            names = _pos_names_from_features(getattr(robot, attr, None))
            if names:
                return names
        # Fall back to the motor bus ordering, then to the canonical SO-101 order.
        bus = getattr(robot, "bus", None)
        motors = getattr(bus, "motors", None)
        if isinstance(motors, dict) and motors:
            return [f"{name}.pos" for name in motors]
        return [f"{name}.pos" for name in SO101_MOTOR_ORDER]

    def disconnect(self) -> None:
        robot = self.robot
        self.robot = None
        self.connected = False
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as exc:  # best effort; we are usually shutting down
                self.last_error = f"disconnect failed: {exc}"

    # -- control tick ------------------------------------------------------

    def read_state(self) -> list[float]:
        """Read current joint positions as the policy ``observation.state`` vector."""

        if self.robot is None:
            raise RuntimeError("Actuator is not connected.")
        observation = self.robot.get_observation()
        if not isinstance(observation, dict):
            raise RuntimeError("Robot observation is not a mapping; cannot extract joint state.")
        try:
            return [float(observation[name]) for name in self.motor_pos_names]
        except KeyError as exc:
            raise RuntimeError(
                f"Robot observation is missing {exc}; expected keys {self.motor_pos_names}."
            ) from exc

    def send_action(self, action_values: list[float]) -> dict[str, float]:
        """Map a flat policy action vector to motor targets and command the arm.

        No-op that still returns the intended targets when actuation is disabled,
        so the caller can log what *would* have been sent during a dry run.
        """

        if len(action_values) < len(self.motor_pos_names):
            raise RuntimeError(
                f"Policy action has {len(action_values)} values, "
                f"robot needs {len(self.motor_pos_names)} ({self.motor_pos_names})."
            )
        # The policy may emit extra trailing values; the leading N map to motors
        # in the trained order.
        action = {
            name: float(action_values[index]) for index, name in enumerate(self.motor_pos_names)
        }
        if not self.actuation_enabled:
            return action
        if self.robot is None:
            raise RuntimeError("Actuator is not connected.")
        # send_action applies max_relative_target clamping inside LeRobot and may
        # return the actually-sent (clamped) targets.
        sent = self.robot.send_action(action)
        if isinstance(sent, dict):
            return {key: float(value) for key, value in sent.items() if isinstance(value, (int, float))}
        return action

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": self.robot_type or None,
            "port_configured": bool(self.port),
            "id": self.robot_id or None,
            "actuation_enabled": self.actuation_enabled,
            "actuator_connected": self.connected,
            "max_relative_target": self.max_relative_target,
            "motor_pos_names": list(self.motor_pos_names),
            "last_error": self.last_error,
        }
