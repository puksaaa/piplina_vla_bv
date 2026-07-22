"""Unit tests for the SO-101 actuation boundary mapping logic.

These use fakes and never touch hardware or LeRobot, so they run anywhere and
guard the version-sensitive parts: motor-order resolution, the state vector,
and the action -> motor-target mapping (including the dry-run no-motion path).
"""

import unittest

from robot_actuator import SO101_MOTOR_ORDER, RobotActuator


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StrictConfig:
    """A config that only accepts ``port`` and ``id`` (older LeRobot layout)."""

    __dataclass_fields__ = {"port": None, "id": None}

    def __init__(self, port, id=None):
        self.port = port
        self.id = id


class FakeRobot:
    def __init__(self, features=None, observation=None):
        self.action_features = features or {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
        }
        self.observation_features = dict(self.action_features)
        self.observation_features["observation.images.camera1"] = object
        self._observation = observation
        self.sent = []
        self.disconnected = False

    def get_observation(self):
        if self._observation is not None:
            return self._observation
        return {name: 10.0 + index for index, name in enumerate(self.action_features)}

    def send_action(self, action):
        self.sent.append(action)
        return action

    def disconnect(self):
        self.disconnected = True


def make_actuator(actuation_enabled=False):
    return RobotActuator(
        robot_type="so101_follower",
        port="/dev/ttyACM0",
        robot_id="arm",
        max_relative_target=5.0,
        use_degrees=True,
        actuation_enabled=actuation_enabled,
    )


class RobotActuatorTests(unittest.TestCase):
    def test_motor_order_from_action_features(self):
        actuator = make_actuator()
        names = actuator._resolve_motor_pos_names(FakeRobot())
        self.assertEqual(
            names,
            [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            ],
        )

    def test_motor_order_excludes_camera_keys(self):
        actuator = make_actuator()
        robot = FakeRobot(features={})
        robot.action_features = {}
        robot.observation_features = {"gripper.pos": float, "observation.images.camera1": object}
        self.assertEqual(actuator._resolve_motor_pos_names(robot), ["gripper.pos"])

    def test_motor_order_falls_back_to_canonical(self):
        actuator = make_actuator()

        class Bare:
            pass

        self.assertEqual(
            actuator._resolve_motor_pos_names(Bare()),
            [f"{name}.pos" for name in SO101_MOTOR_ORDER],
        )

    def test_read_state_orders_values_by_motor(self):
        actuator = make_actuator()
        robot = FakeRobot(
            observation={
                "gripper.pos": 6.0,
                "shoulder_pan.pos": 1.0,
                "shoulder_lift.pos": 2.0,
                "elbow_flex.pos": 3.0,
                "wrist_flex.pos": 4.0,
                "wrist_roll.pos": 5.0,
            }
        )
        actuator.robot = robot
        actuator.motor_pos_names = actuator._resolve_motor_pos_names(robot)
        self.assertEqual(actuator.read_state(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_read_state_missing_key_raises(self):
        actuator = make_actuator()
        robot = FakeRobot(observation={"shoulder_pan.pos": 1.0})
        actuator.robot = robot
        actuator.motor_pos_names = ["shoulder_pan.pos", "gripper.pos"]
        with self.assertRaises(RuntimeError):
            actuator.read_state()

    def test_send_action_dry_run_maps_without_moving(self):
        actuator = make_actuator(actuation_enabled=False)
        robot = FakeRobot()
        actuator.robot = robot
        actuator.motor_pos_names = actuator._resolve_motor_pos_names(robot)
        result = actuator.send_action([1, 2, 3, 4, 5, 6])
        self.assertEqual(result["shoulder_pan.pos"], 1.0)
        self.assertEqual(result["gripper.pos"], 6.0)
        # Dry run must not command the robot.
        self.assertEqual(robot.sent, [])

    def test_send_action_enabled_commands_robot(self):
        actuator = make_actuator(actuation_enabled=True)
        robot = FakeRobot()
        actuator.robot = robot
        actuator.motor_pos_names = actuator._resolve_motor_pos_names(robot)
        actuator.send_action([1, 2, 3, 4, 5, 6])
        self.assertEqual(len(robot.sent), 1)
        self.assertEqual(robot.sent[0]["shoulder_pan.pos"], 1.0)
        self.assertEqual(robot.sent[0]["gripper.pos"], 6.0)

    def test_send_action_uses_leading_values_when_action_is_longer(self):
        actuator = make_actuator(actuation_enabled=True)
        robot = FakeRobot()
        actuator.robot = robot
        actuator.motor_pos_names = actuator._resolve_motor_pos_names(robot)
        actuator.send_action([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(robot.sent[0]), 6)
        self.assertEqual(robot.sent[0]["gripper.pos"], 6.0)

    def test_send_action_too_short_raises(self):
        actuator = make_actuator(actuation_enabled=True)
        robot = FakeRobot()
        actuator.robot = robot
        actuator.motor_pos_names = actuator._resolve_motor_pos_names(robot)
        with self.assertRaises(RuntimeError):
            actuator.send_action([1, 2, 3])

    def test_build_config_drops_unknown_kwargs(self):
        actuator = make_actuator()
        config = actuator._build_config(
            StrictConfig,
            {"port": "/dev/ttyACM0", "id": "arm", "cameras": {}, "max_relative_target": 5.0},
        )
        self.assertEqual(config.port, "/dev/ttyACM0")
        self.assertEqual(config.id, "arm")

    def test_build_config_passes_all_when_accepted(self):
        actuator = make_actuator()
        config = actuator._build_config(FakeConfig, {"port": "/dev/ttyACM0", "cameras": {}})
        self.assertEqual(config.kwargs["port"], "/dev/ttyACM0")
        self.assertEqual(config.kwargs["cameras"], {})

    def test_wrong_robot_type_rejected_on_connect(self):
        actuator = RobotActuator(
            robot_type="ur5",
            port="/dev/ttyACM0",
            robot_id="arm",
            max_relative_target=None,
            use_degrees=True,
            actuation_enabled=True,
        )
        with self.assertRaises(RuntimeError):
            actuator.connect()


if __name__ == "__main__":
    unittest.main()
