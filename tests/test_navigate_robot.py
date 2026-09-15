"""Real ROS transport tests; predictor doubles isolate the robot interface."""

import threading
import time

import cv2
import numpy as np
import pytest
import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Path
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CompressedImage
from tf2_msgs.msg import TFMessage
from visualnav_transformer.deployment.src.navigate_robot import RobotNavigationNode
from visualnav_transformer.deployment.src.robot_interface import Prediction


@pytest.fixture
def rig(tmp_path):
    rclpy.init()
    model = tmp_path / "model.yaml"
    model.write_text(yaml.safe_dump({"context_size": 0, "normalize": True}))
    weights = tmp_path / "weights.pth"
    weights.touch()
    topomap = tmp_path / "topomap"
    topomap.mkdir()
    settings = dict(
        model_config=str(model),
        checkpoint=str(weights),
        topomap=str(topomap),
        context_interval_sec=0.1,
        action_scale_m=0.5,
        observation_timeout_sec=0.4,
        image_receive_timeout_sec=0.4,
        freshness_check_period_sec=0.01,
        image_topic="/test_camera",
        path_topic="/test_path",
    )
    io = rclpy.create_node("interface_test_io")
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    paths = []
    io.create_subscription(Path, "/test_path", paths.append, qos)
    image_pub = io.create_publisher(CompressedImage, "/test_camera", 1)
    tf_pub = io.create_publisher(TFMessage, "/tf_static", qos)
    dynamic_tf_pub = io.create_publisher(TFMessage, "/tf", 10)
    clock_pub = io.create_publisher(Clock, "/clock", 10)
    executor = SingleThreadedExecutor()
    executor.add_node(io)
    nodes = []
    gates = []

    def spin(seconds=0.05):
        until = time.monotonic() + seconds
        while time.monotonic() < until:
            executor.spin_once(timeout_sec=0.002)

    def start(predictor=None, overrides=None):
        predictor = predictor or (
            lambda images, closest: Prediction(
                np.array([[2.0, 0.0], [4.0, 0.0]]), 0, False
            )
        )
        node = RobotNavigationNode(
            settings,
            predictor_factory=lambda **kwargs: predictor,
            parameter_overrides=overrides or [],
        )
        nodes.append(node)
        executor.add_node(node)
        tr = TransformStamped()
        tr.header.frame_id = "map"
        tr.child_frame_id = "chassis/body"
        tr.transform.translation.x = 10.0
        tr.transform.rotation.w = 1.0
        tf_pub.publish(TFMessage(transforms=[tr]))
        spin(0.15)
        return node

    def image(node, stamp=None):
        msg = CompressedImage()
        msg.header.stamp = stamp or node.get_clock().now().to_msg()
        msg.format = "rgb8; jpeg compressed bgr8"
        msg.data = cv2.imencode(".jpg", np.zeros((16, 16, 3), np.uint8))[1].tobytes()
        image_pub.publish(msg)
        return msg.header.stamp

    yield settings, paths, start, image, spin, clock_pub, gates, dynamic_tf_pub
    for gate in gates:
        gate.set()
    for node in nodes:
        executor.remove_node(node)
        node.close()
        node.destroy_node()
    executor.remove_node(io)
    io.destroy_node()
    executor.shutdown()
    rclpy.shutdown()


def test_ros_metric_path_and_timeout(rig):
    settings, paths, start, image, spin, _, _, _ = rig
    node = start()
    stamp = image(node)
    spin(0.15)
    msg = paths[-1]
    assert [p.pose.position.x for p in msg.poses] == [10.0, 11.0, 12.0]
    assert msg.header.frame_id == "map"
    assert msg.header.stamp == stamp
    spin(0.4)
    assert not paths[-1].poses
    image(node)
    spin(0.15)
    assert paths[-1].poses


def test_timeout_while_inference_blocked_discards_late_result(rig):
    _, paths, start, image, spin, _, gates, _ = rig
    gate = threading.Event()
    gates.append(gate)

    def slow(images, closest):
        gate.wait(3)
        return Prediction(np.array([[1.0, 0.0]]), 0, False)

    node = start(slow)
    image(node)
    spin(0.6)
    assert paths and not paths[-1].poses
    gate.set()
    spin(0.1)
    assert not any(p.poses for p in paths)


def test_goal_is_terminal(rig):
    _, paths, start, image, spin, _, _, _ = rig
    count = []

    def goal(images, closest):
        count.append(1)
        return Prediction(np.array([[1.0, 0.0]]), 1, True)

    node = start(goal)
    image(node)
    spin(0.15)
    image(node)
    spin(0.15)
    assert len(count) == 1
    assert paths and all(not p.poses for p in paths)


def test_invalid_output_and_missing_tf_stop(rig):
    settings, paths, start, image, spin, _, _, _ = rig
    settings["global_frame"] = "missing"
    node = start()
    image(node)
    spin(0.15)
    assert paths and not paths[-1].poses


def test_paused_ros_clock_receipt_timeout(rig):
    _, paths, start, image, spin, clocks, _, _ = rig
    from rclpy.parameter import Parameter

    node = start(overrides=[Parameter("use_sim_time", value=True)])
    c = Clock()
    c.clock.sec = 10
    clocks.publish(c)
    spin(0.05)
    image(node)
    spin(0.15)
    assert paths[-1].poses
    spin(0.4)
    assert not paths[-1].poses


def test_required_scale_is_not_guessed(rig):
    settings, _, start, _, _, _, _, _ = rig
    settings["action_scale_m"] = None
    with pytest.raises(ValueError, match="action_scale_m"):
        start()


def test_transform_uses_observation_stamp_not_latest(rig):
    settings, paths, start, image, spin, _, _, transforms = rig
    settings["robot_frame"] = "dynamic_body"
    node = start()
    stamp_ns = node.get_clock().now().nanoseconds
    data = []
    for offset, x in [(-100_000_000, 10.0), (0, 20.0), (100_000_000, 100.0)]:
        tr = TransformStamped()
        tr.header.frame_id = "map"
        tr.child_frame_id = "dynamic_body"
        tr.header.stamp.sec, tr.header.stamp.nanosec = divmod(stamp_ns + offset, 10**9)
        tr.transform.translation.x = x
        tr.transform.rotation.w = 1.0
        data.append(tr)
    transforms.publish(TFMessage(transforms=data))
    spin(0.04)
    image(node, data[1].header.stamp)
    spin(0.1)
    assert paths[-1].poses[0].pose.position.x == 20.0
    assert paths[-1].poses[1].pose.position.x == 21.0


def test_gap_discards_old_goal_result(rig):
    _, paths, start, image, spin, _, gates, _ = rig
    gate = threading.Event()
    gates.append(gate)
    calls = []

    def predict(images, closest):
        calls.append(1)
        if len(calls) == 1:
            gate.wait(3)
            return Prediction(np.array([[1.0, 0.0]]), 2, True)
        return Prediction(np.array([[1.0, 0.0]]), 0, False)

    node = start(predict)
    image(node)
    spin(0.2)  # exceeds the history gap threshold, but not the watchdog
    image(node)
    spin(0.02)
    gate.set()
    spin(0.12)
    assert paths[-1].poses
    assert not node.terminal


def test_clock_rewind_discards_inflight_result(rig):
    _, paths, start, image, spin, clocks, gates, _ = rig
    from rclpy.parameter import Parameter

    gate = threading.Event()
    gates.append(gate)
    calls = []

    def predict(images, closest):
        calls.append(1)
        if len(calls) == 1:
            gate.wait(3)
            return Prediction(np.array([[1.0, 0.0]]), 2, True)
        return Prediction(np.array([[1.0, 0.0]]), 0, False)

    node = start(predict, [Parameter("use_sim_time", value=True)])
    c = Clock()
    c.clock.sec = 10
    clocks.publish(c)
    spin(0.03)
    image(node)
    spin(0.04)
    c.clock.sec = 9
    clocks.publish(c)
    spin(0.04)
    assert not paths[-1].poses
    image(node)
    gate.set()
    spin(0.15)
    assert paths[-1].poses
    assert paths[-1].header.stamp.sec == 9


def test_nonfinite_inference_output_stops(rig):
    _, paths, start, image, spin, _, _, _ = rig
    node = start(
        lambda images, closest: Prediction(np.array([[np.nan, 0.0]]), 0, False)
    )
    image(node)
    spin(0.1)
    assert paths and all(not p.poses for p in paths)


def test_close_sends_empty_path(rig):
    _, paths, start, image, spin, _, _, _ = rig
    node = start()
    image(node)
    spin(0.1)
    assert paths[-1].poses
    node.close()
    spin(0.03)
    assert not paths[-1].poses


def test_published_path_expires_even_while_images_arrive(rig):
    _, paths, start, image, spin, _, gates, _ = rig
    gate = threading.Event()
    gates.append(gate)
    calls = []

    def predict(images, closest):
        calls.append(1)
        if len(calls) > 1:
            gate.wait(3)
        return Prediction(np.array([[1.0, 0.0]]), 0, False)

    node = start(predict)
    image(node)
    spin(0.1)
    assert paths[-1].poses
    for _ in range(14):
        image(node)
        spin(0.04)
    assert not paths[-1].poses


def test_reset_does_not_reinfer_same_observation(rig):
    _, paths, start, image, spin, _, _, _ = rig
    calls = []

    def fail(images, closest):
        calls.append(1)
        raise ValueError("failed prediction")

    node = start(fail)
    stamp = image(node)
    spin(0.08)
    assert len(calls) == 1
    image(node, stamp)
    spin(0.08)
    assert len(calls) == 1


def test_goal_image_settings_resolve_relative_path(tmp_path):
    from visualnav_transformer.deployment.src.navigate_robot import load_settings

    cfg = tmp_path / "robot.yaml"
    cfg.write_text("goal_image: goal.jpg\n")
    assert load_settings(cfg)["goal_image"] == str(tmp_path / "goal.jpg")
