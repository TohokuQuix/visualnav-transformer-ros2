import math

import cv2
import numpy as np
import pytest
from geometry_msgs.msg import TransformStamped
from visualnav_transformer.deployment.src.robot_interface import (
    ImageHistory,
    decode_compressed,
    metric_path,
)


def test_jpeg_is_rgb():
    bgr = np.zeros((16, 16, 3), np.uint8)
    bgr[:] = (0, 0, 255)
    ok, jpeg = cv2.imencode(".jpg", bgr)
    assert ok
    rgb = np.asarray(decode_compressed(jpeg.tobytes()))
    assert rgb[8, 8, 0] > 240
    assert rgb[8, 8, 2] < 10


@pytest.mark.parametrize("data", [b"", b"not a jpeg"])
def test_bad_image(data):
    with pytest.raises(ValueError):
        decode_compressed(data)


def test_history_sampling_and_gaps():
    h = ImageHistory(3, 0.1)
    assert not h.add("a", 0)
    h.add("ignored", 50_000_000)
    h.add("b", 100_000_000)
    h.add("duplicate", 100_000_000)
    h.add("c", 200_000_000)
    assert h.snapshot() == (("a", "b", "c"), 200_000_000)
    assert h.add("gap", 400_000_000)  # reset notification
    assert h.snapshot() is None
    assert h.add("rewind", 10_000_000)
    assert h.snapshot() is None


@pytest.mark.parametrize(
    "yaw,expected",
    [
        (0, [11, 20, 3]),
        (math.pi / 2, [10, 21, 3]),
        (-math.pi / 2, [10, 19, 3]),
        (math.pi, [9, 20, 3]),
    ],
)
def test_map_transform(yaw, expected):
    tf = TransformStamped()
    tf.transform.translation.x = 10.0
    tf.transform.translation.y = 20.0
    tf.transform.translation.z = 3.0
    tf.transform.rotation.z = math.sin(yaw / 2)
    tf.transform.rotation.w = math.cos(yaw / 2)
    msg = metric_path(np.array([[2.0, 0.0], [-2.0, 0.0]]), 0.5, tf, "map", 1234567890)
    assert len(msg.poses) == 3
    p = msg.poses[1].pose.position
    np.testing.assert_allclose([p.x, p.y, p.z], expected, atol=1e-12)
    assert msg.poses[0].pose.position.x == 10
    assert msg.header.stamp.sec == 1
    assert msg.header.stamp.nanosec == 234567890
    assert all(p.header == msg.header for p in msg.poses)
    assert all(p.pose.orientation.w == 1 for p in msg.poses)
    # Reverse point is retained, not clipped or cumulatively summed twice.
    assert msg.poses[2].pose.position.x == pytest.approx(20 - expected[0])


@pytest.mark.parametrize("points", [[], [[0, 0]], [[float("nan"), 0]], [[1, 2, 3]]])
def test_invalid_trajectory(points):
    tf = TransformStamped()
    tf.transform.rotation.w = 1.0
    with pytest.raises(ValueError):
        metric_path(points, 1.0, tf, "map", 1)
