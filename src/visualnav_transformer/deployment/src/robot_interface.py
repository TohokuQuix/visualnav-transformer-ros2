"""Image and coordinate adapters; no model or inference dependencies."""

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from PIL import Image


def decode_compressed(data):
    if not data:
        raise ValueError("Empty compressed image")
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode compressed image")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


class ImageHistory:
    def __init__(self, count, interval_sec):
        if count < 1 or not math.isfinite(interval_sec) or interval_sec <= 0:
            raise ValueError("Image count and interval must be positive")
        self.count = count
        self.interval_ns = max(1, round(interval_sec * 1e9))
        self.frames = deque(maxlen=count)
        self.last_seen = None

    def clear(self):
        self.frames.clear()
        self.last_seen = None

    def add(self, image, stamp_ns):
        """Return True if a gap/rewind invalidated the previous history."""
        reset = False
        if self.last_seen is not None and stamp_ns < self.last_seen:
            self.clear()
            reset = True
        if self.last_seen == stamp_ns:
            return reset
        self.last_seen = stamp_ns
        if self.frames:
            elapsed = stamp_ns - self.frames[-1][1]
            if elapsed > self.interval_ns * 1.5:
                self.frames.clear()
                reset = True
            elif elapsed < self.interval_ns:
                return reset
        self.frames.append((image, stamp_ns))
        return reset

    def snapshot(self):
        if len(self.frames) != self.count:
            return None
        return tuple(frame[0] for frame in self.frames), self.frames[-1][1]


def empty_path(frame, stamp_ns):
    msg = Path()
    msg.header.frame_id = frame
    msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(stamp_ns), 10**9)
    return msg


def metric_path(points, scale_m, transform, frame, stamp_ns):
    """Scale cumulative local XY once, then apply translation + yaw only."""
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) == 0
        or not np.isfinite(points).all()
        or not math.isfinite(scale_m)
        or scale_m <= 0
    ):
        raise ValueError("Invalid trajectory or metric scale")
    local = np.vstack(([0.0, 0.0], points * scale_m))
    if not np.isfinite(local).all() or not np.any(
        np.linalg.norm(np.diff(local, axis=0), axis=1) > 1e-6
    ):
        raise ValueError("Trajectory has no nonzero segment")
    tr = transform.transform.translation
    q = transform.transform.rotation
    values = np.array([tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w])
    if not np.isfinite(values).all() or not frame or stamp_ns < 0:
        raise ValueError("Invalid transform, frame or stamp")
    norm = np.linalg.norm(values[3:])
    if norm < 1e-9:
        raise ValueError("Invalid transform quaternion")
    x, y, z, w = values[3:] / norm
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = math.cos(yaw), math.sin(yaw)
    xy = local @ np.array([[c, s], [-s, c]]) + [tr.x, tr.y]
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite map coordinates")
    msg = empty_path(frame, stamp_ns)
    for px, py in xy:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position.x = float(px)
        pose.pose.position.y = float(py)
        pose.pose.position.z = tr.z
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


@dataclass(frozen=True)
class Prediction:
    points: np.ndarray
    closest_node: int
    reached_goal: bool
