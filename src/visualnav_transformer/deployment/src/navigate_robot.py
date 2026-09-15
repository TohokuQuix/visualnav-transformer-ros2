"""ROS transport around unchanged NoMaD inference. No velocity publisher."""

import argparse
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path as FilePath

import rclpy
import yaml
from nav_msgs.msg import Path
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage
from tf2_ros import Buffer, TransformListener
from visualnav_transformer.deployment.src.robot_interface import (
    ImageHistory,
    decode_compressed,
    empty_path,
    metric_path,
)


class RobotNavigationNode(Node):
    def __init__(self, settings, predictor_factory=None, **kwargs):
        super().__init__("nomad_robot", **kwargs)
        self.pool = None
        try:
            self._configure(settings, predictor_factory)
        except Exception:
            self.destroy_node()
            raise

    def _configure(self, settings, predictor_factory):
        defaults = dict(
            image_topic="/front_camera/image_raw/compressed",
            path_topic="/teaching_playback/path",
            robot_frame="chassis/body",
            global_frame="map",
            observation_timeout_sec=0.8,
            image_receive_timeout_sec=0.8,
            freshness_check_period_sec=0.05,
            context_interval_sec=0.0,
            action_scale_m=0.0,
            model_config="",
            checkpoint="",
            topomap="",
            goal_node=-1,
            radius=4,
            close_threshold=3.0,
            num_samples=8,
            device="auto",
        )
        descriptor = ParameterDescriptor(read_only=True)
        self.cfg = {}
        for name, default in defaults.items():
            value = settings.get(name, default)
            if value is None:
                raise ValueError(f"{name} must be configured")
            if isinstance(default, float):
                value = float(value)
            self.cfg[name] = self.declare_parameter(name, value, descriptor).value
        c = self.cfg
        for name in (
            "context_interval_sec",
            "action_scale_m",
            "observation_timeout_sec",
            "image_receive_timeout_sec",
            "freshness_check_period_sec",
        ):
            if not math.isfinite(c[name]) or c[name] <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("image_topic", "path_topic", "robot_frame", "global_frame"):
            if not c[name]:
                raise ValueError(f"{name} must not be empty")
        if (
            c["radius"] < 0
            or c["num_samples"] < 1
            or not math.isfinite(c["close_threshold"])
        ):
            raise ValueError("Invalid radius, sample count or goal threshold")
        for name in ("model_config", "checkpoint"):
            if not FilePath(c[name]).is_file():
                raise ValueError(f"{name} must name an existing file")
        if not FilePath(c["topomap"]).is_dir():
            raise ValueError("topomap must name an existing directory")
        with open(c["model_config"]) as stream:
            model_cfg = yaml.safe_load(stream)
        if not model_cfg["normalize"] and c["action_scale_m"] != 1.0:
            raise ValueError("action_scale_m must be 1 for an unnormalized model")
        self.history = ImageHistory(
            model_cfg["context_size"] + 1, c["context_interval_sec"]
        )
        if predictor_factory is None:
            from visualnav_transformer.deployment.src.robot_inference import (
                NoMaDPredictor,
            )

            predictor_factory = NoMaDPredictor
        self.predict = predictor_factory(
            model_config=model_cfg,
            checkpoint=c["checkpoint"],
            topomap=c["topomap"],
            goal_node=c["goal_node"],
            radius=c["radius"],
            close_threshold=c["close_threshold"],
            num_samples=c["num_samples"],
            device=c["device"],
        )
        self.publisher = self.create_publisher(
            Path,
            c["path_topic"],
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.buffer = Buffer(node=self)
        self.listener = TransformListener(self.buffer, self)
        self.subscription = self.create_subscription(
            CompressedImage,
            c["image_topic"],
            self.on_image,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nomad_inference"
        )
        self.future = None
        self.pending = None
        self.generation = 0
        self.last_submitted = None
        self.last_received = None
        self.published_stamp = None
        self.last_input_stamp = None
        self.last_ros = self.get_clock().now().nanoseconds
        self.closest_node = 0
        self.terminal = False
        self.closed = False
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(
            c["freshness_check_period_sec"], self.tick, clock=self.steady_clock
        )
        self.publish_stop()

    def publish_stop(self):
        self.published_stamp = None
        self.publisher.publish(
            empty_path(self.cfg["global_frame"], self.get_clock().now().nanoseconds)
        )

    def invalidate(self):
        self.generation += 1
        self.history.clear()
        self.last_received = None
        self.last_submitted = None
        self.publish_stop()

    def check_clock(self):
        now = self.get_clock().now().nanoseconds
        if now < self.last_ros:
            self.invalidate()
            self.closest_node = 0
            self.last_input_stamp = None
        self.last_ros = now
        return now

    def on_image(self, msg):
        if self.closed or self.terminal:
            return
        now = self.check_clock()
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        if (
            stamp <= 0
            or stamp > now
            or now - stamp >= self.cfg["observation_timeout_sec"] * 1e9
        ):
            return
        try:
            image = decode_compressed(msg.data)
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=2.0)
            return
        # Replayed frames must not refresh the receipt timeout.
        if self.last_input_stamp == stamp:
            return
        if self.last_input_stamp is not None and stamp < self.last_input_stamp:
            self.invalidate()
        self.last_input_stamp = stamp
        reset = self.history.add(image, stamp)
        if reset:
            # Keep the first image of the new history but invalidate old work.
            self.generation += 1
            self.last_submitted = None
            self.publish_stop()
        self.last_received = time.monotonic()

    def tick(self):
        if self.closed or self.terminal:
            return
        now = self.check_clock()
        snapshot = self.history.snapshot()
        stale = self.last_received is not None and (
            time.monotonic() - self.last_received
            >= self.cfg["image_receive_timeout_sec"]
            or (
                self.history.last_seen is not None
                and now - self.history.last_seen
                >= self.cfg["observation_timeout_sec"] * 1e9
            )
        )
        stale_path = self.published_stamp is not None and (
            now < self.published_stamp
            or now - self.published_stamp >= self.cfg["observation_timeout_sec"] * 1e9
        )
        if stale or stale_path:
            self.invalidate()
            snapshot = None
        if self.future is not None and self.future.done():
            future, pending = self.future, self.pending
            self.future = None
            self.pending = None
            generation, stamp = pending
            try:
                result = future.result()
                if generation != self.generation:
                    return
                if (
                    now < stamp
                    or now - stamp >= self.cfg["observation_timeout_sec"] * 1e9
                ):
                    self.invalidate()
                    return
                if result.reached_goal:
                    self.terminal = True
                    self.invalidate()
                    return
                transform = self.buffer.lookup_transform(
                    self.cfg["global_frame"],
                    self.cfg["robot_frame"],
                    Time(nanoseconds=stamp, clock_type=ClockType.ROS_TIME),
                    timeout=Duration(),
                )
                path = metric_path(
                    result.points,
                    self.cfg["action_scale_m"],
                    transform,
                    self.cfg["global_frame"],
                    stamp,
                )
                self.publisher.publish(path)
                self.published_stamp = stamp
                self.closest_node = result.closest_node
            except Exception as exc:
                self.get_logger().error(
                    f"Prediction discarded: {exc}", throttle_duration_sec=2.0
                )
                self.invalidate()
                return
        # One worker, no queued backlog. Snapshot is immutable and owned by the job.
        snapshot = self.history.snapshot()
        if self.future is None and snapshot is not None:
            images, stamp = snapshot
            if stamp == self.last_submitted:
                return
            if now < stamp or now - stamp >= self.cfg["observation_timeout_sec"] * 1e9:
                self.invalidate()
                return
            self.last_submitted = stamp
            self.pending = (self.generation, stamp)
            self.future = self.pool.submit(
                self.predict, list(images), self.closest_node
            )

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.invalidate()
        self.timer.cancel()
        self.pool.shutdown(wait=False, cancel_futures=True)


def load_settings(filename):
    filename = FilePath(filename).resolve()
    with filename.open() as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Interface config must be a YAML mapping")
    for name in ("model_config", "checkpoint", "topomap"):
        value = settings.get(name)
        if value and not FilePath(value).is_absolute():
            settings[name] = str(filename.parent / value)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface-config", required=True)
    args, ros_args = parser.parse_known_args()
    settings = load_settings(args.interface_config)
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = RobotNavigationNode(settings)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
