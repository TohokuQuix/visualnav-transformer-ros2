#!/usr/bin/env /usr/bin/python3
import json
import math
import socket
import sys
import time
from typing import Dict, Optional, Sequence, Tuple

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from rclpy.duration import Duration
from rclpy.exceptions import ParameterAlreadyDeclaredException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float32MultiArray, Float64, Float64MultiArray, Int8, String
from tf2_ros import Buffer, TransformListener

DEFAULT_FLIPPER_JOINT_NAMES = (
    "rr_flipper_joint",
    "rl_flipper_joint",
    "fr_flipper_joint",
    "fl_flipper_joint",
)
DEFAULT_FLIPPER_CMD_JOINT_NAMES = (
    "fl_flipper_joint",
    "fr_flipper_joint",
    "rl_flipper_joint",
    "rr_flipper_joint",
)
DEFAULT_RIGHT_TRACK_VEL_NAMES = (
    "fr_sprocket_axle",
    "rr_sprocket_axle",
)
DEFAULT_LEFT_TRACK_VEL_NAMES = (
    "fl_sprocket_axle",
    "rl_sprocket_axle",
)
VALID_MODES = ("manual", "semiauto", "auto")
FLIPPER_FOLDED_THRESHOLD_RAD = math.radians(120.0)
FLIPPER_OPPOSITE_MAX_RAD = math.radians(90.0)


class OnixPolicyBridgeFrontend(Node):
    def __init__(self) -> None:
        super().__init__("onix_policy_bridge")
        self._throttle_state: Dict[str, float] = {}
        self._sock: Optional[socket.socket] = None
        self._reader = None
        self._writer = None

        self._declare_param("use_sim_time", True)
        self.rate_hz = float(self._declare_param("rate", 20.0))
        self.cmd_timeout = float(self._declare_param("cmd_timeout", 0.5))
        self.flipper_manual_timeout = float(self._declare_param("flipper_manual_timeout", self.cmd_timeout))
        self.state_timeout = float(self._declare_param("state_timeout", 0.5))
        self.sprocket_radius = float(self._declare_param("sprocket_radius", 0.0703))
        self.track_tread = float(self._declare_param("track_tread", 0.56))
        self.cmd_deadband_linear = float(self._declare_param("cmd_deadband_linear", 0.01))
        self.cmd_deadband_angular = float(self._declare_param("cmd_deadband_angular", 0.01))
        self.use_stack_avoidance = bool(self._declare_param("use_stack_avoidance", False))
        self.stuck_turn_rate_ratio = float(self._declare_param("stuck_turn_rate_ratio", 0.25))
        self.stuck_detection_duration = float(self._declare_param("stuck_detection_duration", 0.5))
        self.stuck_linear_random_amplitude = float(self._declare_param("stuck_linear_random_amplitude", 0.2))
        self.stuck_linear_gain = float(self._declare_param("stuck_linear_gain", 1.0))
        self.stuck_direction_switch_period = float(self._declare_param("stuck_direction_switch_period", 0.5))
        self.mode_topic = str(self._declare_param("mode_topic", "/chassis_mode_cmd"))
        self.mode_state_topic = str(self._declare_param("mode_state_topic", "/chassis_mode_state"))
        self.mode_state_period = float(self._declare_param("mode_state_period", 1.0))
        self.default_mode = str(self._declare_param("default_mode", "manual")).strip().lower()
        if self.default_mode not in VALID_MODES:
            raise ValueError(f"Unsupported default_mode: {self.default_mode}")
        self.drive_direction_topic = str(self._declare_param("drive_direction_topic", "/chassis/direction"))
        self.cmd_manual_topic = str(self._declare_param("cmd_vel_manual_topic", "/cmd_vel_manual"))
        self.cmd_manual_use_stamped = bool(self._declare_param("cmd_vel_manual_use_stamped", True))
        self.cmd_auto_topic = str(self._declare_param("cmd_vel_auto_topic", "/cmd_vel_auto"))
        self.cmd_auto_use_stamped = bool(self._declare_param("cmd_vel_auto_use_stamped", True))
        self.cmd_out_topic = str(self._declare_param("cmd_vel_out_topic", "/cmd_vel_out"))
        self.cmd_out_use_stamped = bool(self._declare_param("cmd_vel_out_use_stamped", False))
        self.cmd_out_max_linear_velocity = float(self._declare_param("cmd_vel_out_max_linear_velocity", 1.0))
        self.cmd_out_max_angular_velocity = float(self._declare_param("cmd_vel_out_max_angular_velocity", 2.0))
        self.cmd_out_max_linear_acceleration = float(self._declare_param("cmd_vel_out_max_linear_acceleration", 0.5))
        self.cmd_out_max_angular_acceleration = float(self._declare_param("cmd_vel_out_max_angular_acceleration", 2.0))
        self.cmd_out_limit_dt_max = float(self._declare_param("cmd_vel_out_limit_dt_max", 0.1))
        self.joint_states_topic = str(self._declare_param("joint_states_topic", "/joint_states"))
        self.imu_topic = str(self._declare_param("imu_topic", "/imu"))
        self.fallback_imu_topic = str(self._declare_param("fallback_imu_topic", "/livox/imu_172_16_94_141"))
        self.fallback_orientation_parent_frame = str(self._declare_param("fallback_orientation_parent_frame", "map"))
        self.fallback_orientation_child_frame = str(self._declare_param("fallback_orientation_child_frame", "chassis/body"))
        self.fallback_tf_timeout = float(self._declare_param("fallback_tf_timeout", 0.0))
        self.height_scan_topic = str(self._declare_param("height_scan_topic", "/height_scan"))
        self.flipper_fold_mode_topic = str(self._declare_param("flipper_fold_mode_topic", "/flipper_fold_mode"))
        self.flipper_cmd_topic = str(self._declare_param("flipper_cmd_topic", ""))
        self.flipper_manual_topic = str(self._declare_param("flipper_manual_topic", "/flipper_manual"))
        self.flipper_cmd_mode = str(self._declare_param("flipper_cmd_mode", "position")).strip().lower()
        self.flipper_velocity_kp = float(self._declare_param("flipper_velocity_kp", 3.0))
        self.flipper_velocity_limit = float(self._declare_param("flipper_velocity_limit", 1.5))
        self.flipper_velocity_deadband = float(self._declare_param("flipper_velocity_deadband", 0.02))
        self.flipper_manual_deadband = float(self._declare_param("flipper_manual_deadband", self.flipper_velocity_deadband))
        self.follower_status_topic = str(self._declare_param("follower_status_topic", "/simple_trajectory_follower/status"))
        self.follower_goal_flipper_angle_deg = float(self._declare_param("follower_goal_flipper_angle_deg", 90.0))
        self.flipper_topics = {
            "fr": str(self._declare_param("fr_topic", "sub_tracks/fr_pos/command")),
            "fl": str(self._declare_param("fl_topic", "sub_tracks/fl_pos/command")),
            "rr": str(self._declare_param("rr_topic", "sub_tracks/rr_pos/command")),
            "rl": str(self._declare_param("rl_topic", "sub_tracks/rl_pos/command")),
        }
        self.flipper_joint_names = self._tuple_string_param("flipper_joint_names", DEFAULT_FLIPPER_JOINT_NAMES)
        self.flipper_cmd_joint_names = self._tuple_string_param("flipper_cmd_joint_names", DEFAULT_FLIPPER_CMD_JOINT_NAMES)
        self.right_track_vel_names = self._tuple_string_param("right_track_vel_names", DEFAULT_RIGHT_TRACK_VEL_NAMES)
        self.left_track_vel_names = self._tuple_string_param("left_track_vel_names", DEFAULT_LEFT_TRACK_VEL_NAMES)
        self.use_joint_effort = self._optional_bool_param("use_joint_effort")
        self.use_height_scan = self._optional_bool_param("use_height_scan")
        self.height_scan_dim = int(self._declare_param("height_scan_dim", 0))
        self.ipc_host = str(self._declare_param("ipc_host", "127.0.0.1"))
        self.ipc_port = int(self._declare_param("ipc_port", 29001))

        self.last_cmd_time = None
        self.last_joint_time = None
        self.last_imu_time = None
        self.last_fallback_imu_time = None
        self.last_height_time = None
        self.joint_sample_id = 0
        self.imu_sample_id = 0
        self.fallback_imu_sample_id = 0
        self.height_sample_id = 0
        self.last_used_joint_sample_id = -1
        self.last_used_imu_sample_id = -1
        self.last_used_fallback_imu_sample_id = -1
        self.last_used_height_sample_id = -1
        self.current_mode = self.default_mode
        self.drive_direction = 1
        self.cmd_manual_msg = Twist()
        self.cmd_auto_msg = Twist()
        self.last_cmd_out = (0.0, 0.0)
        self.last_cmd_out_time = time.monotonic()
        self.last_cmd_manual_time = None
        self.last_cmd_auto_time = None
        self.flipper_manual_cmd: Optional[Sequence[float]] = None
        self.last_flipper_manual_time = None
        self.last_flipper_manual_change_time = None
        self.follower_status = "idle"
        self.follower_goal_flipper_override: Optional[Sequence[float]] = None
        self.flipper_pos: Dict[str, float] = {}
        self.flipper_joint_vel: Dict[str, float] = {}
        self.track_joint_vel: Dict[str, float] = {}
        self.flipper_joint_effort: Dict[str, float] = {}
        self.track_joint_effort: Dict[str, float] = {}
        self.imu_msg = Imu()
        self.fallback_imu_msg = Imu()
        self.height_scan = []
        self.flipper_fold_mode = 0
        self._stuck_turn_start_time: Optional[float] = None
        self._stuck_escape_direction = 1.0
        self._stuck_escape_last_switch_time: Optional[float] = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if self.cmd_out_use_stamped:
            self.cmd_out_pub = self.create_publisher(TwistStamped, self.cmd_out_topic, 1)
        else:
            self.cmd_out_pub = self.create_publisher(Twist, self.cmd_out_topic, 1)
        self.mode_state_pub = self.create_publisher(String, self.mode_state_topic, 1)
        self.flipper_array_pub = self.create_publisher(Float64MultiArray, self.flipper_cmd_topic, 1) if self.flipper_cmd_topic else None
        self.flipper_pubs = (
            {}
            if self.flipper_array_pub is not None
            else {key: self.create_publisher(Float64, topic, 1) for key, topic in self.flipper_topics.items() if topic}
        )
        self.create_subscription(String, self.mode_topic, self._mode_cb, 1)
        if self.drive_direction_topic:
            self.create_subscription(String, self.drive_direction_topic, self._drive_direction_cb, 1)
        if self.cmd_manual_use_stamped:
            self.create_subscription(TwistStamped, self.cmd_manual_topic, self._cmd_manual_stamped_cb, 1)
        else:
            self.create_subscription(Twist, self.cmd_manual_topic, self._cmd_manual_cb, 1)
        if self.cmd_auto_use_stamped:
            self.create_subscription(TwistStamped, self.cmd_auto_topic, self._cmd_auto_stamped_cb, 1)
        else:
            self.create_subscription(Twist, self.cmd_auto_topic, self._cmd_auto_cb, 1)
        self.create_subscription(Float64MultiArray, self.flipper_manual_topic, self._flipper_manual_cb, 1)
        self.create_subscription(String, self.follower_status_topic, self._follower_status_cb, 1)
        self.create_subscription(JointState, self.joint_states_topic, self._joint_cb, 1)
        self.create_subscription(Imu, self.imu_topic, self._imu_cb, 1)
        if self.fallback_imu_topic:
            self.create_subscription(Imu, self.fallback_imu_topic, self._fallback_imu_cb, 1)
        self.create_subscription(Float32MultiArray, self.height_scan_topic, self._height_scan_cb, 1)
        if self.flipper_fold_mode_topic:
            self.create_subscription(Int8, self.flipper_fold_mode_topic, self._flipper_fold_mode_cb, 1)
        self.timer = self.create_timer(1.0 / max(self.rate_hz, 1.0e-6), self._on_timer)
        self.mode_state_timer = self.create_timer(max(self.mode_state_period, 1.0e-3), self._publish_mode_state)

    def _declare_param(self, name: str, default):
        try:
            return self.declare_parameter(name, default).value
        except ParameterAlreadyDeclaredException:
            return self.get_parameter(name).value

    def _tuple_string_param(self, name: str, default: Sequence[str]):
        value = self._declare_param(name, list(default))
        if isinstance(value, str):
            try:
                value = yaml.safe_load(value)
            except Exception:
                value = [value]
        return tuple(str(v) for v in (value or default))

    def _optional_bool_param(self, name: str) -> Optional[bool]:
        value = self._declare_param(name, "")
        if isinstance(value, bool):
            return value
        value_l = str(value).strip().lower()
        if value_l in {"", "auto", "default", "none"}:
            return None
        if value_l in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if value_l in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Unsupported boolean param for {name}: {value}")

    def _now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) / 1.0e9

    def _age_sec(self, stamp_ns: Optional[int]) -> float:
        if stamp_ns is None:
            return float("inf")
        return max(0.0, self._now_sec() - float(stamp_ns) / 1.0e9)

    def _throttled_log(self, key: str, period: float, level: str, message: str) -> None:
        now = time.monotonic()
        if now - self._throttle_state.get(key, float("-inf")) < period:
            return
        self._throttle_state[key] = now
        logger = self.get_logger()
        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "fatal":
            logger.fatal(message)
        else:
            logger.info(message)

    def _mode_cb(self, msg: String) -> None:
        requested = str(msg.data).strip().lower()
        if requested not in VALID_MODES:
            self._throttled_log("mode_invalid", 1.0, "warning", f"Ignoring unsupported mode command: {msg.data!r}")
            return
        self._set_mode(requested, "mode_cmd")

    def _drive_direction_cb(self, msg: String) -> None:
        direction = self._parse_drive_direction(msg.data)
        if direction is None:
            self._throttled_log(
                "drive_direction_invalid",
                1.0,
                "warning",
                f"Ignoring unsupported drive direction: {msg.data!r}",
            )
            return
        self.drive_direction = direction

    @staticmethod
    def _parse_drive_direction(value: str) -> Optional[int]:
        text = str(value).strip().lower()
        if text in {"front", "forward", "forwards", "fwd", "1", "+1"}:
            return 1
        if text in {"rear", "reverse", "backward", "backwards", "back", "rev", "-1"}:
            return -1
        return None

    def _cmd_manual_cb(self, msg: Twist) -> None:
        self.cmd_manual_msg = msg
        self.last_cmd_manual_time = self.get_clock().now().nanoseconds

    def _cmd_manual_stamped_cb(self, msg: TwistStamped) -> None:
        self.cmd_manual_msg = msg.twist
        self.last_cmd_manual_time = self.get_clock().now().nanoseconds

    def _cmd_auto_cb(self, msg: Twist) -> None:
        self.cmd_auto_msg = msg
        self.last_cmd_auto_time = self.get_clock().now().nanoseconds

    def _cmd_auto_stamped_cb(self, msg: TwistStamped) -> None:
        self.cmd_auto_msg = msg.twist
        self.last_cmd_auto_time = self.get_clock().now().nanoseconds

    def _flipper_manual_cb(self, msg: Float64MultiArray) -> None:
        values = [float(v) for v in msg.data]
        if len(values) < len(self.flipper_cmd_joint_names):
            values.extend([0.0] * (len(self.flipper_cmd_joint_names) - len(values)))
        next_cmd = tuple(values[: len(self.flipper_cmd_joint_names)])
        if self.flipper_manual_cmd is None or any(
            abs(float(curr) - float(prev)) > 1.0e-6
            for curr, prev in zip(next_cmd, self.flipper_manual_cmd)
        ):
            self.last_flipper_manual_change_time = self.get_clock().now().nanoseconds
        self.flipper_manual_cmd = next_cmd
        self.follower_goal_flipper_override = None
        self.last_flipper_manual_time = self.get_clock().now().nanoseconds

    def _follower_status_cb(self, msg: String) -> None:
        status = str(msg.data).strip().lower()
        self.follower_status = status
        if status != "goal" or self.current_mode != "auto":
            return
        goal_angle = math.pi * self.follower_goal_flipper_angle_deg / 180.0
        self.follower_goal_flipper_override = tuple(goal_angle for _ in self.flipper_cmd_joint_names)
        self._set_mode("manual", "follower_goal_reached")

    def _joint_cb(self, msg: JointState) -> None:
        name_to_index = {name: idx for idx, name in enumerate(msg.name)}
        self.flipper_pos = self._extract_joint_map(msg.position, name_to_index, self.flipper_joint_names)
        self.flipper_joint_vel = self._extract_joint_map(msg.velocity, name_to_index, self.flipper_joint_names)
        track_vel_names = self.right_track_vel_names + self.left_track_vel_names
        self.track_joint_vel = self._extract_joint_map(msg.velocity, name_to_index, track_vel_names)
        self.flipper_joint_effort = self._extract_joint_map(msg.effort, name_to_index, self.flipper_joint_names)
        self.track_joint_effort = self._extract_joint_map(msg.effort, name_to_index, track_vel_names)
        self.last_joint_time = self.get_clock().now().nanoseconds
        self.joint_sample_id += 1

    def _extract_joint_map(self, values: Sequence[float], name_to_index: Dict[str, int], target_names: Sequence[str]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if not values:
            return result
        for joint_name in target_names:
            if joint_name in name_to_index and name_to_index[joint_name] < len(values):
                result[joint_name] = float(values[name_to_index[joint_name]])
        return result

    def _imu_cb(self, msg: Imu) -> None:
        self.imu_msg = msg
        self.last_imu_time = self.get_clock().now().nanoseconds
        self.imu_sample_id += 1

    def _fallback_imu_cb(self, msg: Imu) -> None:
        self.fallback_imu_msg = msg
        self.last_fallback_imu_time = self.get_clock().now().nanoseconds
        self.fallback_imu_sample_id += 1

    def _height_scan_cb(self, msg: Float32MultiArray) -> None:
        self.height_scan = [float(v) for v in msg.data]
        self.last_height_time = self.get_clock().now().nanoseconds
        self.height_sample_id += 1

    def _flipper_fold_mode_cb(self, msg: Int8) -> None:
        value = int(msg.data)
        if value > 0:
            self.flipper_fold_mode = 1
        elif value < 0:
            self.flipper_fold_mode = -1
        else:
            self.flipper_fold_mode = 0

    def _set_mode(self, mode: str, reason: str) -> None:
        if mode == self.current_mode:
            return
        prev = self.current_mode
        self.current_mode = mode
        if mode != "manual":
            self.follower_goal_flipper_override = None
        self.get_logger().info(f"Mode changed: {prev} -> {mode} ({reason})")
        self._publish_mode_state()

    def _publish_mode_state(self) -> None:
        self.mode_state_pub.publish(String(data=self.current_mode))

    def _has_valid_state(self) -> bool:
        effective_imu, _ = self._effective_imu_state(log_fallback=False)
        if self._age_sec(self.last_joint_time) > self.state_timeout:
            return False
        if effective_imu is None:
            return False
        if not set(self.flipper_joint_names).issubset(self.flipper_pos.keys()):
            return False
        if not set(self.right_track_vel_names).issubset(self.track_joint_vel.keys()):
            return False
        if not set(self.left_track_vel_names).issubset(self.track_joint_vel.keys()):
            return False
        if self.use_height_scan and self._age_sec(self.last_height_time) > self.state_timeout:
            return False
        if self.use_height_scan and self.height_scan_dim and len(self.height_scan) != self.height_scan_dim:
            return False
        return True

    def _missing_state_reasons(self, effective_imu: Optional[Dict], imu_reasons: Sequence[str]) -> Sequence[str]:
        reasons = []
        joint_age = self._age_sec(self.last_joint_time)
        if joint_age > self.state_timeout:
            reasons.append(f"joint_states timeout: age={joint_age:.3f}s > state_timeout={self.state_timeout:.3f}s")

        if effective_imu is None:
            reasons.extend(imu_reasons)

        missing_flipper_pos = [name for name in self.flipper_joint_names if name not in self.flipper_pos]
        if missing_flipper_pos:
            reasons.append("missing flipper positions: " + ", ".join(missing_flipper_pos))

        missing_right_track_vel = [name for name in self.right_track_vel_names if name not in self.track_joint_vel]
        if missing_right_track_vel:
            reasons.append("missing right track velocities: " + ", ".join(missing_right_track_vel))

        missing_left_track_vel = [name for name in self.left_track_vel_names if name not in self.track_joint_vel]
        if missing_left_track_vel:
            reasons.append("missing left track velocities: " + ", ".join(missing_left_track_vel))

        if self.use_height_scan:
            height_age = self._age_sec(self.last_height_time)
            if height_age > self.state_timeout:
                reasons.append(f"height_scan timeout: age={height_age:.3f}s > state_timeout={self.state_timeout:.3f}s")
            if self.height_scan_dim and len(self.height_scan) != self.height_scan_dim:
                reasons.append(
                    f"height_scan dimension mismatch: expected={self.height_scan_dim}, actual={len(self.height_scan)}"
                )

        return reasons

    # def _has_fresh_state(self, effective_imu: Optional[Dict]) -> bool:
    #     if self.joint_sample_id <= self.last_used_joint_sample_id:
    #         return False
    #     if effective_imu is None:
    #         return False
    #     if effective_imu.get("source") == "fallback":
    #         if self.fallback_imu_sample_id <= self.last_used_fallback_imu_sample_id:
    #             return False
    #     elif self.imu_sample_id <= self.last_used_imu_sample_id:
    #         return False
    #     if self.use_height_scan and self.height_sample_id <= self.last_used_height_sample_id:
    #         return False
    #     return True
    
    def _has_fresh_state(self, effective_imu: Optional[Dict]) -> bool:
        if self.joint_sample_id <= self.last_used_joint_sample_id:
            return False
        if effective_imu is None:
            return False

        # primary IMU のときだけ IMU の fresh を要求
        if effective_imu.get("source") == "primary":
            if self.imu_sample_id <= self.last_used_imu_sample_id:
                return False

        # fallback のときは TF / fallback IMU の最新値を使い回す
        # ただし _fallback_imu_state() 内で state_timeout は確認される

        if self.use_height_scan and self.height_sample_id <= self.last_used_height_sample_id:
            return False
        return True

    def _mark_imu_state_used(self, effective_imu: Dict) -> None:
        if effective_imu.get("source") == "fallback":
            self.last_used_fallback_imu_sample_id = self.fallback_imu_sample_id
        else:
            self.last_used_imu_sample_id = self.imu_sample_id

    def _connect(self) -> bool:
        if self._sock is not None:
            return True
        try:
            self._sock = socket.create_connection((self.ipc_host, self.ipc_port), timeout=1.0)
            self._reader = self._sock.makefile("r", encoding="utf-8")
            self._writer = self._sock.makefile("w", encoding="utf-8")
            self.get_logger().info(f"Connected to backend {self.ipc_host}:{self.ipc_port}")
            return True
        except OSError as exc:
            self._throttled_log("backend_connect", 1.0, "warning", f"Backend unavailable: {exc}")
            self._sock = None
            self._reader = None
            self._writer = None
            return False

    def _request(self, payload: Dict) -> Optional[Dict]:
        if not self._connect():
            return None
        try:
            self._writer.write(json.dumps(payload) + "\n")
            self._writer.flush()
            line = self._reader.readline()
            if not line:
                raise ConnectionError("backend closed connection")
            return json.loads(line)
        except Exception as exc:
            self._throttled_log("backend_io", 1.0, "error", f"Backend I/O failed: {exc}")
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._reader = None
            self._writer = None
            return None

    def _build_payload(self, cmd_msg: Twist, effective_imu: Dict) -> Dict:
        return {
            "cmd_vel": {
                "linear_x": float(cmd_msg.linear.x),
                "linear_y": float(cmd_msg.linear.y),
                "angular_z": float(cmd_msg.angular.z),
            },
            "drive_direction": self.drive_direction,
            "flipper_pos": self.flipper_pos,
            "flipper_joint_vel": self.flipper_joint_vel,
            "track_joint_vel": self.track_joint_vel,
            "flipper_joint_effort": self.flipper_joint_effort,
            "track_joint_effort": self.track_joint_effort,
            "imu_source": effective_imu["source"],
            "imu_frame": effective_imu["frame"],
            "imu_orientation": effective_imu["orientation"],
            "imu_angular_velocity": effective_imu["angular_velocity"],
            "height_scan": self.height_scan,
            "flipper_fold_mode": self.flipper_fold_mode,
        }

    def _publish(self, response: Dict) -> None:
        flippers = response["flippers"]
        if self._flipper_joint_states_ready():
            if self.flipper_array_pub is not None:
                msg = Float64MultiArray()
                msg.data = [self._flipper_command_value(name, flippers) for name in self.flipper_cmd_joint_names]
                self.flipper_array_pub.publish(msg)
            else:
                for key, value in flippers.items():
                    pub = self.flipper_pubs.get(key)
                    if pub is not None:
                        pub.publish(Float64(data=float(value)))
        self._publish_cmd_out(response["cmd_out"])

    @staticmethod
    def _controller_joint_to_key(joint_name: str) -> str:
        normalized = joint_name.split("/")[-1]
        if normalized.startswith("rr_"):
            return "rr"
        if normalized.startswith("rl_"):
            return "rl"
        if normalized.startswith("fr_"):
            return "fr"
        if normalized.startswith("fl_"):
            return "fl"
        raise KeyError(f"Unsupported flipper controller joint name: {joint_name}")

    def _flipper_command_value(self, joint_name: str, flippers: Dict[str, float]) -> float:
        key = self._controller_joint_to_key(joint_name)
        target = float(flippers[key])
        if self.flipper_cmd_mode != "velocity":
            return target
        current = float(self.flipper_pos.get(joint_name, self.flipper_pos.get(joint_name.split("/")[-1], target)))
        error = target - current
        if abs(error) < self.flipper_velocity_deadband:
            return 0.0
        return self._clamp(error * self.flipper_velocity_kp, -self.flipper_velocity_limit, self.flipper_velocity_limit)

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def _active_limit(limit: float) -> Optional[float]:
        limit = float(limit)
        if not math.isfinite(limit) or limit <= 0.0:
            return None
        return limit

    @classmethod
    def _clamp_abs_if_limited(cls, value: float, limit: float) -> float:
        active_limit = cls._active_limit(limit)
        if active_limit is None:
            return float(value)
        return cls._clamp(float(value), -active_limit, active_limit)

    @classmethod
    def _rate_limit_if_limited(cls, target: float, previous: float, rate_limit: float, dt: float) -> float:
        active_limit = cls._active_limit(rate_limit)
        if active_limit is None:
            return float(target)
        max_delta = active_limit * max(0.0, float(dt))
        return cls._clamp(float(target), float(previous) - max_delta, float(previous) + max_delta)

    @staticmethod
    def _finite_or_zero(value: float) -> float:
        value = float(value)
        return value if math.isfinite(value) else 0.0

    def _zero_twist(self) -> Twist:
        return Twist()

    def _command_is_active(self, msg: Twist, stamp_ns: Optional[int]) -> bool:
        if self._age_sec(stamp_ns) > self.cmd_timeout:
            return False
        return (
            abs(float(msg.linear.x)) >= self.cmd_deadband_linear
            or abs(float(msg.linear.y)) >= self.cmd_deadband_linear
            or abs(float(msg.angular.z)) >= self.cmd_deadband_angular
        )

    def _flipper_manual_is_active(self) -> bool:
        return self._age_sec(self.last_flipper_manual_change_time) <= self.flipper_manual_timeout

    def _manual_flipper_targets(self) -> Optional[Sequence[float]]:
        if self.follower_goal_flipper_override is not None:
            return [float(v) for v in self.follower_goal_flipper_override]
        if self.flipper_manual_cmd is not None:
            return [float(v) for v in self.flipper_manual_cmd]
        self._throttled_log(
            "flipper_manual_cmd_wait",
            1.0,
            "info",
            "Waiting for first manual flipper command before publishing subcrawler commands",
        )
        return None

    def _flipper_joint_states_ready(self) -> bool:
        missing = [name for name in self.flipper_joint_names if name not in self.flipper_pos]
        if not missing:
            return True
        self._throttled_log(
            "flipper_joint_state_wait",
            1.0,
            "info",
            "Waiting for flipper joint_states before publishing subcrawler commands: "
            + ", ".join(missing),
        )
        return False

    def _current_manual_cmd(self) -> Twist:
        if self._age_sec(self.last_cmd_manual_time) > self.cmd_timeout:
            return self._zero_twist()
        return self.cmd_manual_msg

    def _current_auto_cmd(self) -> Twist:
        if self._age_sec(self.last_cmd_auto_time) > self.cmd_timeout:
            return self._zero_twist()
        return self.cmd_auto_msg

    def _publish_cmd_out(self, cmd: Dict[str, float]) -> None:
        cmd = self._limited_cmd_out(cmd, apply_velocity_limits=(self.current_mode == "auto"))
        if self.cmd_out_use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x = float(cmd["linear_x"])
            msg.twist.angular.z = float(cmd["angular_z"])
        else:
            msg = Twist()
            msg.linear.x = float(cmd["linear_x"])
            msg.angular.z = float(cmd["angular_z"])
        self.cmd_out_pub.publish(msg)

    def _limited_cmd_out(self, cmd: Dict[str, float], apply_velocity_limits: bool) -> Dict[str, float]:
        linear_x = self._finite_or_zero(cmd.get("linear_x", 0.0))
        angular_z = self._finite_or_zero(cmd.get("angular_z", 0.0))

        if apply_velocity_limits:
            linear_x = self._clamp_abs_if_limited(linear_x, self.cmd_out_max_linear_velocity)
            angular_z = self._clamp_abs_if_limited(angular_z, self.cmd_out_max_angular_velocity)

        now = time.monotonic()
        dt = max(0.0, now - self.last_cmd_out_time)
        dt_max = self._active_limit(self.cmd_out_limit_dt_max)
        if dt_max is not None:
            dt = min(dt, dt_max)

        prev_linear_x, prev_angular_z = self.last_cmd_out
        linear_x = self._rate_limit_if_limited(
            linear_x,
            prev_linear_x,
            self.cmd_out_max_linear_acceleration,
            dt,
        )
        angular_z = self._rate_limit_if_limited(
            angular_z,
            prev_angular_z,
            self.cmd_out_max_angular_acceleration,
            dt,
        )

        self.last_cmd_out = (linear_x, angular_z)
        self.last_cmd_out_time = now
        return {"linear_x": linear_x, "angular_z": angular_z}

    def _clear_stuck_detection(self) -> None:
        self._stuck_turn_start_time = None
        self._stuck_escape_direction = 1.0
        self._stuck_escape_last_switch_time = None

    def _stuck_escape_active(self, cmd_msg: Twist, effective_imu: Optional[Dict]) -> bool:
        if not self.use_stack_avoidance:
            self._clear_stuck_detection()
            return False

        if effective_imu is None:
            self._clear_stuck_detection()
            return False

        commanded_yaw_rate = abs(float(cmd_msg.angular.z))
        if commanded_yaw_rate < self.cmd_deadband_angular:
            self._clear_stuck_detection()
            return False

        measured_yaw_rate = abs(float(effective_imu["angular_velocity"]["z"]))
        threshold = commanded_yaw_rate * max(0.0, self.stuck_turn_rate_ratio)
        now = self._now_sec()
        if measured_yaw_rate <= threshold:
            if self._stuck_turn_start_time is None:
                self._stuck_turn_start_time = now
                return False
            return (now - self._stuck_turn_start_time) >= max(0.0, self.stuck_detection_duration)

        self._clear_stuck_detection()
        return False

    def _apply_stuck_escape(self, cmd_msg: Twist, effective_imu: Optional[Dict]) -> Twist:
        escaped = Twist()
        escaped.linear.x = float(cmd_msg.linear.x)
        escaped.linear.y = float(cmd_msg.linear.y)
        escaped.linear.z = float(cmd_msg.linear.z)
        escaped.angular.x = float(cmd_msg.angular.x)
        escaped.angular.y = float(cmd_msg.angular.y)
        escaped.angular.z = float(cmd_msg.angular.z)

        if not self._stuck_escape_active(cmd_msg, effective_imu):
            return escaped

        now = self._now_sec()
        switch_period = max(0.0, self.stuck_direction_switch_period)
        if self._stuck_escape_last_switch_time is None:
            self._stuck_escape_last_switch_time = now
        elif switch_period > 0.0 and (now - self._stuck_escape_last_switch_time) >= switch_period:
            self._stuck_escape_direction *= -1.0
            self._stuck_escape_last_switch_time = now

        cmd_yaw_rate = abs(float(cmd_msg.angular.z))
        measured_yaw_rate = abs(float(effective_imu["angular_velocity"]["z"]))
        yaw_rate_error = max(0.0, cmd_yaw_rate - measured_yaw_rate)
        escape_speed = min(
            abs(self.stuck_linear_random_amplitude),
            max(0.0, self.stuck_linear_gain) * yaw_rate_error,
        )
        delta = self._stuck_escape_direction * escape_speed
        escaped.linear.x += float(delta)
        self._throttled_log(
            "stuck_escape_active",
            1.0,
            "info",
            f"Stack escape active: applying linear.x offset {delta:+.3f} m/s from yaw-rate error {yaw_rate_error:.3f}",
        )
        return escaped

    def _effective_imu_state(self, log_fallback: bool = True) -> Tuple[Optional[Dict], Sequence[str]]:
        primary_age = self._age_sec(self.last_imu_time)
        if primary_age <= self.state_timeout:
            return {
                "source": "primary",
                "frame": "sensor",
                "orientation": self._orientation_payload(self.imu_msg.orientation),
                "angular_velocity": self._angular_velocity_payload(
                    (
                        float(self.imu_msg.angular_velocity.x),
                        float(self.imu_msg.angular_velocity.y),
                        float(self.imu_msg.angular_velocity.z),
                    )
                ),
            }, []

        reasons = [f"imu timeout: age={primary_age:.3f}s > state_timeout={self.state_timeout:.3f}s"]
        fallback_state, fallback_reasons = self._fallback_imu_state()
        if fallback_state is None:
            reasons.extend(fallback_reasons)
            return None, reasons

        if log_fallback:
            self._throttled_log(
                "imu_fallback_active",
                1.0,
                "warning",
                f"BNO055 IMU input {self.imu_topic} is unavailable; using angular velocity from "
                f"{self.fallback_imu_topic} and orientation from TF "
                f"{self.fallback_orientation_parent_frame} -> {self.fallback_orientation_child_frame}",
            )
        return fallback_state, []

    def _fallback_imu_state(self) -> Tuple[Optional[Dict], Sequence[str]]:
        reasons = []
        if not self.fallback_imu_topic:
            return None, ["fallback imu topic is disabled"]

        fallback_age = self._age_sec(self.last_fallback_imu_time)
        if fallback_age > self.state_timeout:
            reasons.append(
                f"fallback imu timeout ({self.fallback_imu_topic}): "
                f"age={fallback_age:.3f}s > state_timeout={self.state_timeout:.3f}s"
            )

        transform, tf_reason = self._lookup_fallback_body_transform()
        if transform is None:
            reasons.append(tf_reason)

        if reasons:
            return None, reasons

        return {
            "source": "fallback",
            "frame": "robot",
            "orientation": self._orientation_payload(transform.transform.rotation),
            "angular_velocity": self._angular_velocity_payload(self._fallback_angular_velocity_in_body_frame()),
        }, []

    def _lookup_fallback_body_transform(self) -> Tuple[Optional[TransformStamped], str]:
        parent_frame = self._tf_frame(self.fallback_orientation_parent_frame)
        child_frame = self._tf_frame(self.fallback_orientation_child_frame)
        try:
            transform = self.tf_buffer.lookup_transform(
                parent_frame,
                child_frame,
                Time(),
                timeout=Duration(seconds=max(0.0, self.fallback_tf_timeout)),
            )
        except Exception as exc:
            return None, f"fallback orientation tf unavailable ({parent_frame} -> {child_frame}): {exc}"

        tf_age = self._stamp_age_sec(transform.header.stamp)
        if tf_age is not None and tf_age > self.state_timeout:
            return None, (
                f"fallback orientation tf timeout ({parent_frame} -> {child_frame}): "
                f"age={tf_age:.3f}s > state_timeout={self.state_timeout:.3f}s"
            )
        return transform, ""

    def _fallback_angular_velocity_in_body_frame(self) -> Tuple[float, float, float]:
        vector = (
            float(self.fallback_imu_msg.angular_velocity.x),
            float(self.fallback_imu_msg.angular_velocity.y),
            float(self.fallback_imu_msg.angular_velocity.z),
        )
        source_frame = self._tf_frame(self.fallback_imu_msg.header.frame_id)
        body_frame = self._tf_frame(self.fallback_orientation_child_frame)
        if not source_frame or source_frame == body_frame:
            return vector

        try:
            transform = self.tf_buffer.lookup_transform(
                body_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=max(0.0, self.fallback_tf_timeout)),
            )
        except Exception as exc:
            self._throttled_log(
                "fallback_angular_tf_unavailable",
                1.0,
                "warning",
                f"Could not transform fallback angular velocity from {source_frame} to {body_frame}; "
                f"using raw {self.fallback_imu_topic} vector: {exc}",
            )
            return vector

        return self._rotate_vector_by_quaternion(vector, self._quat_payload(transform.transform.rotation))

    def _stamp_age_sec(self, stamp) -> Optional[float]:
        stamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if stamp_ns <= 0:
            return None
        return max(0.0, self._now_sec() - float(stamp_ns) / 1.0e9)

    @staticmethod
    def _tf_frame(frame_id: str) -> str:
        return str(frame_id).strip().lstrip("/")

    @staticmethod
    def _orientation_payload(quat) -> Dict[str, float]:
        return {"w": float(quat.w), "x": float(quat.x), "y": float(quat.y), "z": float(quat.z)}

    @staticmethod
    def _angular_velocity_payload(vector: Tuple[float, float, float]) -> Dict[str, float]:
        return {"x": float(vector[0]), "y": float(vector[1]), "z": float(vector[2])}

    @staticmethod
    def _quat_payload(quat) -> Tuple[float, float, float, float]:
        return (float(quat.w), float(quat.x), float(quat.y), float(quat.z))

    @staticmethod
    def _quat_conjugate(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, x, y, z = quat
        return (w, -x, -y, -z)

    @staticmethod
    def _quat_multiply(
        a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float, float]:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    def _rotate_vector_by_quaternion(
        self,
        vector: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
    ) -> Tuple[float, float, float]:
        vec_quat = (0.0, vector[0], vector[1], vector[2])
        rotated = self._quat_multiply(self._quat_multiply(quat, vec_quat), self._quat_conjugate(quat))
        return rotated[1], rotated[2], rotated[3]

    def _publish_manual_outputs(self) -> None:
        manual_cmd = self._current_manual_cmd()
        self._publish_cmd_out({"linear_x": float(manual_cmd.linear.x), "angular_z": float(manual_cmd.angular.z)})
        if not self._flipper_joint_states_ready():
            return
        manual_values = self._manual_flipper_targets()
        if manual_values is None:
            return
        manual_values = self._limit_opposite_flipper_fold(manual_values)
        if self.flipper_array_pub is not None:
            msg = Float64MultiArray()
            msg.data = manual_values
            self.flipper_array_pub.publish(msg)
        else:
            manual_map = {
                self._controller_joint_to_key(name): float(value)
                for name, value in zip(self.flipper_cmd_joint_names, manual_values)
            }
            for key, value in manual_map.items():
                pub = self.flipper_pubs.get(key)
                if pub is not None:
                    pub.publish(Float64(data=value))

    def _limit_opposite_flipper_fold(self, command_values: Sequence[float]) -> Sequence[float]:
        measured = self._measured_flipper_positions_by_key()
        front_folded = any(measured.get(key, float("-inf")) >= FLIPPER_FOLDED_THRESHOLD_RAD for key in ("fr", "fl"))
        rear_folded = any(measured.get(key, float("-inf")) >= FLIPPER_FOLDED_THRESHOLD_RAD for key in ("rr", "rl"))
        limited = [float(value) for value in command_values]
        for idx, joint_name in enumerate(self.flipper_cmd_joint_names[: len(limited)]):
            key = self._controller_joint_to_key(joint_name)
            if front_folded and key in {"rr", "rl"}:
                limited[idx] = min(limited[idx], FLIPPER_OPPOSITE_MAX_RAD)
            if rear_folded and key in {"fr", "fl"}:
                limited[idx] = min(limited[idx], FLIPPER_OPPOSITE_MAX_RAD)
        return limited

    def _measured_flipper_positions_by_key(self) -> Dict[str, float]:
        measured: Dict[str, float] = {}
        for joint_name, value in self.flipper_pos.items():
            try:
                key = self._controller_joint_to_key(joint_name)
            except KeyError:
                continue
            measured[key] = float(value)
        return measured

    def _on_timer(self) -> None:
        if self.current_mode == "auto" and self._command_is_active(self.cmd_manual_msg, self.last_cmd_manual_time):
            self._set_mode("manual", "manual_cmd_intervention")
        elif self.current_mode == "semiauto" and self._flipper_manual_is_active():
            self._set_mode("manual", "manual_flipper_intervention")

        if self.current_mode == "manual":
            self._clear_stuck_detection()
            self._publish_manual_outputs()
            return

        effective_imu, imu_reasons = self._effective_imu_state()
        if self.current_mode == "semiauto":
            cmd_source = self._apply_stuck_escape(self._current_manual_cmd(), effective_imu)
        else:
            cmd_source = self._apply_stuck_escape(self._current_auto_cmd(), effective_imu)

        missing_state_reasons = self._missing_state_reasons(effective_imu, imu_reasons)
        if missing_state_reasons:
            message = "Policy input is incomplete: " + "; ".join(missing_state_reasons)
            self._throttled_log("policy_input_incomplete", 1.0, "error", message)
            return

        if not self._has_fresh_state(effective_imu):
            message = (
                "Policy input did not refresh before inference: "
                f"joint_sample_id={self.joint_sample_id}, last_used_joint_sample_id={self.last_used_joint_sample_id}, "
            )
            if effective_imu is not None and effective_imu.get("source") == "fallback":
                message += (
                    f"fallback_imu_sample_id={self.fallback_imu_sample_id}, "
                    f"last_used_fallback_imu_sample_id={self.last_used_fallback_imu_sample_id}"
                )
            else:
                message += f"imu_sample_id={self.imu_sample_id}, last_used_imu_sample_id={self.last_used_imu_sample_id}"
            if self.use_height_scan:
                message += (
                    f", height_sample_id={self.height_sample_id}, "
                    f"last_used_height_sample_id={self.last_used_height_sample_id}"
                )
            self._throttled_log("policy_input_stale", 1.0, "error", message)
            return

        response = self._request(self._build_payload(cmd_source, effective_imu))
        if response is None:
            return
        self._publish(response)
        self.last_used_joint_sample_id = self.joint_sample_id
        self._mark_imu_state_used(effective_imu)
        if self.use_height_scan:
            self.last_used_height_sample_id = self.height_sample_id


def main() -> None:
    rclpy.init(args=sys.argv)
    node = OnixPolicyBridgeFrontend()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
