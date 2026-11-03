#!/usr/bin/env python3
import argparse
import itertools
import json
import math
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml


def _prepend_python_path(path: str) -> None:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _bootstrap_python_paths() -> None:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        str(Path.home() / ".local" / "lib" / version / "site-packages"),
        f"/usr/lib/{version}/dist-packages",
        f"/usr/local/lib/{version}/dist-packages",
        f"/usr/local/lib/{version}/site-packages",
    ]
    conda_root = Path.home() / "anaconda3" / "envs"
    if conda_root.is_dir():
        for env_dir in sorted(conda_root.iterdir()):
            candidates.append(str(env_dir / "lib" / version / "site-packages"))
    for path in reversed(candidates):
        _prepend_python_path(path)


_bootstrap_python_paths()

DEFAULT_FLIPPER_JOINT_NAMES = (
    "rr_flipper_joint",
    "rl_flipper_joint",
    "fr_flipper_joint",
    "fl_flipper_joint",
)
DEFAULT_RIGHT_TRACK_VEL_NAMES = (
    "fr_sprocket_axle",
    "rr_sprocket_axle",
)
DEFAULT_LEFT_TRACK_VEL_NAMES = (
    "fl_sprocket_axle",
    "rl_sprocket_axle",
)
FLIPPER_REVERSE_KEY = {
    "rr": "fl",
    "rl": "fr",
    "fr": "rl",
    "fl": "rr",
}


def _parse_name_list(value, default: Sequence[str]) -> Tuple[str, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return tuple(default)
        try:
            parsed = yaml.safe_load(text)
        except Exception:
            parsed = None
        if isinstance(parsed, str):
            parsed = [parsed]
        if not isinstance(parsed, (list, tuple)):
            parsed = [part for part in text.split(",") if part]
    return tuple(str(item) for item in parsed)


class ActorOnly(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: Sequence[int], activation_name: str) -> None:
        super().__init__()
        activation = self._get_activation(activation_name)
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(activation.__class__())
            prev = hidden_dim
        layers.append(nn.Linear(prev, action_dim))
        self.actor = nn.Sequential(*layers)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        mapping = {
            "elu": nn.ELU,
            "selu": nn.SELU,
            "relu": nn.ReLU,
            "crelu": nn.ReLU,
            "lrelu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported activation: {name}")
        return mapping[name]()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


@dataclass
class ObservationConfig:
    flipper_joint_names: Tuple[str, ...]
    right_track_vel_names: Tuple[str, ...]
    left_track_vel_names: Tuple[str, ...]
    has_base_lin_vel: bool = False
    has_actions: bool = False
    has_joint_effort: bool = False
    has_flipper_fold_mode: bool = False
    include_flipper_joint_vel: bool = False
    has_height_scan: bool = False
    height_scan_dim: int = 0
    cmd_lin_x_range: Tuple[float, float] = (-100.0, 100.0)
    cmd_lin_y_range: Tuple[float, float] = (-100.0, 100.0)
    cmd_ang_z_range: Tuple[float, float] = (-100.0, 100.0)
    flipper_pos_error_scale: float = 1.0
    track_vel_error_scale: float = 0.05
    joint_effort_scale: float = 0.01
    base_lin_vel_scale: float = 2.0
    base_ang_vel_scale: float = 0.25
    projected_gravity_scale: float = 1.0
    flipper_fold_mode_scale: float = 1.0
    flipper_pos_clip: float = 100.0
    track_vel_clip: float = 100.0
    joint_effort_clip: float = 100.0
    gravity_clip: float = 1.0
    base_lin_vel_clip: float = 100.0
    base_ang_vel_clip: float = 100.0
    velocity_cmd_clip: float = 100.0
    flipper_fold_mode_clip: float = 1.0
    height_scan_scale: float = 1.0
    height_scan_clip: float = 1.0


@dataclass
class ActionConfig:
    front_clip: Tuple[float, float]
    rear_clip: Tuple[float, float]
    flipper_scale: float = 1.0
    flipper_offset: float = 0.0
    flipper_max_velocity: float = 0.0
    right_track_scale: float = 1.0
    left_track_scale: float = 1.0
    right_track_offset: float = 0.0
    left_track_offset: float = 0.0
    right_track_clip: Tuple[float, float] = (-14.2, 14.2)
    left_track_clip: Tuple[float, float] = (-14.2, 14.2)
    interference_limit: float = 2.09439510239
    interference_pairs: Tuple[Tuple[int, int], ...] = ((0, 2), (1, 3))
    use_directional_clip: bool = True
    directional_clip_speed_threshold: float = 0.10
    directional_clip_fold_min: float = math.pi
    directional_clip_clear_max: float = math.pi / 2.0
    directional_clip_rear_indices: Tuple[int, ...] = (0, 1)
    directional_clip_front_indices: Tuple[int, ...] = (2, 3)


class _ObsNormalizer(nn.Module):
    def __init__(self, enabled: bool, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None, eps: float = 1.0e-2, device: str = "cpu"):
        super().__init__()
        self.enabled = enabled
        self.eps = float(eps)
        if mean is None or std is None:
            mean = torch.zeros(1, 1, dtype=torch.float32)
            std = torch.ones(1, 1, dtype=torch.float32)
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

    @classmethod
    def from_state_dict(cls, state: Dict, input_dim: int, device: str) -> "_ObsNormalizer":
        mean = state.get("_mean")
        std = state.get("_std")
        if mean is None or std is None:
            raise ValueError("obs_norm_state_dict missing _mean or _std")
        if int(mean.shape[-1]) != input_dim or int(std.shape[-1]) != input_dim:
            raise ValueError("obs_norm_state_dict dimension mismatch")
        eps = float(state.get("eps", 1.0e-2))
        return cls(True, mean=mean, std=std, eps=eps, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        return (x - self.mean) / (self.std + self.eps)


class Backend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = args.device
        self.track_tread = float(args.track_tread)
        self.sprocket_radius = float(args.sprocket_radius)
        self.right_track_sign = float(args.right_track_sign)
        self.left_track_sign = float(args.left_track_sign)
        self.imu_yaw_offset = float(args.imu_yaw_offset)
        self.cmd_deadband_linear = float(args.cmd_deadband_linear)
        self.cmd_deadband_angular = float(args.cmd_deadband_angular)
        self.joint_effort_scale_override = float(args.joint_effort_scale)
        self.last_action_smooth_alpha = float(args.last_action_smooth_alpha)
        self.policy_rate = max(1.0e-6, float(args.rate))
        self.flipper_rate_limit_dt_max = max(0.0, float(args.flipper_rate_limit_dt_max))
        self.directional_flipper_switch_duration = float(args.directional_flipper_switch_duration)
        self.directional_flipper_turn_angular_threshold = float(args.directional_flipper_turn_angular_threshold)
        self.directional_flipper_turn_radius_threshold = float(args.directional_flipper_turn_radius_threshold)
        self.use_last_action_override = self._optional_bool(args.use_last_action)
        self.use_joint_effort_override = self._optional_bool(args.use_joint_effort)
        self.use_height_scan_override = self._optional_bool(args.use_height_scan)
        self.use_directional_flipper_constraint_override = self._optional_bool(args.use_directional_flipper_constraint)
        self.use_flipper_rate_limit_override = self._optional_bool(args.use_flipper_rate_limit)
        self.height_scan_dim_override = int(args.height_scan_dim)
        self.flipper_joint_names = _parse_name_list(args.flipper_joint_names, DEFAULT_FLIPPER_JOINT_NAMES)
        self.right_track_vel_names = _parse_name_list(args.right_track_vel_names, DEFAULT_RIGHT_TRACK_VEL_NAMES)
        self.left_track_vel_names = _parse_name_list(args.left_track_vel_names, DEFAULT_LEFT_TRACK_VEL_NAMES)

        self.obs_cfg, self.action_cfg = self._load_env_config(args.env_yaml_path)
        if self.use_directional_flipper_constraint_override is None:
            self.use_directional_flipper_constraint = bool(self.action_cfg.use_directional_clip)
        else:
            self.use_directional_flipper_constraint = bool(self.use_directional_flipper_constraint_override)
        if self.use_flipper_rate_limit_override is None:
            self.use_flipper_rate_limit = (
                math.isfinite(float(self.action_cfg.flipper_max_velocity))
                and float(self.action_cfg.flipper_max_velocity) > 0.0
            )
        else:
            self.use_flipper_rate_limit = bool(self.use_flipper_rate_limit_override)
        self.policy, self.num_actor_obs, self.policy_action_dim, self.obs_normalizer = self._load_policy(
            args.model_path, self.device, args.activation
        )
        if self.policy_action_dim not in (2, 6):
            raise ValueError(
                "Unsupported policy action dimension: "
                f"{self.policy_action_dim}. Expected 2 (tracks only) or 6 (flippers + tracks)."
            )
        self._reconcile_observation_config_with_model()
        self.policy.eval()
        self.obs_normalizer.eval()
        self.last_action = torch.zeros(6, dtype=torch.float32)
        self.last_action_obs = torch.zeros(self.policy_action_dim, dtype=torch.float32)
        self.last_action_obs_direction = 1
        self.flipper_rate_limited_targets: Optional[torch.Tensor] = None
        self.flipper_rate_limit_time: Optional[float] = None
        self.flipper_rate_limited_targets_direction = 1
        self.travel_direction = 1
        self.pending_travel_direction: Optional[int] = None
        self.pending_travel_direction_since: Optional[float] = None

    @staticmethod
    def _optional_bool(value: str) -> Optional[bool]:
        value_l = str(value).strip().lower()
        if value_l in {"", "auto", "default", "none"}:
            return None
        if value_l in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if value_l in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Unsupported bool value: {value}")

    def _load_env_config(self, env_yaml_path: str) -> Tuple[ObservationConfig, ActionConfig]:
        with open(env_yaml_path, "r", encoding="utf-8") as f:
            env_cfg = yaml.unsafe_load(f)
        observations = env_cfg.get("observations", {})
        policy = observations.get("policy", {})
        actions = env_cfg.get("actions", {})
        commands = env_cfg.get("commands", {})
        joint_pos = policy.get("joint_pos") or {}
        joint_vel = policy.get("joint_vel") or {}
        height_scan = policy.get("height_scan") or {}
        joint_effort = policy.get("joint_effort") or {}
        base_lin_vel = policy.get("base_lin_vel") or {}
        base_ang_vel = policy.get("base_ang_vel") or {}
        projected_gravity = policy.get("projected_gravity") or {}
        velocity_commands = policy.get("velocity_commands") or {}
        flipper_fold_mode = policy.get("flipper_fold_mode") or {}
        actions_obs = policy.get("actions") or {}
        base_velocity = commands.get("base_velocity") or {}
        base_velocity_ranges = base_velocity.get("ranges") or {}
        flipper_pos = actions.get("flipper_pos") or {}
        right_track_vel = actions.get("right_track_vel") or {}
        left_track_vel = actions.get("left_track_vel") or {}

        front_clip = tuple(flipper_pos.get("clip", {}).get(".*(fr|fl).*flipper.*", [-math.pi / 2.0, 3.0 * math.pi / 2.0]))
        rear_clip = tuple(flipper_pos.get("clip", {}).get(".*(rr|rl).*flipper.*", [-math.pi / 2.0, math.pi / 2.0]))
        right_track_clip = self._tuple_range(right_track_vel.get("clip", {}).get(".*"), (-14.2, 14.2))
        left_track_clip = self._tuple_range(left_track_vel.get("clip", {}).get(".*"), right_track_clip)
        joint_vel_func = str((joint_vel or {}).get("func") or "")

        obs_cfg = ObservationConfig(
            flipper_joint_names=self.flipper_joint_names,
            right_track_vel_names=self.right_track_vel_names,
            left_track_vel_names=self.left_track_vel_names,
            has_base_lin_vel=bool(base_lin_vel and base_lin_vel.get("func")),
            has_actions=self._resolve_obs_flag(
                self.use_last_action_override,
                env_cfg,
                "use_last_action_obs",
                bool(actions_obs and actions_obs.get("func")),
            ),
            has_joint_effort=self._resolve_obs_flag(
                self.use_joint_effort_override,
                env_cfg,
                "use_joint_effort_obs",
                bool(joint_effort and joint_effort.get("func")),
            ),
            has_flipper_fold_mode=bool(flipper_fold_mode and flipper_fold_mode.get("func")),
            has_height_scan=self._resolve_obs_flag(
                self.use_height_scan_override,
                env_cfg,
                "use_height_scan_obs",
                bool(height_scan and height_scan.get("func")),
            ),
            include_flipper_joint_vel=("joint_vel_with_track_mean" in joint_vel_func),
            height_scan_dim=self._infer_height_scan_dim(height_scan, self.height_scan_dim_override),
            cmd_lin_x_range=self._tuple_range(base_velocity_ranges.get("lin_vel_x"), (-100.0, 100.0)),
            cmd_lin_y_range=self._tuple_range(base_velocity_ranges.get("lin_vel_y"), (-100.0, 100.0)),
            cmd_ang_z_range=self._tuple_range(base_velocity_ranges.get("ang_vel_z"), (-100.0, 100.0)),
            flipper_pos_error_scale=float(joint_pos.get("scale", 1.0)),
            track_vel_error_scale=float(joint_vel.get("scale", 1.0)),
            joint_effort_scale=float(joint_effort.get("scale", 0.01)) * self.joint_effort_scale_override,
            base_lin_vel_scale=float(base_lin_vel.get("scale", 1.0)),
            base_ang_vel_scale=float(base_ang_vel.get("scale", 1.0)),
            projected_gravity_scale=float(projected_gravity.get("scale", 1.0)),
            flipper_fold_mode_scale=float(flipper_fold_mode.get("scale", 1.0)),
            flipper_pos_clip=float(self._clip_abs(joint_pos.get("clip", [-100.0, 100.0]))),
            track_vel_clip=float(self._clip_abs(joint_vel.get("clip", [-100.0, 100.0]))),
            joint_effort_clip=float(self._clip_abs(joint_effort.get("clip", [-100.0, 100.0]))),
            gravity_clip=float(self._clip_abs(projected_gravity.get("clip", [-1.0, 1.0]))),
            base_lin_vel_clip=float(self._clip_abs(base_lin_vel.get("clip", [-100.0, 100.0]))),
            base_ang_vel_clip=float(self._clip_abs(base_ang_vel.get("clip", [-100.0, 100.0]))),
            velocity_cmd_clip=float(self._clip_abs(velocity_commands.get("clip", [-100.0, 100.0]))),
            flipper_fold_mode_clip=float(self._clip_abs(flipper_fold_mode.get("clip", [-1.0, 1.0]))),
            height_scan_scale=float(height_scan.get("scale", 1.0)),
            height_scan_clip=float(self._clip_abs(height_scan.get("clip", [-1.0, 1.0]))),
        )
        action_cfg = ActionConfig(
            front_clip=(float(front_clip[0]), float(front_clip[1])),
            rear_clip=(float(rear_clip[0]), float(rear_clip[1])),
            flipper_scale=float(flipper_pos.get("scale", 1.0)),
            flipper_offset=float(flipper_pos.get("offset", 0.0)),
            flipper_max_velocity=float(flipper_pos.get("max_velocity", 0.0)),
            right_track_scale=float(right_track_vel.get("scale", 1.0)),
            left_track_scale=float(left_track_vel.get("scale", right_track_vel.get("scale", 1.0))),
            right_track_offset=float(right_track_vel.get("offset", 0.0)),
            left_track_offset=float(left_track_vel.get("offset", right_track_vel.get("offset", 0.0))),
            right_track_clip=(float(right_track_clip[0]), float(right_track_clip[1])),
            left_track_clip=(float(left_track_clip[0]), float(left_track_clip[1])),
            interference_limit=float(flipper_pos.get("interference_limit", 2.0 * math.pi / 3.0)),
            interference_pairs=self._tuple_pairs(flipper_pos.get("interference_pairs"), ((0, 2), (1, 3))),
            use_directional_clip=self._bool_or_default(flipper_pos.get("use_directional_clip"), True),
            directional_clip_speed_threshold=float(flipper_pos.get("directional_clip_speed_threshold", 0.10)),
            directional_clip_fold_min=float(flipper_pos.get("directional_clip_fold_min", math.pi)),
            directional_clip_clear_max=float(flipper_pos.get("directional_clip_clear_max", math.pi / 2.0)),
            directional_clip_rear_indices=self._tuple_ints(flipper_pos.get("directional_clip_rear_indices"), (0, 1)),
            directional_clip_front_indices=self._tuple_ints(flipper_pos.get("directional_clip_front_indices"), (2, 3)),
        )
        return obs_cfg, action_cfg

    @classmethod
    def _resolve_obs_flag(cls, override: Optional[bool], env_cfg: Dict, flag_name: str, default: bool) -> bool:
        if override is not None:
            return bool(override)
        env_value = cls._optional_env_bool(env_cfg.get(flag_name))
        if env_value is not None:
            return env_value
        return bool(default)

    @staticmethod
    def _optional_env_bool(value) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        value_l = str(value).strip().lower()
        if value_l in {"", "auto", "default", "none", "null"}:
            return None
        if value_l in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if value_l in {"0", "false", "f", "no", "n", "off"}:
            return False
        return None

    @staticmethod
    def _tuple_ints(values: Optional[Sequence[int]], default: Tuple[int, ...]) -> Tuple[int, ...]:
        if values is None:
            return default
        parsed = tuple(int(value) for value in values)
        return parsed or default

    @staticmethod
    def _bool_or_default(value, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        value_l = str(value).strip().lower()
        if value_l in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if value_l in {"0", "false", "f", "no", "n", "off"}:
            return False
        return bool(default)

    @staticmethod
    def _tuple_pairs(values: Optional[Sequence[Sequence[int]]], default: Tuple[Tuple[int, int], ...]) -> Tuple[Tuple[int, int], ...]:
        if values is None:
            return default
        parsed = []
        for pair in values:
            if pair is None or len(pair) != 2:
                continue
            parsed.append((int(pair[0]), int(pair[1])))
        return tuple(parsed) or default

    @staticmethod
    def _infer_height_scan_dim(height_scan_cfg: Dict, override: int) -> int:
        if override and override > 0:
            return int(override)
        params = (height_scan_cfg or {}).get("params") or {}
        sensor_cfg = params.get("sensor_cfg") or {}
        pattern_cfg = sensor_cfg.get("pattern_cfg") or {}
        try:
            size = pattern_cfg.get("size")
            resolution = float(pattern_cfg.get("resolution"))
            if size and len(size) == 2 and resolution > 0:
                nx = int(round(float(size[0]) / resolution)) + 1
                ny = int(round(float(size[1]) / resolution)) + 1
                return max(nx * ny, 0)
        except Exception:
            pass
        return 0

    @staticmethod
    def _tuple_range(values: Optional[Sequence[float]], default: Tuple[float, float]) -> Tuple[float, float]:
        if not values or len(values) != 2:
            return default
        return float(values[0]), float(values[1])

    @staticmethod
    def _clip_abs(clip_values: Sequence[float]) -> float:
        return max(abs(float(clip_values[0])), abs(float(clip_values[1])))

    def _load_policy(self, model_path: str, device: str, activation_name: str) -> Tuple[nn.Module, int, int, nn.Module]:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"]
        actor_state = {k: v for k, v in state_dict.items() if k.startswith("actor.")}
        linear_keys = sorted(
            ((name, tensor) for name, tensor in actor_state.items() if name.endswith("weight")),
            key=lambda item: int(item[0].split(".")[1]),
        )
        if not linear_keys:
            raise RuntimeError("No actor weights found in model_state_dict")
        hidden_dims = [tensor.shape[0] for _, tensor in linear_keys[:-1]]
        input_dim = linear_keys[0][1].shape[1]
        action_dim = linear_keys[-1][1].shape[0]
        model = ActorOnly(input_dim=input_dim, action_dim=action_dim, hidden_dims=hidden_dims, activation_name=activation_name)
        model.load_state_dict(actor_state)
        obs_norm_state = checkpoint.get("obs_norm_state_dict")
        if not obs_norm_state:
            obs_normalizer = _ObsNormalizer(enabled=False, device=device)
        else:
            obs_normalizer = _ObsNormalizer.from_state_dict(obs_norm_state, input_dim, device)
        return model.to(device), input_dim, action_dim, obs_normalizer

    def _reconcile_observation_config_with_model(self) -> None:
        expected = int(self.num_actor_obs)
        configured = self._configured_observation_dim()
        if configured == expected:
            return

        target_delta = expected - configured
        candidates = self._observation_toggle_candidates()
        for count in range(1, len(candidates) + 1):
            for combo in itertools.combinations(candidates, count):
                if sum(delta for _, delta in combo) != target_delta:
                    continue
                for attr_name, _ in combo:
                    setattr(self.obs_cfg, attr_name, not bool(getattr(self.obs_cfg, attr_name)))
                return

    def _observation_toggle_candidates(self) -> List[Tuple[str, int]]:
        candidates: List[Tuple[str, int]] = [
            ("has_flipper_fold_mode", 1),
            ("has_base_lin_vel", 3),
        ]
        if self.use_last_action_override is None:
            candidates.append(("has_actions", self.policy_action_dim))
        if self.use_joint_effort_override is None:
            candidates.append(("has_joint_effort", len(self.obs_cfg.flipper_joint_names) + 2))
        if self.use_height_scan_override is None and self.obs_cfg.height_scan_dim > 0:
            candidates.append(("has_height_scan", self.obs_cfg.height_scan_dim))

        return [
            (attr_name, -dim if bool(getattr(self.obs_cfg, attr_name)) else dim)
            for attr_name, dim in candidates
            if dim > 0
        ]

    def _configured_observation_dim(self) -> int:
        dim = 0
        if self.obs_cfg.has_base_lin_vel:
            dim += 3
        dim += 3  # base angular velocity
        dim += 3  # projected gravity
        dim += 3  # velocity command
        if self.obs_cfg.has_flipper_fold_mode:
            dim += 1
        dim += len(self.obs_cfg.flipper_joint_names)
        if self.obs_cfg.include_flipper_joint_vel:
            dim += len(self.obs_cfg.flipper_joint_names)
        dim += 2  # right/left track mean velocity
        if self.obs_cfg.has_actions:
            dim += self.policy_action_dim
        if self.obs_cfg.has_height_scan:
            dim += self.obs_cfg.height_scan_dim
        if self.obs_cfg.has_joint_effort:
            dim += len(self.obs_cfg.flipper_joint_names) + 2
        return dim

    def handle_request(self, req: Dict) -> Dict:
        direction = self._drive_direction_value(req)
        policy_req = self._policy_frame_request(req)
        obs = self._build_observation(policy_req)
        with torch.no_grad():
            action = self.policy(obs.unsqueeze(0)).squeeze(0).cpu()
        raw_action = self._sanitize_policy_action(action)
        if self.last_action_smooth_alpha > 0.0:
            alpha = max(0.0, min(0.999, self.last_action_smooth_alpha))
            raw_action = alpha * self._last_action_obs(policy_req) + (1.0 - alpha) * raw_action
        processed_action = self._process_action(raw_action, policy_req)
        processed_action = self._apply_flipper_rate_limit(processed_action, policy_req)
        processed_action = self._action_from_policy_frame(processed_action, direction)
        self.last_action = processed_action
        self.last_action_obs = raw_action
        self.last_action_obs_direction = direction

        rr, rl, fr, fl, omega_r, omega_l = [float(v) for v in processed_action.tolist()]
        eff_omega_r = omega_r * self.right_track_sign
        eff_omega_l = omega_l * self.left_track_sign
        return {
            "flippers": {"rr": rr, "rl": rl, "fr": fr, "fl": fl},
            "cmd_out": {
                "linear_x": 0.5 * self.sprocket_radius * (eff_omega_r + eff_omega_l),
                "angular_z": self.sprocket_radius * (eff_omega_r - eff_omega_l) / self.track_tread,
            },
        }

    @staticmethod
    def _sanitize_policy_action(action: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(action.float(), nan=0.0, posinf=1.0e6, neginf=-1.0e6)

    def _policy_frame_request(self, req: Dict) -> Dict:
        direction = self._drive_direction_value(req)
        if direction >= 0:
            return dict(req)

        transformed = dict(req)
        transformed["cmd_vel"] = self._reverse_cmd_vel(req.get("cmd_vel", {}))
        transformed["imu_angular_velocity"] = self._reverse_xy_dict(req.get("imu_angular_velocity", {}))
        transformed["flipper_pos"] = self._reverse_flipper_source(req.get("flipper_pos", {}))
        transformed["flipper_joint_vel"] = self._reverse_flipper_source(req.get("flipper_joint_vel", {}))
        transformed["track_joint_vel"] = self._reverse_track_source(req.get("track_joint_vel", {}), aggregate="mean")
        transformed["flipper_joint_effort"] = self._reverse_flipper_source(req.get("flipper_joint_effort", {}))
        transformed["track_joint_effort"] = self._reverse_track_source(req.get("track_joint_effort", {}), aggregate="sum")
        transformed["height_scan"] = list(reversed(list(req.get("height_scan", []))))
        transformed["drive_direction"] = direction
        return transformed

    def _last_action_obs(self, req: Dict) -> torch.Tensor:
        direction = self._drive_direction_value(req)
        stored_direction = int(getattr(self, "last_action_obs_direction", 1))
        action = self.last_action_obs.clone()
        if direction == stored_direction:
            return action
        return self._reverse_action_tensor(action)

    def _action_from_policy_frame(self, action: torch.Tensor, direction: int) -> torch.Tensor:
        if int(direction) >= 0:
            return action.clone()
        return self._reverse_action_tensor(action)

    def _reverse_flipper_source(self, source: Dict[str, float]) -> Dict[str, float]:
        source_lookup = self._flipper_joint_lookup(source.keys())
        target_lookup = self._flipper_joint_lookup(self.obs_cfg.flipper_joint_names)
        result = dict(source)
        for target_key, target_name in target_lookup.items():
            source_key = FLIPPER_REVERSE_KEY[target_key]
            source_name = source_lookup.get(source_key)
            if source_name is None:
                result[target_name] = 0.0
            else:
                result[target_name] = float(source.get(source_name, 0.0))
        return result

    def _reverse_track_source(self, source: Dict[str, float], aggregate: str) -> Dict[str, float]:
        right_value = self._aggregate_joint_value(source, self.obs_cfg.right_track_vel_names, aggregate)
        left_value = self._aggregate_joint_value(source, self.obs_cfg.left_track_vel_names, aggregate)
        result = dict(source)
        self._write_track_group(result, self.obs_cfg.right_track_vel_names, -left_value, aggregate)
        self._write_track_group(result, self.obs_cfg.left_track_vel_names, -right_value, aggregate)
        return result

    @staticmethod
    def _write_track_group(target: Dict[str, float], joint_names: Sequence[str], value: float, aggregate: str) -> None:
        if not joint_names:
            return
        per_joint = float(value) / len(joint_names) if aggregate == "sum" else float(value)
        for joint_name in joint_names:
            target[joint_name] = per_joint

    @staticmethod
    def _aggregate_joint_value(source: Dict[str, float], joint_names: Sequence[str], aggregate: str) -> float:
        if not joint_names:
            return 0.0
        values = [float(source.get(joint_name, 0.0)) for joint_name in joint_names]
        if aggregate == "sum":
            return sum(values)
        return sum(values) / len(values)

    @staticmethod
    def _reverse_cmd_vel(cmd: Dict[str, float]) -> Dict[str, float]:
        return {
            "linear_x": -float(cmd.get("linear_x", 0.0)),
            "linear_y": -float(cmd.get("linear_y", 0.0)),
            "angular_z": float(cmd.get("angular_z", 0.0)),
        }

    @staticmethod
    def _reverse_xy_dict(values: Dict[str, float]) -> Dict[str, float]:
        return {
            "x": -float(values.get("x", 0.0)),
            "y": -float(values.get("y", 0.0)),
            "z": float(values.get("z", 0.0)),
        }

    @staticmethod
    def _reverse_action_tensor(action: torch.Tensor) -> torch.Tensor:
        reversed_action = action.clone()
        if action.numel() == 2:
            reversed_action[0] = -action[1]
            reversed_action[1] = -action[0]
            return reversed_action
        if action.numel() >= 6:
            reversed_action[0] = action[3]
            reversed_action[1] = action[2]
            reversed_action[2] = action[1]
            reversed_action[3] = action[0]
            reversed_action[4] = -action[5]
            reversed_action[5] = -action[4]
            return reversed_action
        if action.numel() >= 4:
            reversed_action[0] = action[3]
            reversed_action[1] = action[2]
            reversed_action[2] = action[1]
            reversed_action[3] = action[0]
        return reversed_action

    @staticmethod
    def _flipper_joint_lookup(joint_names) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for joint_name in joint_names:
            normalized = str(joint_name).split("/")[-1]
            for key in ("rr", "rl", "fr", "fl"):
                if normalized == key or normalized.startswith(f"{key}_"):
                    lookup[key] = joint_name
                    break
        return lookup

    @staticmethod
    def _drive_direction_value(req: Dict) -> int:
        value = req.get("drive_direction", 1)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"rear", "reverse", "backward", "back", "-1"}:
                return -1
            return 1
        try:
            return -1 if float(value) < 0.0 else 1
        except (TypeError, ValueError):
            return 1

    def _build_observation(self, req: Dict) -> torch.Tensor:
        terms: List[torch.Tensor] = []
        if self.obs_cfg.has_base_lin_vel:
            terms.append(self._base_lin_vel_obs(req))
        terms.extend(
            [
                self._base_ang_vel_obs(req),
                self._projected_gravity_obs(req),
                self._velocity_command_obs(req),
            ]
        )
        if self.obs_cfg.has_flipper_fold_mode:
            terms.append(self._flipper_fold_mode_obs(req))
        terms.extend(
            [
                self._flipper_position_obs(req),
                self._joint_velocity_obs(req),
            ]
        )
        if self.obs_cfg.has_actions:
            if self._is_zero_cmd(req):
                terms.append(torch.zeros_like(self.last_action_obs))
            else:
                terms.append(self._last_action_obs(req))
        if self.obs_cfg.has_height_scan:
            terms.append(self._height_scan_obs(req))
        if self.obs_cfg.has_joint_effort:
            terms.append(self._joint_effort_obs(req))
        obs = torch.cat(terms, dim=0).to(self.device)
        if self.obs_normalizer.enabled:
            obs = self.obs_normalizer(obs.unsqueeze(0)).squeeze(0)
        if obs.numel() != self.num_actor_obs:
            raise ValueError(f"Observation size mismatch: built {obs.numel()} values, model expects {self.num_actor_obs}.")
        return obs

    def _source_joint_obs(self, source: Dict[str, float], joint_names: Sequence[str], scale: float, clip: float) -> torch.Tensor:
        values = []
        for joint_name in joint_names:
            raw = float(source.get(joint_name, 0.0))
            values.append(max(-clip, min(clip, raw)) * scale)
        return torch.tensor(values, dtype=torch.float32)

    def _flipper_position_obs(self, req: Dict) -> torch.Tensor:
        return self._source_joint_obs(req["flipper_pos"], self.obs_cfg.flipper_joint_names, self.obs_cfg.flipper_pos_error_scale, self.obs_cfg.flipper_pos_clip)

    def _joint_velocity_obs(self, req: Dict) -> torch.Tensor:
        measured_right = self._mean_joint_value(req["track_joint_vel"], self.obs_cfg.right_track_vel_names)
        measured_left = self._mean_joint_value(req["track_joint_vel"], self.obs_cfg.left_track_vel_names)
        track_mean_vel = self._source_joint_obs({"right_track": measured_right, "left_track": measured_left}, ("right_track", "left_track"), self.obs_cfg.track_vel_error_scale, self.obs_cfg.track_vel_clip)
        if not self.obs_cfg.include_flipper_joint_vel:
            return track_mean_vel
        flipper_vel = self._source_joint_obs(req["flipper_joint_vel"], self.obs_cfg.flipper_joint_names, self.obs_cfg.track_vel_error_scale, self.obs_cfg.track_vel_clip)
        return torch.cat((flipper_vel, track_mean_vel), dim=0)

    def _joint_effort_obs(self, req: Dict) -> torch.Tensor:
        flipper_effort = self._source_joint_obs(req["flipper_joint_effort"], self.obs_cfg.flipper_joint_names, self.obs_cfg.joint_effort_scale, self.obs_cfg.joint_effort_clip)
        right_effort = sum(float(req["track_joint_effort"].get(name, 0.0)) for name in self.obs_cfg.right_track_vel_names)
        left_effort = sum(float(req["track_joint_effort"].get(name, 0.0)) for name in self.obs_cfg.left_track_vel_names)
        track_effort = self._source_joint_obs({"right_track": right_effort, "left_track": left_effort}, ("right_track", "left_track"), self.obs_cfg.joint_effort_scale, self.obs_cfg.joint_effort_clip)
        return torch.cat((flipper_effort, track_effort), dim=0)

    def _height_scan_obs(self, req: Dict) -> torch.Tensor:
        values = list(req["height_scan"])
        if self.obs_cfg.height_scan_dim:
            values = (values + [0.0] * self.obs_cfg.height_scan_dim)[: self.obs_cfg.height_scan_dim]
        clipped = [max(-self.obs_cfg.height_scan_clip, min(self.obs_cfg.height_scan_clip, v)) for v in values]
        return torch.tensor([v * self.obs_cfg.height_scan_scale for v in clipped], dtype=torch.float32)

    @staticmethod
    def _mean_joint_value(source: Dict[str, float], joint_names: Sequence[str]) -> float:
        if not joint_names:
            return 0.0
        values = [float(source.get(joint_name, 0.0)) for joint_name in joint_names]
        return sum(values) / len(values)

    def _effective_track_omegas(self, req: Dict) -> Tuple[float, float]:
        omega_r = self._mean_joint_value(req["track_joint_vel"], self.obs_cfg.right_track_vel_names) * self.right_track_sign
        omega_l = self._mean_joint_value(req["track_joint_vel"], self.obs_cfg.left_track_vel_names) * self.left_track_sign
        return omega_r, omega_l

    def _base_lin_vel_obs(self, req: Dict) -> torch.Tensor:
        omega_r, omega_l = self._effective_track_omegas(req)
        vx = 0.5 * self.sprocket_radius * (omega_r + omega_l)
        return torch.tensor([max(-self.obs_cfg.base_lin_vel_clip, min(self.obs_cfg.base_lin_vel_clip, vx)) * self.obs_cfg.base_lin_vel_scale, 0.0, 0.0], dtype=torch.float32)

    def _base_ang_vel_obs(self, req: Dict) -> torch.Tensor:
        raw = list(self._imu_angular_velocity_in_robot_frame(req))
        return torch.tensor([max(-self.obs_cfg.base_ang_vel_clip, min(self.obs_cfg.base_ang_vel_clip, v)) * self.obs_cfg.base_ang_vel_scale for v in raw], dtype=torch.float32)

    def _velocity_command_obs(self, req: Dict) -> torch.Tensor:
        raw = list(self._clamped_velocity_command(req))
        return torch.tensor([max(-self.obs_cfg.velocity_cmd_clip, min(self.obs_cfg.velocity_cmd_clip, v)) for v in raw], dtype=torch.float32)

    def _flipper_fold_mode_obs(self, req: Dict) -> torch.Tensor:
        raw = float(req.get("flipper_fold_mode", 0.0))
        clipped = max(-self.obs_cfg.flipper_fold_mode_clip, min(self.obs_cfg.flipper_fold_mode_clip, raw))
        return torch.tensor([clipped * self.obs_cfg.flipper_fold_mode_scale], dtype=torch.float32)

    def _clamped_velocity_command(self, req: Dict) -> Tuple[float, float, float]:
        cmd = req["cmd_vel"]
        return (
            self._clamp(float(cmd["linear_x"]), self.obs_cfg.cmd_lin_x_range),
            self._clamp(float(cmd["linear_y"]), self.obs_cfg.cmd_lin_y_range),
            self._clamp(float(cmd["angular_z"]), self.obs_cfg.cmd_ang_z_range),
        )

    @staticmethod
    def _clamp(value: float, limits: Tuple[float, float]) -> float:
        return max(float(limits[0]), min(float(limits[1]), value))

    def _projected_gravity_obs(self, req: Dict) -> torch.Tensor:
        quat = self._imu_orientation_in_robot_frame(req)
        gravity_body = self._rotate_world_to_body((0.0, 0.0, -1.0), quat)
        if self._drive_direction_value(req) < 0:
            gravity_body = (-gravity_body[0], -gravity_body[1], gravity_body[2])
        return torch.tensor([max(-self.obs_cfg.gravity_clip, min(self.obs_cfg.gravity_clip, v)) * self.obs_cfg.projected_gravity_scale for v in gravity_body], dtype=torch.float32)

    def _imu_angular_velocity_in_robot_frame(self, req: Dict) -> Tuple[float, float, float]:
        ang = req["imu_angular_velocity"]
        raw = (float(ang["x"]), float(ang["y"]), float(ang["z"]))
        if self._imu_payload_is_robot_frame(req):
            return raw
        return self._rotate_z(raw, self.imu_yaw_offset)

    def _imu_orientation_in_robot_frame(self, req: Dict) -> Tuple[float, float, float, float]:
        q = req["imu_orientation"]
        world_from_imu = self._normalize_quaternion((float(q["w"]), float(q["x"]), float(q["y"]), float(q["z"])))
        if self._imu_payload_is_robot_frame(req):
            return world_from_imu
        imu_from_robot = self._yaw_quaternion(-self.imu_yaw_offset)
        return self._normalize_quaternion(self._quat_multiply(world_from_imu, imu_from_robot))

    @staticmethod
    def _imu_payload_is_robot_frame(req: Dict) -> bool:
        frame = str(req.get("imu_frame", "sensor")).strip().lower()
        return frame in {"robot", "body", "base", "chassis/body"}

    @staticmethod
    def _rotate_z(vector: Tuple[float, float, float], yaw: float) -> Tuple[float, float, float]:
        x, y, z = vector
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        return (cos_yaw * x - sin_yaw * y, sin_yaw * x + cos_yaw * y, z)

    @staticmethod
    def _yaw_quaternion(yaw: float) -> Tuple[float, float, float, float]:
        half_yaw = 0.5 * yaw
        return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))

    @staticmethod
    def _normalize_quaternion(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        norm = math.sqrt(sum(v * v for v in quat))
        if norm < 1.0e-9:
            return 1.0, 0.0, 0.0, 0.0
        return tuple(v / norm for v in quat)

    @staticmethod
    def _quat_conjugate(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, x, y, z = quat
        return (w, -x, -y, -z)

    @staticmethod
    def _quat_multiply(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    def _rotate_world_to_body(self, vector: Tuple[float, float, float], world_from_body_quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        vec_quat = (0.0, vector[0], vector[1], vector[2])
        body_from_world = self._quat_conjugate(world_from_body_quat)
        rotated = self._quat_multiply(self._quat_multiply(body_from_world, vec_quat), world_from_body_quat)
        return rotated[1], rotated[2], rotated[3]

    def _process_action(self, action: torch.Tensor, req: Dict) -> torch.Tensor:
        if action.numel() == 2:
            processed = self._current_flipper_hold_action(req)
            processed[4] = self._scaled_clipped_action(
                action[0],
                self.action_cfg.right_track_scale,
                self.action_cfg.right_track_offset,
                self.action_cfg.right_track_clip,
            )
            processed[5] = self._scaled_clipped_action(
                action[1],
                self.action_cfg.left_track_scale,
                self.action_cfg.left_track_offset,
                self.action_cfg.left_track_clip,
            )
            if self._is_zero_cmd(req):
                processed[4] = 0.0
                processed[5] = 0.0
            return processed

        if action.numel() != 6:
            raise ValueError(f"Unsupported policy action dimension: {action.numel()}. Expected 2 or 6.")

        processed = action.clone().float()
        processed[0] = self._scaled_clipped_action(processed[0], self.action_cfg.flipper_scale, self.action_cfg.flipper_offset, self.action_cfg.rear_clip)
        processed[1] = self._scaled_clipped_action(processed[1], self.action_cfg.flipper_scale, self.action_cfg.flipper_offset, self.action_cfg.rear_clip)
        processed[2] = self._scaled_clipped_action(processed[2], self.action_cfg.flipper_scale, self.action_cfg.flipper_offset, self.action_cfg.front_clip)
        processed[3] = self._scaled_clipped_action(processed[3], self.action_cfg.flipper_scale, self.action_cfg.flipper_offset, self.action_cfg.front_clip)
        processed[4] = self._scaled_clipped_action(processed[4], self.action_cfg.right_track_scale, self.action_cfg.right_track_offset, self.action_cfg.right_track_clip)
        processed[5] = self._scaled_clipped_action(processed[5], self.action_cfg.left_track_scale, self.action_cfg.left_track_offset, self.action_cfg.left_track_clip)
        if self._is_zero_cmd(req):
            processed = self._safe_zero_cmd_action(req)
        else:
            if self.use_directional_flipper_constraint:
                processed = self._apply_directional_flipper_clip(processed, req)
            processed = self._constrain_flipper_interference(processed, req)
        return processed

    def _apply_flipper_rate_limit(self, action: torch.Tensor, req: Dict) -> torch.Tensor:
        if action.numel() < 4 or not self.use_flipper_rate_limit:
            return action

        max_velocity = float(self.action_cfg.flipper_max_velocity)
        if not math.isfinite(max_velocity) or max_velocity <= 0.0:
            return action

        now = time.monotonic()
        current = torch.tensor(self._measured_flipper_positions(req, action), dtype=torch.float32)
        direction = self._drive_direction_value(req)
        if self._is_zero_cmd(req):
            self.flipper_rate_limited_targets = current.clone()
            self.flipper_rate_limit_time = now
            self.flipper_rate_limited_targets_direction = direction
            return action

        if self.flipper_rate_limited_targets is None:
            self.flipper_rate_limited_targets = current.clone()
            self.flipper_rate_limited_targets_direction = direction
            dt = 1.0 / self.policy_rate
        else:
            last_time = self.flipper_rate_limit_time
            dt = 1.0 / self.policy_rate if last_time is None else max(0.0, now - last_time)

        self.flipper_rate_limit_time = now
        if self.flipper_rate_limit_dt_max > 0.0:
            dt = min(dt, self.flipper_rate_limit_dt_max)

        max_delta = max_velocity * dt
        limited = action.clone()
        previous = self._flipper_rate_limited_targets_for_direction(direction)
        desired = limited[:4].to(dtype=torch.float32)
        next_targets = previous + torch.clamp(desired - previous, min=-max_delta, max=max_delta)
        limited[:4] = next_targets
        self.flipper_rate_limited_targets = next_targets.clone()
        self.flipper_rate_limited_targets_direction = direction
        return limited

    def _flipper_rate_limited_targets_for_direction(self, direction: int) -> torch.Tensor:
        previous = self.flipper_rate_limited_targets.to(dtype=torch.float32)
        stored_direction = int(getattr(self, "flipper_rate_limited_targets_direction", direction))
        if int(direction) == stored_direction:
            return previous
        return self._reverse_action_tensor(previous)

    @staticmethod
    def _scaled_clipped_action(value: torch.Tensor, scale: float, offset: float, limits: Tuple[float, float]) -> float:
        processed = float(value) * float(scale) + float(offset)
        return max(float(limits[0]), min(float(limits[1]), processed))

    def _apply_directional_flipper_clip(self, action: torch.Tensor, req: Dict) -> torch.Tensor:
        mode = self._flipper_fold_mode_value(req)
        if mode is not None:
            self._clear_pending_travel_direction()
            if mode == 0:
                return action.clone()
            return self._clip_flipper_fold_direction(action, self._flipper_fold_clip_direction(mode, req))

        cmd = self._clamped_velocity_command(req)
        if self._is_turn_in_place_cmd(cmd):
            self._clear_pending_travel_direction()
            return action.clone()

        self._update_travel_direction(cmd)

        return self._clip_flipper_fold_direction(action, self.travel_direction)

    def _flipper_fold_clip_direction(self, mode: int, req: Dict) -> int:
        if self._drive_direction_value(req) < 0:
            return -int(mode)
        return int(mode)

    def _flipper_fold_mode_value(self, req: Dict) -> Optional[int]:
        if not self.obs_cfg.has_flipper_fold_mode:
            return None
        raw = float(req.get("flipper_fold_mode", 0.0))
        if raw > 0.5:
            return 1
        if raw < -0.5:
            return -1
        return 0

    def _clip_flipper_fold_direction(self, action: torch.Tensor, direction: int) -> torch.Tensor:
        clipped = action.clone()
        fold_min = float(self.action_cfg.directional_clip_fold_min)
        clear_max = min(
            float(self.action_cfg.directional_clip_clear_max),
            float(self.action_cfg.interference_limit) - 1.0e-4,
        )
        if int(direction) > 0:
            fold_indices = self.action_cfg.directional_clip_front_indices
            clear_indices = self.action_cfg.directional_clip_rear_indices
        else:
            fold_indices = self.action_cfg.directional_clip_rear_indices
            clear_indices = self.action_cfg.directional_clip_front_indices
        for idx in fold_indices:
            if 0 <= int(idx) < min(4, clipped.numel()):
                clipped[int(idx)] = max(fold_min, float(clipped[int(idx)]))
        for idx in clear_indices:
            if 0 <= int(idx) < min(4, clipped.numel()):
                clipped[int(idx)] = min(clear_max, float(clipped[int(idx)]))
        return clipped

    def _update_travel_direction(self, cmd: Tuple[float, float, float]) -> None:
        threshold = float(self.action_cfg.directional_clip_speed_threshold)
        vx = float(cmd[0])
        if vx >= threshold:
            candidate = 1
        elif vx <= -threshold:
            candidate = -1
        else:
            self._clear_pending_travel_direction()
            return

        if candidate == self.travel_direction:
            self._clear_pending_travel_direction()
            return

        duration = max(0.0, float(self.directional_flipper_switch_duration))
        if duration <= 0.0:
            self.travel_direction = candidate
            self._clear_pending_travel_direction()
            return

        now = time.monotonic()
        if self.pending_travel_direction != candidate:
            self.pending_travel_direction = candidate
            self.pending_travel_direction_since = now
            return
        if self.pending_travel_direction_since is not None and (now - self.pending_travel_direction_since) >= duration:
            self.travel_direction = candidate
            self._clear_pending_travel_direction()

    def _clear_pending_travel_direction(self) -> None:
        self.pending_travel_direction = None
        self.pending_travel_direction_since = None

    def _is_turn_in_place_cmd(self, cmd: Tuple[float, float, float]) -> bool:
        angular_threshold = max(0.0, float(self.directional_flipper_turn_angular_threshold))
        radius_threshold = max(0.0, float(self.directional_flipper_turn_radius_threshold))
        vx = abs(float(cmd[0]))
        wz = abs(float(cmd[2]))
        if angular_threshold <= 0.0 or radius_threshold <= 0.0:
            return False
        return wz >= angular_threshold and vx <= radius_threshold * wz

    def _constrain_flipper_interference(self, action: torch.Tensor, req: Dict) -> torch.Tensor:
        if action.numel() < 4:
            return action

        constrained = action.clone()
        limit = float(self.action_cfg.interference_limit)
        safe_limit = limit - 1.0e-4
        measured = self._measured_flipper_positions(req, constrained)
        for rear_idx, front_idx in self.action_cfg.interference_pairs:
            if not (0 <= rear_idx < min(4, constrained.numel()) and 0 <= front_idx < min(4, constrained.numel())):
                continue
            rear_measured = measured[rear_idx]
            front_measured = measured[front_idx]
            if rear_measured > limit and float(constrained[front_idx]) > safe_limit:
                constrained[front_idx] = safe_limit
            if front_measured > limit and float(constrained[rear_idx]) > safe_limit:
                constrained[rear_idx] = safe_limit

            rear = float(constrained[rear_idx])
            front = float(constrained[front_idx])
            if rear > safe_limit and front > safe_limit:
                if rear >= front:
                    constrained[front_idx] = safe_limit
                else:
                    constrained[rear_idx] = safe_limit
        return constrained

    def _measured_flipper_positions(self, req: Dict, fallback: torch.Tensor) -> Tuple[float, float, float, float]:
        pos = req["flipper_pos"]
        values = []
        for idx, joint_name in enumerate(self.obs_cfg.flipper_joint_names[:4]):
            values.append(float(pos.get(joint_name, pos.get(joint_name.split("/")[-1], fallback[idx]))))
        while len(values) < 4:
            values.append(float(fallback[len(values)]))
        return values[0], values[1], values[2], values[3]

    def _safe_zero_cmd_action(self, req: Dict) -> torch.Tensor:
        pos = req["flipper_pos"]
        joint_lookup = self._canonical_flipper_joint_lookup()
        return torch.tensor(
            [
                self._measured_flipper_position(pos, joint_lookup, "rr"),
                self._measured_flipper_position(pos, joint_lookup, "rl"),
                self._measured_flipper_position(pos, joint_lookup, "fr"),
                self._measured_flipper_position(pos, joint_lookup, "fl"),
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )

    def _current_flipper_hold_action(self, req: Dict) -> torch.Tensor:
        pos = req["flipper_pos"]
        joint_lookup = self._canonical_flipper_joint_lookup()
        return torch.tensor(
            [
                self._measured_flipper_position(pos, joint_lookup, "rr"),
                self._measured_flipper_position(pos, joint_lookup, "rl"),
                self._measured_flipper_position(pos, joint_lookup, "fr"),
                self._measured_flipper_position(pos, joint_lookup, "fl"),
                0.0,
                0.0,
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _measured_flipper_position(pos: Dict[str, float], joint_lookup: Dict[str, str], key: str) -> float:
        joint_name = joint_lookup.get(key)
        if not joint_name:
            return 0.0
        return float(pos.get(joint_name, pos.get(joint_name.split("/")[-1], 0.0)))

    def _canonical_flipper_joint_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for joint_name in self.obs_cfg.flipper_joint_names:
            normalized = joint_name.split("/")[-1]
            for key in ("rr", "rl", "fr", "fl"):
                if normalized.startswith(f"{key}_"):
                    lookup[key] = joint_name
                    break
        return lookup

    def _is_zero_cmd(self, req: Dict) -> bool:
        cmd = self._clamped_velocity_command(req)
        return (
            abs(float(cmd[0])) < self.cmd_deadband_linear
            and abs(float(cmd[1])) < self.cmd_deadband_linear
            and abs(float(cmd[2])) < self.cmd_deadband_angular
        )


def _serve(args: argparse.Namespace) -> None:
    backend = Backend(args)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.ipc_host, int(args.ipc_port)))
    server.listen(1)
    print(f"[backend] listening on {args.ipc_host}:{args.ipc_port}", flush=True)
    while True:
        conn, _ = server.accept()
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8")
            for line in reader:
                req = json.loads(line)
                resp = backend.handle_request(req)
                writer.write(json.dumps(resp) + "\n")
                writer.flush()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--env_yaml_path", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--track_tread", type=float, default=0.56)
    p.add_argument("--sprocket_radius", type=float, default=0.0703)
    p.add_argument("--right_track_sign", type=float, default=1.0)
    p.add_argument("--left_track_sign", type=float, default=-1.0)
    p.add_argument("--imu_yaw_offset", type=float, default=0.0)
    p.add_argument("--cmd_deadband_linear", type=float, default=0.05)
    p.add_argument("--cmd_deadband_angular", type=float, default=0.05)
    p.add_argument("--joint_effort_scale", type=float, default=1.0)
    p.add_argument("--last_action_smooth_alpha", type=float, default=0.0)
    p.add_argument("--rate", type=float, default=20.0)
    p.add_argument("--use_flipper_rate_limit", default="auto")
    p.add_argument("--flipper_rate_limit_dt_max", type=float, default=0.1)
    p.add_argument("--directional_flipper_switch_duration", type=float, default=1.0)
    p.add_argument("--directional_flipper_turn_angular_threshold", type=float, default=0.2)
    p.add_argument("--directional_flipper_turn_radius_threshold", type=float, default=0.25)
    p.add_argument("--use_last_action", default="")
    p.add_argument("--use_joint_effort", default="")
    p.add_argument("--use_height_scan", default="")
    p.add_argument("--use_directional_flipper_constraint", default="auto")
    p.add_argument("--height_scan_dim", type=int, default=0)
    p.add_argument("--activation", default="elu")
    p.add_argument("--ipc_host", default="127.0.0.1")
    p.add_argument("--ipc_port", type=int, default=29001)
    p.add_argument("--flipper_joint_names", default=str(list(DEFAULT_FLIPPER_JOINT_NAMES)))
    p.add_argument("--right_track_vel_names", default=str(list(DEFAULT_RIGHT_TRACK_VEL_NAMES)))
    p.add_argument("--left_track_vel_names", default=str(list(DEFAULT_LEFT_TRACK_VEL_NAMES)))
    return p.parse_args()


def main() -> None:
    _serve(_parse_args())


if __name__ == "__main__":
    main()
