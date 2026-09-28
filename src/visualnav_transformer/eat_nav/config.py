"""Versioned configuration shared by EAT collection, training, and inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EATConfig:
    context_size: int = 3
    dem_history_interval_sec: float = 0.25
    len_traj_pred: int = 8
    trajectory_interval_sec: float = 0.25
    dem_height_cells: int = 20
    dem_width_cells: int = 20
    dem_resolution_m: float = 0.1
    dem_min_valid_fraction: float = 0.25
    goal_max_radius_m: float = 0.8
    reference_frame: str = "map"
    base_frame: str = "chassis/body"
    quality_enabled: bool = True
    quality_target: float = 0.95
    max_epochs: int = 100
    validation_metric: str = "diffusion_loss"
    validation_interval_epochs: int = 1
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    best_checkpoint_metric_mode: str = "min"
    best_checkpoint_weights: str = "ema"
    overlap_min_mask_iou: float = 0.8
    overlap_max_height_rmse_m: float = 0.05
    overlap_max_goal_distance_m: float = 0.1
    overlap_min_distinct_anchors: int = 50
    overlap_min_anchor_fraction: float = 0.05

    def __post_init__(self) -> None:
        if type(self.context_size) is not int or self.context_size not in (3, 5):
            raise ValueError("context_size must be 3 or 5")
        for name in ("len_traj_pred", "dem_height_cells", "dem_width_cells", "max_epochs",
                     "validation_interval_epochs", "early_stopping_patience",
                     "overlap_min_distinct_anchors"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dem_history_interval_sec", "trajectory_interval_sec",
                     "dem_resolution_m", "goal_max_radius_m", "overlap_max_height_rmse_m",
                     "overlap_max_goal_distance_m"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("dem_min_valid_fraction", "quality_target",
                     "overlap_min_mask_iou", "overlap_min_anchor_fraction"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not isfinite(self.early_stopping_min_delta) or self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be nonnegative")
        if not self.reference_frame or not self.base_frame:
            raise ValueError("reference_frame and base_frame must be nonempty")
        if type(self.quality_enabled) is not bool:
            raise ValueError("quality_enabled must be a bool")
        if self.best_checkpoint_metric_mode not in ("min", "max"):
            raise ValueError("best_checkpoint_metric_mode must be min or max")
        if self.best_checkpoint_weights not in ("ema", "raw"):
            raise ValueError("best_checkpoint_weights must be ema or raw")

    @property
    def dem_frame_count(self) -> int:
        return self.context_size + 1

    @property
    def trajectory_horizon_sec(self) -> float:
        return self.len_traj_pred * self.trajectory_interval_sec

    def checkpoint_metadata(self) -> dict[str, Any]:
        names = (
            "context_size", "dem_history_interval_sec", "len_traj_pred",
            "trajectory_interval_sec", "dem_height_cells", "dem_width_cells",
            "dem_resolution_m", "goal_max_radius_m", "quality_enabled",
        )
        result = {name: getattr(self, name) for name in names}
        result["trajectory_horizon_sec"] = self.trajectory_horizon_sec
        result["schema_version"] = SCHEMA_VERSION
        return result


def load_eat_config(path: Path) -> EATConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("EAT config must be a mapping")
    raw = dict(raw)
    if "quality_conditioning" in raw:
        quality = raw.pop("quality_conditioning")
        if not isinstance(quality, dict) or set(quality) != {"enabled"}:
            raise ValueError("quality_conditioning must contain only enabled")
        if "quality_enabled" in raw:
            raise ValueError("quality_enabled given twice")
        raw["quality_enabled"] = quality["enabled"]
    explicit_horizon = raw.pop("trajectory_horizon_sec", None)
    known = {field.name for field in fields(EATConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown EAT config keys: {sorted(unknown)}")
    config = EATConfig(**raw)
    if explicit_horizon is not None:
        if not isinstance(explicit_horizon, (int, float)) or not isfinite(explicit_horizon):
            raise ValueError("trajectory_horizon_sec must be finite")
        if not isclose(explicit_horizon, config.trajectory_horizon_sec, rel_tol=0, abs_tol=1e-9):
            raise ValueError("trajectory_horizon_sec must equal len_traj_pred * trajectory_interval_sec")
    return config


def assert_checkpoint_compatible(config: EATConfig, metadata: Mapping[str, Any]) -> None:
    expected = config.checkpoint_metadata()
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {metadata.get('schema_version')!r}")
    for name, wanted in expected.items():
        actual = metadata.get(name)
        if isinstance(wanted, float):
            matches = isinstance(actual, (int, float)) and isclose(
                actual, wanted, rel_tol=0, abs_tol=1e-9
            )
        else:
            matches = actual == wanted and type(actual) is type(wanted)
        if not matches:
            raise ValueError(f"checkpoint {name} mismatch: expected {wanted!r}, got {actual!r}")
