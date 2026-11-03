from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def foot_contact_force_dirs(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return the contact forces of the feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    force_magnitude = torch.norm(forces, dim=2)
    eps = 1e-10
    force_dir = forces / (force_magnitude.unsqueeze(2) + eps)
    zero_dir = torch.tensor([0.0, 0.0, 0.0], dtype=forces.dtype, device=forces.device)
    # if the force is zero, the direction is 0,0,0
    force_dir[force_magnitude < eps] = zero_dir
    return force_dir.flatten(1)


def foot_contact_forces(
    env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return the contact forces of the feet"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_magnitude = torch.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=2
    )
    return force_magnitude


def foot_friction_coeffs(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Return the contact dynamic and static friction coefficients of the feet

    Note: This implementation only works when each link has a single shape
          For the case where a link has multiple shapes,
          see the implementation in `randomize_rigid_body_material` in events.py
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = torch.tensor(np.array(asset_cfg.body_ids, dtype=np.int_), device="cpu")
    env_ids = torch.arange(env.scene.num_envs, device="cpu")
    materials = asset.root_physx_view.get_material_properties()
    return materials[env_ids][:, body_ids, 0:2].flatten(1).to(env.device)


def contact_count(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return the count of bodies whose contact force exceeds the threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_magnitude = torch.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=2
    )
    return (force_magnitude > threshold).sum(dim=1, keepdim=True).to(force_magnitude.dtype)


def contact_binary(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return per-body binary contacts as a float tensor."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_magnitude = torch.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=2
    )
    return (force_magnitude > threshold).to(force_magnitude.dtype)


def _joint_data_by_cfg(asset: Articulation, asset_cfg: SceneEntityCfg, tensor: torch.Tensor) -> torch.Tensor:
    return tensor[:, asset_cfg.joint_ids]


def _get_action_term(env: ManagerBasedEnv, term_name: str):
    action_manager = env.action_manager
    getter = getattr(action_manager, "get_term", None)
    if callable(getter):
        try:
            return getter(term_name)
        except Exception:
            pass

    terms = getattr(action_manager, "_terms", None)
    if isinstance(terms, dict) and term_name in terms:
        return terms[term_name]

    term_names = getattr(action_manager, "_term_names", None)
    if terms is not None and term_names is not None:
        for name, term in zip(term_names, terms):
            if name == term_name:
                return term

    raise KeyError(f"Action term not found: {term_name}")


def _processed_action_tensor(env: ManagerBasedEnv, term_name: str) -> torch.Tensor:
    term = _get_action_term(env, term_name)
    try:
        processed = getattr(term, "processed_actions", None)
    except Exception:
        processed = None
    if processed is None:
        processed = getattr(term, "_processed_actions", None)
    if processed is None:
        processed = getattr(term, "_processed", None)
    if processed is None:
        action_dim = getattr(term, "action_dim", None)
        if action_dim is None:
            raise AttributeError(f"Action term does not expose processed actions: {term_name}")
        processed = torch.zeros((env.num_envs, int(action_dim)), device=env.device, dtype=torch.float32)
    return processed


def _joint_effort_tensor(asset: Articulation) -> torch.Tensor:
    data = asset.data
    for attr in (
        "applied_torque",
        "computed_torque",
        "joint_torque",
        "joint_effort",
        "joint_effort_target",
    ):
        if hasattr(data, attr):
            return getattr(data, attr)
    raise AttributeError("No joint effort tensor found on articulation data.")


def joint_vel_with_track_mean(
    env: ManagerBasedEnv,
    flipper_cfg: SceneEntityCfg,
    right_track_cfg: SceneEntityCfg,
    left_track_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return flipper joint velocities plus mean velocity of the right/left crawler groups."""
    asset: Articulation = env.scene[flipper_cfg.name]
    flipper_vel = _joint_data_by_cfg(asset, flipper_cfg, asset.data.joint_vel)
    right_track_vel = _joint_data_by_cfg(asset, right_track_cfg, asset.data.joint_vel).mean(dim=1, keepdim=True)
    left_track_vel = _joint_data_by_cfg(asset, left_track_cfg, asset.data.joint_vel).mean(dim=1, keepdim=True)
    return torch.cat((flipper_vel, right_track_vel, left_track_vel), dim=1)


def joint_effort_with_track_sum(
    env: ManagerBasedEnv,
    flipper_cfg: SceneEntityCfg,
    right_track_cfg: SceneEntityCfg,
    left_track_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return flipper efforts plus summed effort of the right/left crawler groups."""
    asset: Articulation = env.scene[flipper_cfg.name]
    joint_effort = _joint_effort_tensor(asset)
    flipper_effort = _joint_data_by_cfg(asset, flipper_cfg, joint_effort)
    right_track_effort = _joint_data_by_cfg(asset, right_track_cfg, joint_effort).sum(dim=1, keepdim=True)
    left_track_effort = _joint_data_by_cfg(asset, left_track_cfg, joint_effort).sum(dim=1, keepdim=True)
    return torch.cat((flipper_effort, right_track_effort, left_track_effort), dim=1)


def flipper_position_error(
    env: ManagerBasedEnv,
    flipper_cfg: SceneEntityCfg,
    action_term_name: str = "flipper_pos",
) -> torch.Tensor:
    """Return flipper position target minus measured joint position."""
    asset: Articulation = env.scene[flipper_cfg.name]
    target = _processed_action_tensor(env, action_term_name)
    measured = _joint_data_by_cfg(asset, flipper_cfg, asset.data.joint_pos)
    return target - measured


def flipper_fold_mode(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the per-environment flipper fold mode: 0=no limit, 1=front fold, -1=rear fold."""
    mode = getattr(env, "_onix_flipper_fold_mode", None)
    if mode is None:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    if not torch.is_tensor(mode):
        mode = torch.as_tensor(mode, device=env.device, dtype=torch.float32)
    return mode.to(device=env.device, dtype=torch.float32).view(-1, 1)


def control_latency(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the sampled control latency normalized by the configured maximum latency."""
    latency_s = getattr(env, "_onix_latency_seconds", None)
    if latency_s is None:
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    max_latency_s = max(float(getattr(env, "_latency_max_seconds", 1.0)), 1.0e-9)
    return latency_s.to(device=env.device, dtype=torch.float32).view(-1, 1) / max_latency_s


def track_mean_velocity(
    env: ManagerBasedEnv,
    right_track_cfg: SceneEntityCfg,
    left_track_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return mean joint velocity of the right/left crawler groups."""
    asset: Articulation = env.scene[right_track_cfg.name]
    right_measured = _joint_data_by_cfg(asset, right_track_cfg, asset.data.joint_vel).mean(dim=1, keepdim=True)
    left_measured = _joint_data_by_cfg(asset, left_track_cfg, asset.data.joint_vel).mean(dim=1, keepdim=True)
    return torch.cat((right_measured, left_measured), dim=1)


def track_velocity_error(
    env: ManagerBasedEnv,
    right_track_cfg: SceneEntityCfg,
    left_track_cfg: SceneEntityCfg,
    right_action_term_name: str = "right_track_vel",
    left_action_term_name: str = "left_track_vel",
) -> torch.Tensor:
    """Return per-side track velocity target minus measured mean track velocity."""
    asset: Articulation = env.scene[right_track_cfg.name]
    right_target = _processed_action_tensor(env, right_action_term_name)
    left_target = _processed_action_tensor(env, left_action_term_name)
    right_measured = _joint_data_by_cfg(asset, right_track_cfg, asset.data.joint_vel).mean(dim=1, keepdim=True)
    left_measured = _joint_data_by_cfg(asset, left_track_cfg, asset.data.joint_vel).mean(dim=1, keepdim=True)
    return torch.cat((right_target - right_measured, left_target - left_measured), dim=1)
