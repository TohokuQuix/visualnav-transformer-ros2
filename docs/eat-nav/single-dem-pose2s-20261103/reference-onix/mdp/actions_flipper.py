from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import configclass


def constrain_flipper_interference(
    targets: torch.Tensor,
    interference_limit: float,
    pairs: tuple[tuple[int, int], ...] = ((0, 2), (1, 3)),
    measured_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prevent front/rear flippers on the same side from both folding inward."""
    if targets.shape[-1] < 4:
        return targets

    constrained = targets.clone()
    limit = float(interference_limit)
    safe_limit = limit - 1.0e-4
    for rear_idx, front_idx in pairs:
        if measured_positions is not None:
            rear_measured = measured_positions[:, rear_idx]
            front_measured = measured_positions[:, front_idx]
            front_blocked = (rear_measured > limit) & (constrained[:, front_idx] > safe_limit)
            rear_blocked = (front_measured > limit) & (constrained[:, rear_idx] > safe_limit)
            constrained[:, front_idx] = torch.where(
                front_blocked,
                torch.full_like(constrained[:, front_idx], safe_limit),
                constrained[:, front_idx],
            )
            constrained[:, rear_idx] = torch.where(
                rear_blocked,
                torch.full_like(constrained[:, rear_idx], safe_limit),
                constrained[:, rear_idx],
            )

        rear = constrained[:, rear_idx]
        front = constrained[:, front_idx]
        both_folded = (rear > safe_limit) & (front > safe_limit)
        rear_wins = rear >= front

        clamp_front = both_folded & rear_wins
        clamp_rear = both_folded & ~rear_wins
        constrained[:, front_idx] = torch.where(clamp_front, torch.full_like(front, safe_limit), front)
        constrained[:, rear_idx] = torch.where(clamp_rear, torch.full_like(rear, safe_limit), rear)

    return constrained


@configclass
class RateLimitedJointPositionActionCfg(JointPositionActionCfg):
    """Joint position action with a per-step target-rate limit."""

    class_type = None  # assigned after class definition
    max_velocity: float = 2.0
    interference_limit: float = 2.09439510239
    interference_pairs: tuple[tuple[int, int], ...] = ((0, 2), (1, 3))
    use_directional_clip: bool = True
    directional_clip_command_name: str = "base_velocity"
    directional_clip_speed_threshold: float = 0.10
    directional_clip_fold_min: float = 1.57079632679
    directional_clip_clear_max: float = 1.57079632679
    directional_clip_mode_deadband: float = 0.5
    directional_clip_rear_indices: tuple[int, int] = (0, 1)
    directional_clip_front_indices: tuple[int, int] = (2, 3)


class RateLimitedJointPositionAction(JointPositionAction):
    """Clamp position target changes to emulate a real joint velocity limit."""

    cfg: RateLimitedJointPositionActionCfg

    def __init__(self, cfg: RateLimitedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation
        self._previous_targets = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._pre_clip_processed_actions = self._processed_actions.clone()
        self._max_delta = float(self.cfg.max_velocity) * float(env.step_dt)
        self._travel_direction = torch.ones(env.num_envs, device=self.device, dtype=torch.int8)

    @property
    def pre_clip_processed_actions(self) -> torch.Tensor:
        return self._pre_clip_processed_actions

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._pre_clip_processed_actions = self._processed_actions.clone()
        self._apply_directional_clip()
        self._processed_actions = constrain_flipper_interference(
            self._processed_actions,
            self.cfg.interference_limit,
            pairs=self.cfg.interference_pairs,
            measured_positions=self._asset.data.joint_pos[:, self._joint_ids],
        )

    def apply_actions(self):
        desired_targets = self.processed_actions
        latency_fn = getattr(self._env, "apply_action_latency", None)
        if callable(latency_fn):
            desired_targets = latency_fn("flipper_pos", desired_targets)
        delta = torch.clamp(desired_targets - self._previous_targets, min=-self._max_delta, max=self._max_delta)
        limited_targets = self._previous_targets + delta
        limited_targets = self._apply_directional_clip(limited_targets)
        limited_targets = constrain_flipper_interference(
            limited_targets,
            self.cfg.interference_limit,
            pairs=self.cfg.interference_pairs,
            measured_positions=self._asset.data.joint_pos[:, self._joint_ids],
        )
        self._asset.set_joint_position_target(limited_targets, joint_ids=self._joint_ids)
        self._previous_targets.copy_(limited_targets)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._previous_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._pre_clip_processed_actions[env_ids] = self._previous_targets[env_ids]

    def _update_travel_direction(self) -> None:
        command = self._env.command_manager.get_command(self.cfg.directional_clip_command_name)
        vx = command[:, 0]
        threshold = float(self.cfg.directional_clip_speed_threshold)
        self._travel_direction = torch.where(
            vx > threshold,
            torch.ones_like(self._travel_direction),
            torch.where(vx < -threshold, -torch.ones_like(self._travel_direction), self._travel_direction),
        )

    def _apply_directional_clip(self, targets: torch.Tensor | None = None) -> torch.Tensor:
        if not bool(self.cfg.use_directional_clip):
            if targets is None:
                return self._processed_actions
            return targets

        if targets is None:
            targets = self._processed_actions

        clipped = targets.clone()
        fold_min = float(self.cfg.directional_clip_fold_min)
        clear_max = min(float(self.cfg.directional_clip_clear_max), float(self.cfg.interference_limit) - 1.0e-4)
        mode = getattr(self._env, "_onix_flipper_fold_mode", None)
        if mode is None:
            self._update_travel_direction()
            forward = self._travel_direction > 0
            reverse = self._travel_direction < 0
        else:
            mode = mode.to(device=self.device, dtype=torch.float32).view(-1)
            deadband = float(self.cfg.directional_clip_mode_deadband)
            forward = mode > deadband
            reverse = mode < -deadband

        for idx in self.cfg.directional_clip_front_indices:
            clipped[:, idx] = torch.where(forward, torch.clamp(clipped[:, idx], min=fold_min), clipped[:, idx])
            clipped[:, idx] = torch.where(reverse, torch.clamp(clipped[:, idx], max=clear_max), clipped[:, idx])
        for idx in self.cfg.directional_clip_rear_indices:
            clipped[:, idx] = torch.where(reverse, torch.clamp(clipped[:, idx], min=fold_min), clipped[:, idx])
            clipped[:, idx] = torch.where(forward, torch.clamp(clipped[:, idx], max=clear_max), clipped[:, idx])

        if targets is self._processed_actions:
            self._processed_actions = clipped
        return clipped


RateLimitedJointPositionActionCfg.class_type = RateLimitedJointPositionAction
