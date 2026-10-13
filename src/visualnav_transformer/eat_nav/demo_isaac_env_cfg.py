"""Collection-only Onix environment; the trained RL observation groups stay intact."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from math import ceil, sqrt
from typing import Sequence

import numpy as np
import torch
from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.sensors import patterns
from isaaclab.terrains.terrain_generator import TerrainGenerator
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from visualnav_transformer.eat_nav.config import EATConfig
from onix_isaac_lab.tasks.locomotion.velocity import mdp
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import (
    OnixRoughEnvCfg,
    OnixSceneCfg,
)
from onix_isaac_lab.tasks.locomotion.velocity.mdp.commands import (
    TerrainAwareUniformVelocityCommand,
    TerrainAwareUniformVelocityCommandCfg,
)


def _seed(base: int, stream: int, env_id: int) -> int:
    return int(np.random.SeedSequence([base, stream, env_id]).generate_state(1)[0])


@dataclass(frozen=True)
class EATSeedPlan:
    base_seed: int
    num_envs: int

    def __post_init__(self) -> None:
        if self.base_seed < 0 or self.num_envs <= 0:
            raise ValueError("seed must be nonnegative and num_envs positive")

    @property
    def rows(self) -> int:
        return ceil(sqrt(self.num_envs))

    @property
    def cols(self) -> int:
        return ceil(self.num_envs / self.rows)

    @property
    def terrain_tiles(self) -> tuple[tuple[int, int], ...]:
        return tuple((env // self.cols, env % self.cols) for env in range(self.num_envs))

    @property
    def terrain_seeds(self) -> tuple[int, ...]:
        return tuple(_seed(self.base_seed, 0, env) for env in range(self.num_envs))

    @property
    def command_seeds(self) -> tuple[int, ...]:
        return tuple(_seed(self.base_seed, 1, env) for env in range(self.num_envs))

    @property
    def initial_seeds(self) -> tuple[int, ...]:
        return tuple(_seed(self.base_seed, 2, env) for env in range(self.num_envs))


class PiecewiseCommandSampler:
    """Independent per-env command streams with explicit held intervals."""

    def __init__(self, seeds: EATSeedPlan, hold_range_sec: tuple[float, float] = (0.5, 3.0)):
        if not 0 < hold_range_sec[0] <= hold_range_sec[1]:
            raise ValueError("invalid hold range")
        self.hold_range_sec = hold_range_sec
        self.rngs = [np.random.default_rng(seed) for seed in seeds.command_seeds]
        self._states: dict[int, tuple[float, tuple[float, float, float]]] = {}
        self._last_times: dict[int, float] = {}

    def draw_next(self, env_id: int) -> tuple[tuple[float, float, float], float]:
        rng = self.rngs[env_id]
        category = int(rng.integers(0, 6))
        if category == 0:
            vx, omega = rng.uniform(0.2, 1.0), rng.uniform(-0.15, 0.15)
        elif category == 1:
            vx, omega = rng.uniform(-0.8, -0.2), rng.uniform(-0.15, 0.15)
        elif category == 2:
            vx, omega = rng.uniform(0.1, 0.7), rng.uniform(0.15, 0.5)
        elif category == 3:
            vx, omega = rng.uniform(0.1, 0.7), rng.uniform(-0.5, -0.15)
        elif category == 4:
            vx, omega = rng.uniform(-0.05, 0.05), rng.choice((-1, 1)) * rng.uniform(0.15, 0.5)
        else:
            vx, omega = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
        hold = float(rng.uniform(*self.hold_range_sec))
        return (float(vx), 0.0, float(omega)), hold

    def command_at(self, env_id: int, time_s: float) -> tuple[float, float, float]:
        if time_s < self._last_times.get(env_id, -float("inf")):
            raise ValueError("command time cannot move backwards")
        self._last_times[env_id] = time_s
        if env_id not in self._states:
            command, hold = self.draw_next(env_id)
            self._states[env_id] = (hold, command)
        expiry, command = self._states[env_id]
        while time_s >= expiry:
            command, hold = self.draw_next(env_id)
            expiry += hold
        self._states[env_id] = (expiry, command)
        return command


class EATTerrainGenerator(TerrainGenerator):
    """Give each generated tile its own seed, including legacy np.random users."""

    def __init__(self, cfg, device):
        self._tile_index = 0
        self.tile_metadata = []
        super().__init__(cfg, device)
        self.cfg.eat_nav_tile_metadata = list(self.tile_metadata)

    def _get_terrain_mesh(self, difficulty, cfg):
        tile_seed = _seed(int(self.cfg.seed), 0, self._tile_index)
        self._tile_index += 1
        old_cfg_seed = self.cfg.seed
        old_numpy_state = np.random.get_state()
        try:
            self.cfg.seed = tile_seed
            np.random.seed(tile_seed)
            mesh, origin = super()._get_terrain_mesh(difficulty, cfg)
            name = next((key for key, item in self.cfg.sub_terrains.items() if item is cfg), type(cfg).__name__)
            digest = hashlib.sha256()
            for array in (np.ascontiguousarray(mesh.vertices), np.ascontiguousarray(mesh.faces)):
                digest.update(str((array.shape, array.dtype)).encode())
                digest.update(array.tobytes())
            self.tile_metadata.append({"tile_index": self._tile_index - 1, "seed": tile_seed,
                                       "terrain_type": name, "difficulty": float(difficulty),
                                       "geometry_sha256": digest.hexdigest()})
            return mesh, origin
        finally:
            self.cfg.seed = old_cfg_seed
            np.random.set_state(old_numpy_state)


class EATTerrainImporter(TerrainImporter):
    """Keep env i on unique tile i across episode resets."""

    def _compute_env_origins_curriculum(self, num_envs, origins):
        rows, cols = origins.shape[:2]
        if num_envs > rows * cols:
            raise ValueError("not enough terrain tiles for EAT envs")
        ids = torch.arange(num_envs, device=self.device)
        selected = getattr(self.cfg, "eat_nav_selected_level", None)
        if selected is None:
            self.terrain_levels = ids // cols
            self.terrain_types = ids % cols
        else:
            if not 0 <= int(selected) < rows or num_envs > cols:
                raise ValueError("selected terrain row/column mismatch")
            self.terrain_levels = torch.full_like(ids, int(selected))
            self.terrain_types = ids
        self.max_terrain_level = rows
        return origins[self.terrain_levels, self.terrain_types].clone()

    def update_env_origins(self, env_ids, move_up, move_down):
        return


def draw_initial_state(rng: np.random.Generator, pose_range, velocity_range):
    """Draw one env's pose and twist from its independent stream."""
    keys = ("x", "y", "z", "roll", "pitch", "yaw")
    pose = [float(rng.uniform(*pose_range.get(key, (0.0, 0.0)))) for key in keys]
    twist = [float(rng.uniform(*velocity_range.get(key, (0.0, 0.0)))) for key in keys]
    return pose, twist


def reset_root_state_eat(env, env_ids, pose_range, velocity_range, base_seed, asset_cfg=mdp.SceneEntityCfg("robot")):
    """Reset each robot using a persistent per-env RNG, independent of reset order."""
    if not hasattr(env, "eat_nav_initial_rngs"):
        plan = EATSeedPlan(base_seed, env.num_envs)
        env.eat_nav_initial_rngs = [np.random.default_rng(seed) for seed in plan.initial_seeds]
    asset = env.scene[asset_cfg.name]
    root = asset.data.default_root_state[env_ids].clone()
    samples = [draw_initial_state(env.eat_nav_initial_rngs[int(i)], pose_range, velocity_range) for i in env_ids]
    poses = torch.tensor([sample[0] for sample in samples], device=asset.device, dtype=root.dtype)
    twists = torch.tensor([sample[1] for sample in samples], device=asset.device, dtype=root.dtype)
    positions = root[:, :3] + env.scene.env_origins[env_ids] + poses[:, :3]
    delta = math_utils.quat_from_euler_xyz(poses[:, 3], poses[:, 4], poses[:, 5])
    orientations = math_utils.quat_mul(root[:, 3:7], delta)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root[:, 7:13] + twists, env_ids=env_ids)


@configclass
class EATPiecewiseVelocityCommandCfg(TerrainAwareUniformVelocityCommandCfg):
    class_type: type = None
    base_seed: int = 0
    hold_range_sec: tuple[float, float] = (0.5, 3.0)


class EATPiecewiseVelocityCommand(TerrainAwareUniformVelocityCommand):
    cfg: EATPiecewiseVelocityCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sampler = PiecewiseCommandSampler(
            EATSeedPlan(cfg.base_seed, self.num_envs), cfg.hold_range_sec
        )

    def _resample(self, env_ids: Sequence[int]):
        ids = range(self.num_envs) if isinstance(env_ids, slice) else env_ids
        for env_id in ids:
            command, hold = self.sampler.draw_next(int(env_id))
            self.time_left[env_id] = hold
            self.vel_command_b[env_id] = torch.tensor(command, device=self.device)
            self.command_counter[env_id] += 1


EATPiecewiseVelocityCommandCfg.class_type = EATPiecewiseVelocityCommand


@configclass
class OnixEATSceneCfg(OnixSceneCfg):
    height_scanner_eat: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/onix_chassis/chassis_body",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.3)),
        ray_alignment="world",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.9, 1.9]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


def make_collection_cfg(
    base_cfg: OnixRoughEnvCfg, config: EATConfig, seed: int, num_envs: int = 100
) -> OnixRoughEnvCfg:
    """Copy a checkpoint-compatible RL cfg, adding collection-only sensors and RNGs."""
    plan = EATSeedPlan(seed, num_envs)
    cfg = deepcopy(base_cfg)
    cfg.seed = seed
    scene = OnixEATSceneCfg(num_envs=num_envs, env_spacing=cfg.scene.env_spacing)
    for name, value in vars(cfg.scene).items():
        if not name.startswith("_") and name != "height_scanner_eat":
            setattr(scene, name, deepcopy(value))
    scene.num_envs = num_envs
    scene.height_scanner_eat.pattern_cfg.resolution = config.dem_resolution_m
    scene.height_scanner_eat.pattern_cfg.size = [
        (config.dem_width_cells - 1) * config.dem_resolution_m,
        (config.dem_height_cells - 1) * config.dem_resolution_m,
    ]
    scene.height_scanner_eat.update_period = config.dem_history_interval_sec
    scene.terrain.class_type = EATTerrainImporter
    scene.terrain.max_init_terrain_level = plan.rows - 1
    scene.terrain.terrain_generator.class_type = EATTerrainGenerator
    scene.terrain.terrain_generator.num_rows = plan.rows
    scene.terrain.terrain_generator.num_cols = plan.cols
    scene.terrain.terrain_generator.curriculum = False
    scene.terrain.terrain_generator.seed = seed
    cfg.scene = scene
    cfg.events.reset_base.func = reset_root_state_eat
    cfg.events.reset_base.params["base_seed"] = seed
    old = cfg.commands.base_velocity
    ranges = deepcopy(old.ranges)
    ranges.lin_vel_x = (-1.0, 1.0)
    ranges.lin_vel_y = (0.0, 0.0)
    ranges.ang_vel_z = (-0.5, 0.5)
    cfg.commands.base_velocity = EATPiecewiseVelocityCommandCfg(
        class_type=EATPiecewiseVelocityCommand,
        asset_name=old.asset_name,
        ranges=ranges,
        heading_command=False,
        rel_heading_envs=0.0,
        rel_standing_envs=0.0,
        resampling_time_range=(0.5, 3.0),
        hold_range_sec=(0.5, 3.0),
        base_seed=seed,
        debug_vis=False,
    )
    return cfg
