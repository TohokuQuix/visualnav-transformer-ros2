"""Eight physically separated, geometrically identical terrain branches for smoke."""
from __future__ import annotations

import numpy as np
from isaaclab.terrains.terrain_generator import TerrainGenerator
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import make_collection_cfg


class IdenticalTerrainGenerator(TerrainGenerator):
    def _get_terrain_mesh(self, difficulty, cfg):
        # Base generator uses the same cfg seed; reset legacy global NumPy users too.
        previous = np.random.get_state()
        try:
            np.random.seed(int(self.cfg.seed))
            return super()._get_terrain_mesh(difficulty, cfg)
        finally:
            np.random.set_state(previous)


def make_phase_c_clone_cfg(base_cfg, eat_config, seed=20260929, branches=8,
                           terrain_type='random_rough', difficulty=0.5):
    if branches != 8:
        raise ValueError('Phase C duplicate control requires eight branches')
    cfg = make_collection_cfg(base_cfg, eat_config, seed, branches)
    generator = cfg.scene.terrain.terrain_generator
    if terrain_type not in generator.sub_terrains:
        raise ValueError(f'unknown terrain {terrain_type}')
    selected = generator.sub_terrains[terrain_type]
    selected.proportion = 1.0
    generator.sub_terrains = {terrain_type: selected}
    generator.num_rows = branches
    generator.num_cols = 1
    generator.curriculum = False
    generator.difficulty_range = (float(difficulty), float(difficulty))
    generator.class_type = IdenticalTerrainGenerator
    generator.use_cache = True
    cfg.scene.terrain.max_init_terrain_level = branches - 1
    return cfg
