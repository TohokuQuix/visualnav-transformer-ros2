#!/usr/bin/env python3
"""Generate all 35 existing Onix terrain conditions without robot rollout."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=2026101101)
parser.add_argument('--eat-config', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np
import torch
import yaml
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import (
    EATTerrainGenerator, EATTerrainImporter, make_collection_cfg,
)
from visualnav_transformer.eat_nav.demo_terrain import TERRAIN_TYPES, configure_selected_terrain, terrain_band


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as file:
        json.dump(value, file, sort_keys=True, allow_nan=False)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


def main():
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    saved = yaml.load(Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml').read_text(),
                      Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in
               ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
    base = make_collection_cfg(OnixRoughEnvCfg(**toggles), load_eat_config(args.eat_config), args.seed, 1)
    rows = []
    try:
        for terrain_type in TERRAIN_TYPES:
            cfg = deepcopy(base.scene.terrain.terrain_generator)
            importer_cfg = deepcopy(base.scene.terrain)
            configure_selected_terrain(cfg, importer_cfg, terrain_type, 0, 1)
            cfg.seed = args.seed
            generated = EATTerrainGenerator(cfg, device='cuda:0')
            metadata = generated.tile_metadata
            if len(metadata) != 5 or generated.terrain_origins.shape != (5, 1, 3):
                raise ValueError(f'{terrain_type}: expected five generated rows')
            for level in range(5):
                tile = metadata[level]
                lower, upper = terrain_band(level, cfg.difficulty_range)
                if tile['terrain_type'] != terrain_type or not lower <= tile['difficulty'] < upper:
                    raise ValueError(f'{terrain_type}/{level}: wrong type/difficulty')
                importer_cfg.eat_nav_selected_level = level
                standin = type('ImporterStandin', (), {'cfg': importer_cfg, 'device': 'cpu'})()
                origin = EATTerrainImporter._compute_env_origins_curriculum(
                    standin, 1, torch.as_tensor(generated.terrain_origins, dtype=torch.float32))
                if tuple(origin.shape) != (1, 3) or not bool(torch.isfinite(origin).all()) or \
                        int(standin.terrain_levels[0]) != level:
                    raise ValueError(f'{terrain_type}/{level}: wrong selected origin')
                row = {'terrain_type': terrain_type, 'level': level, 'difficulty_band': [lower, upper],
                       'generated_difficulty': tile['difficulty'], 'tile_seed': tile['seed'],
                       'tile_geometry_sha256': tile['geometry_sha256'],
                       'origin': origin[0].tolist(), 'source_tile_index': tile['tile_index'],
                       'generator_difficulty_range': list(cfg.difficulty_range)}
                rows.append(row)
                durable(args.output/f'{terrain_type}-level-{level}.json', row)
        if len(rows) != 35:
            raise ValueError('missing terrain condition')
        durable(args.output/'complete.json', {'status': 'terrain_generation_and_origin_smoke_pass',
                                              'conditions': 35, 'seed': args.seed,
                                              'source_sha256': sha(__file__),
                                              'eat_config_sha256': sha(args.eat_config)})
    except Exception as exc:
        durable(args.output/'failure.json', {'type': type(exc).__name__, 'message': str(exc),
                                            'traceback': traceback.format_exc(), 'completed_rows': len(rows)})
        raise
    finally:
        ledger = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
        durable(args.output/'sha-ledger.json', ledger)
        durable(args.output/'host-seal.json', {'ledger_sha256': sha(args.output/'sha-ledger.json'),
                                              'file_count': len(ledger)})


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
