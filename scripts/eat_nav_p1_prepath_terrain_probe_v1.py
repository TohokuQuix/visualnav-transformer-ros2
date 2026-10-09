"""Candidate-free P1 source/batch terrain geometry comparison; no physics steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import numpy as np
import torch
import yaml
from isaaclab.terrains.terrain_generator import TerrainGenerator
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.eat_nav_env_cfg import EATTerrainGenerator, make_collection_cfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER, terrain_seed


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


manifest = json.loads(args.manifest.read_text())
if digest(__file__) != manifest['probe_source_sha256']:
    raise ValueError('probe source SHA drift')
if digest(manifest['plan_path']) != manifest['plan_sha256']:
    raise ValueError('plan SHA drift')
if digest(manifest['eat_config_path']) != manifest['eat_config_sha256']:
    raise ValueError('EAT config SHA drift')
if digest(manifest['checkpoint_env_path']) != manifest['checkpoint_env_sha256']:
    raise ValueError('checkpoint env SHA drift')
plan = json.loads(Path(manifest['plan_path']).read_text())
target_by_index = {r['target_env_index']: r for r in plan['targets']}


class ProbeGenerator(EATTerrainGenerator):
    mode = 'source'
    batch_seed = 0

    def __init__(self, cfg, device):
        self.canonical = []
        super().__init__(cfg, device)

    def _generate_random_terrains(self):
        if (self.cfg.num_rows, self.cfg.num_cols) != (16, 7):
            raise ValueError('terrain layout drift')
        for row in range(16):
            for col in range(7):
                index = row*7+col
                name = target_by_index[index]['terrain_type'] if self.mode == 'batch' and index in target_by_index else TERRAIN_ORDER[col]
                cfg = self.cfg.sub_terrains[name]
                mesh, origin = self._get_terrain_mesh(.5, cfg)
                self._add_sub_terrain(mesh, origin, row, col, cfg)

    def _get_terrain_mesh(self, difficulty, cfg):
        index = self._tile_index
        col = index % 7
        name = next(name for name, item in self.cfg.sub_terrains.items() if item is cfg)
        if self.mode == 'batch' and index in target_by_index:
            seed = target_by_index[index]['terrain_seed']
        else:
            seed = terrain_seed(self.batch_seed, col)
        old_seed, old_np = self.cfg.seed, np.random.get_state()
        try:
            self.cfg.seed = int(seed)
            np.random.seed(int(seed))
            mesh, origin = TerrainGenerator._get_terrain_mesh(self, difficulty, cfg)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            self.canonical.append({'index': index, 'terrain_type': name, 'seed': int(seed),
                                   'difficulty': float(difficulty),
                                   'origin_local': np.asarray(origin).tolist(),
                                   'vertex_count': len(vertices), 'triangle_count': len(faces),
                                   'tile_local_vertices_faces_sha256': hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest()})
            self.tile_metadata.append({'tile_index': index, 'row': index//7, 'column': col,
                                       'seed': int(seed), 'terrain_type': name, 'difficulty': float(difficulty)})
            self._tile_index += 1
            return mesh, origin
        finally:
            self.cfg.seed = old_seed
            np.random.set_state(old_np)


def build(mode, batch_seed):
    saved = yaml.load(Path(manifest['checkpoint_env_path']).read_text(), Loader=yaml.BaseLoader)
    toggles = {name: saved[name].lower() == 'true' for name in
               ('use_last_action_obs', 'use_joint_effort_obs', 'use_height_scan_obs')}
    cfg = make_collection_cfg(OnixRoughEnvCfg(**toggles),
                              load_eat_config(Path(manifest['eat_config_path'])), batch_seed, 112)
    gen = cfg.scene.terrain.terrain_generator
    gen.class_type = ProbeGenerator
    gen.num_rows, gen.num_cols = 16, 7
    gen.curriculum = False
    gen.difficulty_range = (.5, .5)
    gen.use_cache = False
    gen.seed = batch_seed
    gen.sub_terrains = {name: gen.sub_terrains[name] for name in TERRAIN_ORDER}
    ProbeGenerator.mode, ProbeGenerator.batch_seed = mode, batch_seed
    generated = ProbeGenerator(gen, 'cpu')
    vertices = np.asarray(generated.terrain_mesh.vertices, dtype=np.float64)
    faces = np.asarray(generated.terrain_mesh.faces, dtype=np.int64)
    return {'mode': mode, 'batch_seed': batch_seed, 'tiles': generated.canonical,
            'world_mesh_vertex_count': len(vertices), 'world_mesh_triangle_count': len(faces),
            'world_mesh_f64_i64_sha256': hashlib.sha256(vertices.tobytes()+faces.tobytes()).hexdigest(),
            'world_mesh_f32_i32_sha256': hashlib.sha256(vertices.astype(np.float32).tobytes()+faces.astype(np.int32).tobytes()).hexdigest(),
            'tile_origins_w': generated.terrain_origins.tolist(),
            'collision_cfg': str(cfg.scene.terrain.collision_group),
            'physics_material_cfg': str(cfg.scene.terrain.physics_material)}


def main():
    args.output.mkdir(parents=True, exist_ok=False)
    atomic(args.output/'status.json', {'stage': 'started', 'pid': os.getpid(),
                                      'start_tick': int(Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                                      'Path_publications': 0, 'physics_steps': 0})
    sources = {str(seed): build('source', seed) for seed in manifest['batch_seeds']}
    batch = build('batch', manifest['batch_scene_seed'])
    rows = []
    for target in plan['targets']:
        i = target['target_env_index']
        source_seed = int(target['source_pose_id'].split('-', 1)[0][1:])
        source = sources[str(source_seed)]
        j = target['source_env_index']
        a, b = source['tiles'][j], batch['tiles'][i]
        rows.append({'physical_run_id': target['physical_run_id'], 'source_index': j,
                     'batch_index': i, 'source_tile': a, 'batch_tile': b,
                     'canonical_geometry_equal': a['tile_local_vertices_faces_sha256'] == b['tile_local_vertices_faces_sha256'],
                     'source_world_tile_origin': source['tile_origins_w'][j//7][j%7],
                     'batch_world_tile_origin': batch['tile_origins_w'][i//7][i%7],
                     'collision_config_equal': source['collision_cfg'] == batch['collision_cfg'],
                     'physics_material_config_equal': source['physics_material_cfg'] == batch['physics_material_cfg']})
    atomic(args.output/'terrain-geometry.json', {'sources': sources, 'batch': batch, 'target_comparison': rows,
                                                  'Path_publications': 0, 'physics_steps': 0})
    atomic(args.output/'complete.json', {'stage': 'complete', 'pid': os.getpid(), 'physics_steps': 0,
                                         'Path_publications': 0,
                                         'geometry_sha256': digest(args.output/'terrain-geometry.json')})


try:
    main()
finally:
    simulation_app.close()
