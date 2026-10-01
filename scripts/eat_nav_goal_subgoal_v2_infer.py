"""Container-side, unchanged reference NoMaD sampler for frozen v2 inputs."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config

ROOT = Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2')
CHECKPOINT = Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')
REFERENCE_CONFIG = Path('/tmp/eatnav-train/docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
SEEDS = (1001, 1002)
CHECKPOINT_SHA = '191d6a03266bca2ce8d43afac44b04828ae0d2571632be4f47a5f0dc3564b4e0'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    goals_file = ROOT / 'goals-v2.json'
    arrays_file = ROOT / 'selected-states-v2.npz'
    output = ROOT / 'paths-v2.jsonl.gz'
    marker = ROOT / 'v3-paths.done.json'
    if output.exists() or marker.exists() or sha(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('output already exists or checkpoint drift')
    goals = json.loads(goals_file.read_text())['goals']
    if len(goals) != 16 or [r['goal_order'] for r in goals] != list(range(4)) * 4:
        raise ValueError('frozen goal order drift')
    config = load_eat_config(REFERENCE_CONFIG)
    model, stats, _ = load_checkpoint(CHECKPOINT, config)
    if not np.allclose(stats.mean, [.028116857632994652, -4.290757715352811e-05], atol=1e-8):
        raise ValueError('train-only action normalization drift')
    model = model.to('cuda').eval()
    torch.cuda.reset_peak_memory_stats()
    selected = np.load(arrays_file)
    rows = []
    with torch.inference_mode():
        for item in goals:
            dem = np.asarray(selected['dem'][item['state_index']], dtype=np.float32)
            goal = np.asarray(item['goal_xy'], dtype=np.float32)
            if hashlib.sha256(dem.tobytes() + goal.tobytes()).hexdigest() != item['input_sha256']:
                raise ValueError('NoMaD input SHA drift')
            d = torch.as_tensor(dem[None], device='cuda')
            g = torch.as_tensor(goal[None], device='cuda')
            for seed in SEEDS:
                waypoints = model.sample(d, g, None, seed=seed)[0].detach().cpu().numpy()
                if waypoints.shape != (8, 2) or not np.isfinite(waypoints).all():
                    raise ValueError('nonfinite or malformed Path')
                increments = np.diff(np.vstack((np.zeros((1, 2), dtype=np.float32), waypoints)), axis=0)
                rows.append({**item, 'diffusion_seed': seed, 'generation_order': len(rows),
                             'path_increments': increments.astype(float).tolist(),
                             'cumulative_waypoints': waypoints.astype(float).tolist(),
                             'endpoint_error_m': float(np.linalg.norm(waypoints[-1] - goal)),
                             'path_sha256': hashlib.sha256(waypoints.astype(np.float32).tobytes()).hexdigest(),
                             'model_checkpoint_sha256': CHECKPOINT_SHA})
    if len(rows) != 32:
        raise ValueError('Path count')
    temp = output.with_name(output.name + '.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())
    temp.replace(output)
    content = {'paths_sha256': sha(output), 'path_count': 32, 'goals_sha256': sha(goals_file),
               'checkpoint_sha256': CHECKPOINT_SHA, 'gpu_peak_bytes': torch.cuda.max_memory_allocated(),
               'gpu_reserved_peak_bytes': torch.cuda.max_memory_reserved()}
    temp = marker.with_name(marker.name + '.tmp')
    temp.write_text(json.dumps(content, indent=2, sort_keys=True) + '\n')
    temp.replace(marker)
    print('V3_PATHS_FROZEN', len(rows), content['paths_sha256'], flush=True)


if __name__ == '__main__':
    main()
