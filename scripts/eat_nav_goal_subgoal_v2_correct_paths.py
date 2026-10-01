"""Correct only the stored interpretation of NoMaD's already sampled waypoints."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2')
OLD_SHA = 'e7c0dcc8b4a7dccb18c7b44a7896305443f596b883bf058c93abf4345cfe4cd3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / 'paths-v2.jsonl.gz'
    marker = ROOT / 'v3-paths.done.json'
    archived = ROOT / 'paths-v2-sampler-output-pre-correction.jsonl.gz'
    archived_marker = ROOT / 'v3-paths.pre-correction.done.json'
    if any(path.exists() for path in (archived, archived_marker, ROOT / 'v3-path-representation-correction.json')):
        raise FileExistsError('representation correction already attempted')
    if sha(source) != OLD_SHA or json.loads(marker.read_text())['paths_sha256'] != OLD_SHA:
        raise ValueError('frozen pre-correction inference SHA mismatch')
    shutil.copy2(source, archived)
    shutil.copy2(marker, archived_marker)
    if sha(archived) != OLD_SHA:
        raise ValueError('pre-correction archive SHA mismatch')
    with gzip.open(archived, 'rt') as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        # The recorded "path_increments" were exactly model.sample() output.
        # model.sample() already calls cumsum, so they are the original waypoints.
        waypoints = np.asarray(row['path_increments'], dtype=np.float32)
        if waypoints.shape != (8, 2) or hashlib.sha256(waypoints.tobytes()).hexdigest() != row['path_sha256']:
            raise ValueError('recorded model sample changed')
        increments = np.diff(np.vstack((np.zeros((1, 2), dtype=np.float32), waypoints)), axis=0)
        goal = np.asarray(row['goal_xy'], dtype=np.float32)
        row['path_increments'] = increments.astype(float).tolist()
        row['cumulative_waypoints'] = waypoints.astype(float).tolist()
        row['endpoint_error_m'] = float(np.linalg.norm(waypoints[-1] - goal))
        row['sample_output_sha256'] = row['path_sha256']
    if len(rows) != 32:
        raise ValueError('frozen model sample count drift')
    temp = source.with_name(source.name + '.corrected.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())
    temp.replace(source)
    old_marker = json.loads(archived_marker.read_text())
    updated = {**old_marker, 'paths_sha256': sha(source), 'representation': 'sample returns cumulative waypoints',
               'pre_correction_paths_sha256': OLD_SHA, 'resampling_performed': False}
    temp = marker.with_name(marker.name + '.corrected.tmp')
    temp.write_text(json.dumps(updated, indent=2, sort_keys=True) + '\n')
    temp.replace(marker)
    correction = {'status': 'corrected', 'old_paths_sha256': OLD_SHA, 'new_paths_sha256': sha(source),
                  'archived_marker_sha256': sha(archived_marker), 'path_count': 32,
                  'model_samples_reused_exactly': True, 'resampling_performed': False,
                  'source_semantics': 'model.sample returns cumsum(delta) in robot-local frame'}
    temp = ROOT / 'v3-path-representation-correction.json.tmp'
    temp.write_text(json.dumps(correction, indent=2, sort_keys=True) + '\n')
    temp.replace(ROOT / 'v3-path-representation-correction.json')
    print('V3_PATH_REPRESENTATION_CORRECTED', correction['new_paths_sha256'])


if __name__ == '__main__':
    main()
