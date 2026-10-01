"""Frozen numeric equivalence check for four replayed DEM frames.

The collector and replay use a 20 Hz control clock.  A frame captured after
global step s has logical timestamp (s + 1) * 50 ms.  The simulator's
SourceGrid.stamp_ns is a placeholder (zero), so this checks the control-clock
timestamp and records that provenance explicitly.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from visualnav_transformer.eat_nav.config import load_eat_config


AXIS = 'row=+world_y,col=+world_x'
UNKNOWN = 'mask=0,height=0'
CONTROL_STEP_NS = 50_000_000


def _digest(array):
    return hashlib.sha256(np.asarray(array, dtype=np.float32).tobytes()).hexdigest()


def verify_dem_history(*, entry, summary, recorded_frames, selected_dem, reference_config,
                       state_config, source_sha256, expected_source_sha256):
    """Require exact frame metadata and numeric height/mask equivalence.

    recorded_frames is ordered oldest to newest and contains the original raw
    collector records.  selected_dem is the immutable dataset window, checked
    against those records before comparison to replay output.
    """
    if source_sha256 != expected_source_sha256 or summary['source_sha256']['runner_source'] != source_sha256:
        raise ValueError('replay source SHA differs from the parity-tested frame capture code')
    ref = vars(load_eat_config(reference_config))
    limits = yaml.safe_load(Path(state_config).read_text())['anchor_limits']
    if limits['dem_rms_m'] != .03 or limits['mask_iou_min'] != .95:
        raise ValueError('pre-existing DEM equivalence limits changed')
    history = np.asarray(summary['anchor']['dem_history'], dtype=np.float32)
    selected = np.asarray(selected_dem, dtype=np.float32)
    shape = (4, 2, ref['dem_height_cells'], ref['dem_width_cells'])
    if history.shape != shape or selected.shape != shape or len(recorded_frames) != 4:
        raise ValueError('four-frame DEM shape differs')
    if not np.isfinite(history).all() or not np.isfinite(selected).all():
        raise ValueError('nonfinite DEM history')
    if _digest(history) != summary['anchor']['dem_history_sha256']:
        raise ValueError('replay DEM raw SHA does not match saved anchor')
    if _digest(selected) != summary['state']['dem_sha256']:
        raise ValueError('selected DEM raw SHA does not match frozen state')
    if summary['state']['state_id'] != entry['state_id'] or summary['state']['episode_key'] != entry['episode_key']:
        raise ValueError('anchor identity differs')
    steps = [entry['anchor_step'] - 15, entry['anchor_step'] - 10,
             entry['anchor_step'] - 5, entry['anchor_step']]
    results = []
    for index, (step, record) in enumerate(zip(steps, recorded_frames)):
        identity = [record[k] for k in ('session_id', 'env_id', 'episode_id')]
        if identity != entry['episode_key'] or record['step'] != step or record['pre_reset']:
            raise ValueError(f'DEM frame {index} identity/episode differs')
        stamp = (step + 1) * CONTROL_STEP_NS
        if record['timestamp_ns'] != stamp:
            raise ValueError(f'DEM frame {index} logical timestamp differs')
        dem = record['dem']
        if dem['axis'] != AXIS or dem['unknown'] != UNKNOWN:
            raise ValueError(f'DEM frame {index} coordinate or unknown convention differs')
        if dem['resolution_m'] != ref['dem_resolution_m']:
            raise ValueError(f'DEM frame {index} resolution differs')
        recorded = np.asarray(dem['transformed'], dtype=np.float32)
        if recorded.shape != shape[1:] or not np.isfinite(recorded).all():
            raise ValueError(f'DEM frame {index} recorded shape/finite gate failed')
        if not np.array_equal(recorded, selected[index]):
            raise ValueError(f'DEM frame {index} does not match selected train window')
        a, b = history[index], recorded
        valid_a, valid_b = a[1] > .5, b[1] > .5
        common = valid_a & valid_b
        if not common.any():
            raise ValueError(f'DEM frame {index} has no common valid cells')
        rms = float(np.sqrt(np.mean((a[0][common].astype(np.float64) -
                                      b[0][common].astype(np.float64)) ** 2)))
        iou = float(common.sum() / (valid_a | valid_b).sum())
        if rms > limits['dem_rms_m'] or iou < limits['mask_iou_min']:
            raise ValueError(f'DEM frame {index} numeric gate failed: RMS={rms}, IoU={iou}')
        results.append({'index': index, 'identity': identity, 'global_step': step,
                        'timestamp_ns': stamp, 'timestamp_basis': '20Hz post-step control clock',
                        'shape': list(a.shape), 'source_frame': ref['reference_frame'],
                        'transformed_frame': ref['base_frame'], 'axis': AXIS,
                        'resolution_m': dem['resolution_m'], 'unknown': UNKNOWN,
                        'finite': True, 'rms_m': rms, 'mask_iou': iou,
                        'recorded_raw_sha256': _digest(b), 'replay_raw_sha256': _digest(a),
                        'raw_sha_match': _digest(a) == _digest(b)})
    return {'status': 'pass', 'frames': results, 'history_raw_sha_match': _digest(history) == _digest(selected),
            'limits': {'dem_rms_m': limits['dem_rms_m'], 'mask_iou_min': limits['mask_iou_min']},
            'timestamp_provenance': 'recorded timestamp vs replay capture-step logical timestamp; scanner SourceGrid stamp_ns=0'}
