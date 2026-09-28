"""Frozen relative-label/window identity checks for follow-up diagnostics."""
from __future__ import annotations
import math
import numpy as np

FIELDS=('quality_abs','v0','delta_v','advantage_h','drawdown','recovery','return_h')


def validate_aligned_record(row,feature,position:int,split:str):
    if row['split']!=split or int(row['window_index'])!=int(feature['index'][position]):
        raise ValueError('split/window index mismatch')
    if tuple(row['episode_key'])!=tuple(int(x) for x in feature['episode'][position]):
        raise ValueError('episode identity mismatch')
    if not row['primary_available'] or row['reward_steps']!=40 or row['end_timestamp_ns']-row['start_timestamp_ns']!=2_000_000_000:
        raise ValueError('not a 40-reward aligned primary window')
    for name in FIELDS:
        a=float(row[name]);b=float(feature[name][position])
        if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError(f'{name} label mismatch')
    for name in FIELDS[1:]:
        a=float(row['relative_z'][name]);b=float(feature['z_'+name][position])
        if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-5,abs_tol=1e-5):
            raise ValueError(f'z_{name} label mismatch')
