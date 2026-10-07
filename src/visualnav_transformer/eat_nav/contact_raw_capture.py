"""Lossless raw PhysX contact capture before any pair-buffer validation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import time
import numpy as np

_NAMES=('forces','points','normals','separations','counts','starts')

def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def capture_contact_api(root: Path, step: int, pose_id: str, arrays,
                        sensor_paths, filter_paths, reported_capacity: int) -> dict:
    """Save all six arrays first; return mapping and bounds evidence without slicing data."""
    values=[np.asarray(a) for a in arrays]
    if len(values)!=6:
        raise ValueError('PhysX returned other than six contact arrays')
    folder=Path(root)/'raw'/f'step-{step:04d}'
    folder.mkdir(parents=True,exist_ok=True)
    raw=folder/f'{pose_id}.npz';temporary=folder/f'{pose_id}.npz.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**dict(zip(_NAMES,values)))
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(raw)
    digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    counts,starts=values[4],values[5]
    dimensions_ok=(counts.ndim==2 and starts.shape==counts.shape and
                   counts.shape==(len(sensor_paths),len(filter_paths)))
    observed_capacity=int(values[0].shape[0]) if values[0].ndim else 0
    summary={'schema':'eatnav_raw_contact_v1','pose_id':pose_id,'physics_step':step,
        'logical_time_s':step*.005,'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
        'raw_path':str(raw),'raw_sha256':digest,
        'array_shapes':[list(v.shape) for v in values],
        'array_dtypes':[str(v.dtype) for v in values],
        'reported_capacity':int(reported_capacity),'observed_capacity':observed_capacity,
        'sensor_count':len(sensor_paths),'filter_count':len(filter_paths),
        'sensor_paths':list(sensor_paths),'filter_paths':list(filter_paths),
        'counts':counts.tolist(),'starts':starts.tolist(),
        'index_shape_match':bool(dimensions_ok),'backend_overflow_flag':'not_exposed',
        'first_invalid':None}
    if dimensions_ok:
        c=counts.astype(np.int64,copy=False);s=starts.astype(np.int64,copy=False)
        sums=c+s
        summary.update({'min_count':int(c.min(initial=0)),'max_count':int(c.max(initial=0)),
            'min_start':int(s.min(initial=0)),'max_start':int(s.max(initial=0)),
            'max_start_plus_count':int(sums.max(initial=0)),
            'sum_counts':int(c.sum()),'nonzero_count_rows':int(np.count_nonzero(c)),
            'buffer_full_by_count':bool(c.sum()>=observed_capacity)})
        for i,j in np.ndindex(c.shape):
            count,start=int(c[i,j]),int(s[i,j])
            if count<0 or start<0 or start+count>observed_capacity:
                sensor=sensor_paths[i];match=re.search(r'/env_(\d+)/',sensor)
                summary['first_invalid']={'sensor_index':i,'filter_index':j,
                    'sensor_path':sensor,'filter_path':filter_paths[j],
                    'environment_index':int(match.group(1)) if match else None,
                    'rigid_body':sensor.rsplit('/',1)[-1],
                    'count':count,'start':start,'start_plus_count':start+count}
                break
    else:
        summary['first_invalid']={'reason':'count_start_shape_or_path_mapping_mismatch'}
    _atomic_json(folder/f'{pose_id}.json',summary)
    return summary


def diagnostic_should_stop(physics_step: int, *, invalid: bool) -> bool:
    """Bound the candidate-free raw-index diagnostic to its frozen 60-step budget."""
    return invalid or physics_step >= 60


def choose_contact_capacity(observed_required_entries: int) -> int:
    """Smallest power of two with a 2x margin over observed required entries."""
    if observed_required_entries <= 0:
        raise ValueError('observed contact requirement must be positive')
    return 1 << (2 * observed_required_entries - 1).bit_length()


def validate_view_capacity(mode: str, configured: int, observed_required: int | None) -> int:
    """Lock diagnostic capacity and the one prospective full-run capacity."""
    if mode == 'diagnostic' and configured == 256 and observed_required is None:
        return configured
    if mode == 'full' and observed_required is not None and configured == choose_contact_capacity(observed_required):
        return configured
    raise ValueError('contact capacity violates frozen diagnostic/prospective rule')


def capacity_decision_from_raw(summary: dict) -> dict:
    """Classify B only for one-view flat-buffer exhaustion with mapped indices."""
    shapes=summary.get('array_shapes',[])
    capacity=summary.get('observed_capacity')
    paths=summary.get('sensor_paths',[])
    if (not summary.get('index_shape_match') or not summary.get('first_invalid') or
        len(shapes)!=6 or shapes[4]!=[len(paths),1] or shapes[5]!=[len(paths),1] or
        len(summary.get('filter_paths',[]))!=1 or capacity!=summary.get('reported_capacity') or
        summary.get('sum_counts',0)<=capacity or summary.get('max_start_plus_count',0)<=capacity or
        not paths or len({p.split('/onix_chassis/')[0] for p in paths})!=1):
        raise ValueError('raw evidence does not isolate per-view capacity exhaustion')
    required=int(summary['sum_counts'])
    return {'classification':'B_contact_capacity_insufficient',
            'observed_required_entries':required,
            'previous_capacity':capacity,
            'prospective_capacity':choose_contact_capacity(required),
            'first_invalid':summary['first_invalid']}


def verify_frozen_view_contract(sensor_paths, expected_sensor_paths,
                                filter_paths, expected_filter_paths,
                                array_shapes, expected_array_shapes) -> bool:
    """Fail closed on identity or installed API layout drift."""
    if (list(sensor_paths)!=list(expected_sensor_paths) or
        list(filter_paths)!=list(expected_filter_paths) or
        list(array_shapes)!=list(expected_array_shapes)):
        raise ValueError('frozen contact view identity or array shape drift')
    return True
