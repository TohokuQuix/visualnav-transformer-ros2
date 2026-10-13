"""Read-only sealed P1 evidence adapter and versioned derived pilot dataset."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ..p1_g3_rosfree_crossover_evidence import validate_output
from ..p1_teacher_chunks import verify_chunks
from .labels import LABELS, SCHEMA as LABEL_SCHEMA, labels_from_trace
from .preprocessing import body_local_path, resample_path


SCHEMA = 'eatnav_baseline_dataset_v1'
SOURCE_CROSSOVER = 'eatnav_p1_g3_rosfree_crossover_v1'
SOURCE_TEACHER = 'eatnav_p1_teacher_capture_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_host_seal(batch):
    batch=Path(batch)
    seal=json.loads((batch/'host-seal.json').read_text())
    ledger_path=batch/'sha-ledger.json'
    ledger=json.loads(ledger_path.read_text())
    if sha(ledger_path)!=seal['sha_ledger_sha256']:
        raise ValueError('host seal ledger SHA')
    expected=ledger['file_sha256']
    actual={str(p.relative_to(batch)):sha(p) for p in batch.rglob('*')
            if p.is_file() and p.name not in ('sha-ledger.json','host-seal.json')}
    if actual!=expected:
        raise ValueError('host seal files changed')
    return sha(batch/'host-seal.json'),seal['stage']


def validate_group_split(samples, split):
    ids=[s['sample_id'] for s in samples]
    if len(set(ids))!=len(ids) or set(split)!={'train','validation','test'}:
        raise ValueError('split IDs/schema')
    assigned=[sid for part in split.values() for sid in part]
    if len(assigned)!=len(ids) or set(assigned)!=set(ids):
        raise ValueError('split coverage/duplicate')
    group_for={s['sample_id']:s['group_id'] for s in samples}
    groups=[{group_for[sid] for sid in split[part]} for part in ('train','validation','test')]
    if any(groups[i]&groups[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError('paired group split leakage')


def _root_yaw(root):
    w,x,y,z=root[3:7]
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def load_sealed_period(root, period, source):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    assignment_path=root/'g3-assignment.json'
    assignment=json.loads(assignment_path.read_text())
    if manifest.get('schema')!=SOURCE_CROSSOVER or sha(assignment_path)!=manifest['assignment_sha256']:
        raise ValueError('source/assignment schema or SHA')
    if source==SOURCE_TEACHER and manifest.get('teacher_capture_schema')!=SOURCE_TEACHER:
        raise ValueError('teacher capture schema')
    if source==SOURCE_CROSSOVER and manifest.get('teacher_capture_schema') is not None:
        raise ValueError('mixed source protocol')
    batch=root/f'batch-first-differential-0-period-{period}'
    if manifest['authorized_process_ids']!=[batch.name]:
        raise ValueError('authorized process mismatch')
    seal_sha,seal_stage=verify_host_seal(batch)
    required_stage=('G3_teacher_capture_protocol_valid' if source==SOURCE_TEACHER
                    else 'G3_rosfree_crossover_protocol_valid')
    if seal_stage!=required_stage:
        raise ValueError('source not protocol valid')
    output=batch/'output'
    if source==SOURCE_TEACHER:
        teacher=json.loads((output/'teacher-log-complete.json').read_text())
        verified=verify_chunks(output/'teacher-chunks')
        if teacher.get('rows')!=verified['rows'] or verified['rows']<160 or \
                teacher.get('reward_manifest_sha256')!=sha(output/'teacher-reward-manifest.json'):
            raise ValueError('teacher chunk/reward provenance')
    assigned=assignment['processes'][period]['trials']
    validate_output(output,manifest,assigned)
    strict=json.loads((output/'cross-period-prepath-gate.json').read_text())
    if not strict['pass'] or (period and not strict['cross_period_equality_assessed']):
        raise ValueError('cross period pre-Path gate')
    snapshot=json.loads((output/'prepath-crossover-snapshot.json').read_text())
    terrain=json.loads((output/'terrain.json').read_text())
    dem_metadata=json.loads((output/'prepath-shared-dem-metadata.json').read_text())
    publication=json.loads((output/'path-activation.json').read_text())['trial_paths']
    inputs=np.load(output/'prepath-branch-inputs.npz',allow_pickle=False)
    samples=[]
    for env in range(4):
        trace_path=output/f'trace-slot-{env}.jsonl'
        trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
        source_files=[output/f'prepath-shared-dem-env-{env}-frame-{frame}.npy' for frame in range(4)]
        frame_meta=[item for item in dem_metadata if item.get('branch_env')==env]
        if len(frame_meta)!=4 or any(item.get('frame')!='robot_base' or
                item.get('history_role')!='shared_experimental_history' or
                item.get('resolution_m')!=.1 or item.get('source_env')!=0 for item in frame_meta):
            raise ValueError('shared DEM frame/resolution/history contract')
        frame_meta=sorted(frame_meta,key=lambda item:item['timestamp_ns'])
        if any(frame_meta[frame]['array_file']!=source_files[frame].name or
                frame_meta[frame]['array_sha256']!=sha(source_files[frame]) for frame in range(4)) or \
                [item['timestamp_ns'] for item in frame_meta]!=sorted(set(item['timestamp_ns'] for item in frame_meta)):
            raise ValueError('shared DEM frame order/SHA')
        dem=np.stack([np.load(path,allow_pickle=False) for path in source_files])
        if dem.shape!=(4,2,20,20) or not np.isfinite(dem[:,0][dem[:,1]==1]).all():
            raise ValueError('DEM shape/validity')
        state=inputs['actor_pre_noise'][env]
        if state.shape!=(22,) or not np.isfinite(state).all() or \
                not np.allclose(state,snapshot['envs'][env]['actor_pre_noise'],atol=0,rtol=0):
            raise ValueError('pre-Path Actor mismatch')
        root_state=snapshot['envs'][env]['root_state_w']
        origin=terrain['world_tile_origins'][env]
        robot_tile_xy=np.asarray(root_state[:2])-np.asarray(origin[:2])
        path=resample_path(body_local_path(publication[env]['map_xy'],robot_tile_xy,
                                              _root_yaw(root_state)))
        label=labels_from_trace(trace,common_goal_xy=publication[env]['common_goal_map_xy'],
                                env_origin_xy=origin[:2])
        if source==SOURCE_TEACHER:
            label['reward_term_availability']='logged_resolved_terms'
        sample_id=f'first-differential-2026101101-period-{period}-env-{env}'
        samples.append({'sample_id':sample_id,'group_id':'first-differential:2026101101:'+sha(output/'prepath-crossover-snapshot.json'),
                        'state_group':'first-differential','seed':2026101101,'period':period,
                        'env_id':env,'path_role':publication[env]['path_role'],
                        'dem':dem,'state':state,'path':path,'label':label,
                        'provenance':{'source_protocol':source,'source_root':str(root),
                                      'manifest_sha256':sha(root/'manifest.json'),
                                      'host_seal_sha256':seal_sha,'trace_sha256':sha(trace_path),
                                      'prepath_snapshot_sha256':sha(output/'prepath-crossover-snapshot.json'),
                                      'path_sha256':publication[env]['path_sha256'],
                                      'dem_frame_sha256':[sha(path) for path in source_files],
                                      'actor_input_source_sha256':sha(output/'prepath-branch-inputs.npz'),
                                      'simulator_source_sha256':manifest['g3_simulator_source_sha256']}})
    return samples


def build_crossover(roots):
    if len(roots)!=3 or len(set(map(str,roots)))!=3:
        raise ValueError('three period roots required')
    samples=[sample for period,root in enumerate(roots)
             for sample in load_sealed_period(root,period,SOURCE_CROSSOVER)]
    if len(samples)!=12 or len({s['group_id'] for s in samples})!=1 or \
            any({s['path_role'] for s in samples if s['env_id']==env}!=
                {'direct','left_20','right_20'} for env in range(4)):
        raise ValueError('balanced crossover/source independence')
    return samples


def build_teacher_capture(root):
    samples=load_sealed_period(root,0,SOURCE_TEACHER)
    if any(s['path_role']!='direct' for s in samples):
        raise ValueError('teacher diagnostic is Direct only')
    return samples


def write_derived(samples, root, *, source):
    root=Path(root)
    if root.exists() or not samples:
        raise FileExistsError(root)
    if len({s['provenance']['source_protocol'] for s in samples})!=1:
        raise ValueError('source protocol mixing')
    split={'train':[s['sample_id'] for s in samples],'validation':[],'test':[]}
    validate_group_split(samples,split)
    root.mkdir(parents=True)
    arrays={'dem':np.stack([s['dem'] for s in samples]).astype(np.float32),
            'state':np.stack([s['state'] for s in samples]).astype(np.float32),
            'path':np.stack([s['path'] for s in samples]).astype(np.float32),
            'label':np.asarray([[np.nan if value is None else value for value in s['label']['values']]
                                for s in samples],dtype=np.float32),
            'label_mask':np.asarray([s['label']['availability'] for s in samples],dtype=bool)}
    np.savez_compressed(root/'samples.npz',**arrays)
    metadata={'schema':SCHEMA,'source_protocol':source,'label_schema':LABEL_SCHEMA,
              'adapter_source_sha256':sha(Path(__file__)),
              'label_source_sha256':sha(Path(__file__).with_name('labels.py')),
              'label_names':LABELS,'sample_count':len(samples),
              'independent_group_count':len({s['group_id'] for s in samples}),
              'split':split,'samples':[{k:v for k,v in sample.items()
                                         if k not in ('dem','state','path')} for sample in samples],
              'source_sha256':{str(root_path):sha(root_path/'manifest.json')
                               for root_path in {Path(s['provenance']['source_root']) for s in samples}},
              'arrays_sha256':sha(root/'samples.npz')}
    with (root/'metadata.json').open('x') as stream:
        json.dump(metadata,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')
    return metadata
