"""Read-only initial input/v2 teacher join and portable sealed dataset export."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .fall_stall_audit import (REPO, SPEC, read, require, seal_derived, sha,
                               verify_assignment, verify_attempt, verify_derived,
                               verify_path_activation, write)
from .fall_stall_dataset_splits import freeze_splits, validate_split
from .fall_stall_dataset_tensors import (CANDIDATE_IDS, INPUT_SCHEMA, pack_groups,
                                         target_value, validate_arrays)
from .fall_stall_evidence import validate_initial
from .fall_stall_targets import EVENT_NAMES, HORIZONS, SCHEMA as TARGET_SCHEMA, build_targets


SCHEMA='eatnav_fall_stall_dataset_v1'
DATA_FILES=('samples.npz','sample-index.json','split.json','targets.jsonl','summary.json','source-integrity.json')


def record_sha(record):
    return hashlib.sha256(json.dumps(record,sort_keys=True,allow_nan=False).encode()).hexdigest()


def bounded_file(root,relative):
    path=Path(relative)
    require(not path.is_absolute() and '..' not in path.parts,'source/implementation path escapes root')
    resolved=(root/path).resolve()
    require(resolved.is_relative_to(root) and resolved.is_file(),'source/implementation path escapes root or is missing')
    return resolved


def teacher_records(path):
    records={}
    for line_number,line in enumerate(path.read_text().splitlines(),1):
        record=json.loads(line)
        key=(record.get('state_group'),record.get('candidate'))
        require(key not in records,'duplicate teacher identity')
        require(isinstance(key[0],str) and key[1] in CANDIDATE_IDS,'invalid teacher identity')
        records[key]=(record,line_number)
    return records


def verify_teacher_source(source,teachers,pilot):
    verify_derived(teachers)
    manifest=read(teachers/'dataset-manifest.json')
    require(manifest.get('schema')==TARGET_SCHEMA and manifest.get('artifact_kind')=='teachers_and_differential_audit_only',
            'teacher artifact/schema mismatch')
    require(Path(manifest['source_root']).resolve()==source,'teacher source root mismatch')
    require(manifest['source_manifest_sha256']==sha(source/'pilot-manifest.json') and
            manifest['assignment_sha256']==sha(source/'assignment.json'),'teacher/source manifest SHA mismatch')
    require(manifest['frozen_source_sha256']==pilot['source_sha256'],'teacher/source frozen code identity mismatch')
    for relative,digest in manifest['source_file_sha256'].items():
        require(sha(bounded_file(source,relative))==digest,f'source SHA mismatch: {relative}')
    for relative,digest in manifest['implementation_sha256'].items():
        require(sha(bounded_file(REPO,relative))==digest,f'teacher implementation SHA mismatch: {relative}')
    # The prior prose SHA is a historical snapshot, not the frozen collection SHA.
    return manifest,teacher_records(teachers/'targets.jsonl')


def join_source(source,pilot,assignment,teachers):
    groups={};initial_cache={};baselines={};integrity=[];seen=set();windows=0
    definitions={g['state_group_id']:g for g in pilot['state_groups']}
    for number,a in enumerate(assignment):
        attempt=source/f'{number:03d}-{a["state_group_id"]}-{a["candidate_id"] or "prepath"}'
        host,info,complete,_=verify_attempt(attempt,a,pilot)
        sim=attempt/'sim-output'
        initial=validate_initial(sim,a['terrain_seed'],REPO,read,sha,require,initial_cache)
        prepath_sha=sha(sim/'prepath.npz')
        integrity.append({'attempt':attempt.name,'outer_seal_sha256':sha(attempt/'outer-seal.json'),
                          'inner_seal_sha256':sha(sim/'host-seal.json'),'prepath_sha256':prepath_sha,
                          'dem_reconstruction_max_abs_m':initial['dem_reconstruction_max_abs_m'],'pass':True})
        if a['stage']=='prepath':
            baselines[a['state_group_id']]=prepath_sha
            with np.load(sim/'prepath.npz',allow_pickle=False) as arrays:
                dem=arrays['dem'].copy();actor=arrays['actor_pre_noise'].copy();root=arrays['root'].copy()
            for env in range(4):
                sample_id=f'{a["state_group_id"]}/env{env}'
                groups[sample_id]={'sample_index':len(groups),'sample_id':sample_id,'base_group':a['state_group_id'],
                    'seed':a['terrain_seed'],'env':env,'terrain':a['terrain_type'],'level':a['terrain_level'],
                    'replicate':definitions[a['state_group_id']]['replicate'],
                    'initial':{'dem':dem[env],'actor_pre_noise':actor[env],'root':root[env]},'candidates':{},
                    'initial_provenance':{'prepath_npz':str((sim/'prepath.npz').relative_to(source)),
                        'prepath_sha256':prepath_sha,'identity_sha256':sha(sim/'prepath-identity.json'),
                        'raw_dem_sha256':sha(sim/'raw-dem.npz'),'decision_stamp_ns':1_000_000_000,
                        'dem_stamps_ns':read(sim/'prepath-identity.json')['dem_stamps_ns']}}
            continue
        require(host.get('reference_prepath_sha256')==info.get('reference_prepath_sha256')==
                baselines[a['state_group_id']]==prepath_sha,'candidate prepath/baseline identity mismatch')
        activation=read(sim/'path-activation.json')
        require(activation['direction']==a['direction'] and activation['arc_angle_deg']==a['angle'],
                'activation candidate identity mismatch')
        require(activation['activation_stamp_ns']==1_000_000_000,'decision/initial input timestamp mismatch')
        actual_terminated=[];actual_censored=[]
        for env in range(4):
            sample_id=f'{a["state_group_id"]}/env{env}';key=(sample_id,a['candidate_id'])
            require(key in teachers,'teacher coverage is missing a candidate')
            saved,line_number=teachers[key];seen.add(key)
            trace_path=sim/f'trace-env-{env}.jsonl'
            trace_sha=sha(trace_path)
            rows=[json.loads(line) for line in trace_path.read_text().splitlines()]
            require(trace_sha==complete['trace_sha256'][env] and len(rows)==complete['trace_lengths'][env],
                    'trace SHA/length mismatch')
            verify_path_activation(activation,info['path_body_xy_m'],rows,env,
                                   initial['root'][env],initial['root_tile_local'][env])
            expected={'attempt':attempt.name,'state_group':sample_id,'seed':a['terrain_seed'],'env':env,
                      'candidate':a['candidate_id'],'terrain':a['terrain_type'],'level':a['terrain_level'],
                      'trace':str(trace_path.relative_to(source)),'trace_sha256':trace_sha,
                      'v1_label_sha256':sha(sim/'long-horizon-event-labels.json')}
            require(all(saved.get(k)==v for k,v in expected.items()),'teacher identity/provenance mismatch')
            regenerated=build_targets(rows,info['path_body_xy_m'],decision_stamp_ns=activation['activation_stamp_ns'])
            require(saved.get('targets')==regenerated,'teacher target numerical regeneration mismatch')
            terminal=rows[-1].get('terminal') or {}
            if terminal.get('pre_reset_state'):
                actual_terminated.append(env)
            elif terminal.get('termination_reason')=='parallel_env_censored_by_other_env':
                actual_censored.append(env)
            else:
                require(not terminal and len(rows)==200,'partial trace without complete termination/censor evidence')
            slot=CANDIDATE_IDS.index(a['candidate_id'])
            groups[sample_id]['candidates'][a['candidate_id']]={'path':info['path_body_xy_m'],'direction':a['direction'],
                'targets':saved['targets'],'provenance':{**expected,'candidate_index':slot,
                    'decision_stamp_ns':activation['activation_stamp_ns'],'end_reason':saved['targets']['end_reason'],
                    'observed_until_sec':saved['targets']['observed_until_sec'],
                    'prepath_npz':str((sim/'prepath.npz').relative_to(source)),'prepath_sha256':prepath_sha,
                    'input_manifest_sha256':sha(sim/'input-manifest.json'),'activation_sha256':sha(sim/'path-activation.json'),
                    'teacher_line':line_number,'teacher_record_sha256':record_sha(saved)}}
            windows+=len(rows)
        require(sorted(complete['terminated_envs'])==actual_terminated and sorted(complete['censored_envs'])==actual_censored,
                'complete terminal/censor membership mismatch')
    require(seen==set(teachers) and len(seen)==pilot['candidate_trajectory_count'],'teacher coverage mismatch')
    require(len(groups)==len(pilot['state_groups'])*4,'StateGroup count mismatch')
    return list(groups.values()),integrity,windows


def sample_index(groups):
    result=[]
    for group in groups:
        row={key:group[key] for key in ('sample_index','sample_id','base_group','seed','env','terrain','level','replicate','initial_provenance')}
        row['candidates']=[group['candidates'][name]['provenance'] for name in CANDIDATE_IDS if name in group['candidates']]
        result.append(row)
    return result


def validate_target_join(arrays,index,records):
    expected=set()
    require(len(index)==len(arrays['dem']),'sample index/tensor count mismatch')
    for row in index:
        i=row['sample_index'];slots=set()
        require(row['sample_id']==f'{row["base_group"]}/env{row["env"]}','sample StateGroup/env identity mismatch')
        initial=row['initial_provenance'];stamp=initial['decision_stamp_ns']
        require(stamp==1_000_000_000 and initial['dem_stamps_ns']==[250_000_000,500_000_000,750_000_000,stamp],
                'initial input timestamps include future or wrong frames')
        for candidate in row['candidates']:
            slot=candidate['candidate_index'];name=candidate['candidate'];key=(row['sample_id'],name)
            require(slot in range(4) and slot not in slots and CANDIDATE_IDS[slot]==name,'candidate slot identity mismatch')
            slots.add(slot);expected.add(key)
            require(key in records,'teacher index coverage mismatch')
            saved,line=records[key];targets=saved['targets']
            require(candidate['teacher_line']==line and candidate['teacher_record_sha256']==record_sha(saved),'teacher line/record SHA mismatch')
            require(all(candidate.get(k)==saved.get(k) for k in ('candidate','seed','env','state_group','trace','trace_sha256'))
                    and saved['seed']==row['seed'] and saved['env']==row['env'] and saved['state_group']==row['sample_id'],
                    'sample/teacher identity mismatch')
            require(candidate['decision_stamp_ns']==stamp==targets['activation_stamp_ns'],'teacher/input decision time mismatch')
            for h,seconds in enumerate(HORIZONS):
                horizon=targets['horizons'][f'{seconds}s']
                for head,event in enumerate(EVENT_NAMES):
                    value=target_value(horizon['events'][event]);label=-1 if value is None else value
                    require(arrays['event_targets'][i,slot,h,head]==label and
                            bool(arrays['event_mask'][i,slot,h,head])==(value is not None),'tensor/teacher event mismatch')
                value=target_value(horizon['progress'],'value_m')
                require(arrays['progress_targets'][i,slot,h]==np.float32(0 if value is None else value) and
                        bool(arrays['progress_mask'][i,slot,h])==(value is not None),'tensor/teacher progress mismatch')
            free=target_value(targets['event_free_10s'])
            require(arrays['event_free_10s'][i,slot]==(-1 if free is None else free),'tensor/teacher event-free mismatch')
        require(arrays['candidate_mask'][i].tolist()==[slot in slots for slot in range(4)],'candidate/index mask mismatch')
    require(expected==set(records),'teacher coverage has missing/extra candidates')


def build_dataset(source_root,teacher_root,output_root):
    source,teachers,output=(Path(p).resolve() for p in (source_root,teacher_root,output_root))
    require(all(output!=root and root not in output.parents for root in (source,teachers)),
            'dataset output must be outside source and teacher evidence')
    if output.exists():
        raise FileExistsError(f'fresh dataset output required: {output}')
    pilot=read(source/'pilot-manifest.json');assignment=read(source/'assignment.json')['rows']
    verify_assignment(source,pilot,assignment)
    teacher_manifest,records=verify_teacher_source(source,teachers,pilot)
    groups,integrity,windows=join_source(source,pilot,assignment,records)
    arrays=pack_groups(groups);index=sample_index(groups)
    split=freeze_splits(index,arrays);validate_target_join(arrays,index,records)
    summary={'schema':SCHEMA,'input_schema':INPUT_SCHEMA,'target_schema':TARGET_SCHEMA,
             'sample_count':len(index),'trajectory_count':int(arrays['candidate_mask'].sum()),
             'base_seed_count':len(pilot['state_groups']),'processes_verified':len(integrity),
             'raw_observation_windows':windows,'sample_unit':'initial_StateGroup_with_paired_candidates',
             'array_shapes':{name:list(value.shape) for name,value in arrays.items()},
             'event_support_10s':{name:{'positive':int((arrays['event_targets'][:,:,3,head]==1).sum()),
                                         'negative':int((arrays['event_targets'][:,:,3,head]==0).sum()),
                                         'unknown':int((arrays['event_targets'][:,:,3,head]==-1).sum())}
                                 for head,name in enumerate(EVENT_NAMES)},
             'known_progress_by_horizon':{f'{h}s':int(arrays['progress_mask'][:,:,i].sum()) for i,h in enumerate(HORIZONS)},
             'normalization_applied':False,'normalization_fit_performed':False,'training_performed':False,
             'split_rotations':3,'formal_unseen_test':False,
             'max_dem_reconstruction_error_m':max(v['dem_reconstruction_max_abs_m'] for v in integrity)}
    output.mkdir(parents=True,exist_ok=False)
    np.savez(output/'samples.npz',**arrays)
    write(output/'sample-index.json',{'schema':SCHEMA,'samples':index})
    write(output/'split.json',split)
    (output/'targets.jsonl').write_bytes((teachers/'targets.jsonl').read_bytes())
    write(output/'source-integrity.json',integrity);write(output/'summary.json',summary)
    implementations={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),
        Path(__file__).with_name('fall_stall_dataset_tensors.py'),Path(__file__).with_name('fall_stall_dataset_splits.py'),
        REPO/'scripts/eat_nav_fall_stall_dataset.py']}
    implementations.update(teacher_manifest['implementation_sha256'])
    write(output/'dataset-manifest.json',{'schema':SCHEMA,'input_schema':INPUT_SCHEMA,'target_schema':TARGET_SCHEMA,
        'artifact_kind':'initial_paired_learning_dataset','candidate_ids':list(CANDIDATE_IDS),
        'event_head_order':list(EVENT_NAMES),'horizons_sec':list(HORIZONS),'primary_event_horizon_sec':10,
        'event_unknown_value':-1,'masked_progress_placeholder':0,'normalization':'none; fit only on train in future training',
        'input_contract':{'dem_frame':'robot_base_yaw','path_frame':'robot_base_yaw','dem_resolution_m':.1,
            'actor_source':'initial_prepath.actor_pre_noise','body_source':'initial_prepath.root',
            'body_layout':['projected_gravity_body_xyz','linear_velocity_body_xyz','angular_velocity_body_xyz'],
            'height_unit':'m','path_unit':'m','linear_velocity_unit':'m/s','angular_velocity_unit':'rad/s'},
        'source_root':str(source),'source_manifest_sha256':sha(source/'pilot-manifest.json'),
        'assignment_sha256':sha(source/'assignment.json'),'frozen_source_sha256':pilot['source_sha256'],
        'teacher_root':str(teachers),'teacher_manifest_sha256':sha(teachers/'dataset-manifest.json'),
        'teacher_ledger_sha256':sha(teachers/'sha-ledger.json'),'teacher_seal_sha256':sha(teachers/'derived-seal.json'),
        'teacher_targets_sha256':sha(teachers/'targets.jsonl'),'teacher_spec_snapshot_sha256':teacher_manifest['spec_sha256'],
        'spec_path':str(SPEC.relative_to(REPO)),'spec_sha256':sha(SPEC),'implementation_sha256':implementations,
        'python_version':sys.version,'numpy_version':np.__version__,'formal_unseen_test':False,
        'files_sha256':{name:sha(output/name) for name in DATA_FILES}})
    seal_derived(output)
    load_dataset(output)
    return summary


def load_dataset(root):
    """Validate portable sealed artifacts; source simulator files are not needed."""
    root=Path(root)
    verify_derived(root)
    manifest=read(root/'dataset-manifest.json')
    require(manifest.get('schema')==SCHEMA and manifest.get('input_schema')==INPUT_SCHEMA and
            manifest.get('target_schema')==TARGET_SCHEMA,'dataset input/target schema mismatch')
    require(manifest.get('candidate_ids')==list(CANDIDATE_IDS) and manifest.get('event_head_order')==list(EVENT_NAMES)
            and manifest.get('horizons_sec')==list(HORIZONS),'dataset candidate/head/horizon ordering mismatch')
    require(manifest.get('primary_event_horizon_sec')==10 and manifest.get('event_unknown_value')==-1
            and manifest.get('masked_progress_placeholder')==0 and manifest.get('formal_unseen_test') is False,
            'dataset target/split contract mismatch')
    require(set(manifest['files_sha256'])==set(DATA_FILES),'dataset artifact membership mismatch')
    for name,digest in manifest['files_sha256'].items():
        require(sha(root/name)==digest,f'dataset artifact SHA mismatch: {name}')
    with np.load(root/'samples.npz',allow_pickle=False) as saved:
        arrays={name:saved[name].copy() for name in saved.files}
    validate_arrays(arrays)
    index_file=read(root/'sample-index.json')
    require(index_file.get('schema')==SCHEMA,'sample index schema mismatch')
    index=index_file['samples'];split=read(root/'split.json')
    validate_split(index,arrays,split)
    validate_target_join(arrays,index,teacher_records(root/'targets.jsonl'))
    return arrays,index,split
