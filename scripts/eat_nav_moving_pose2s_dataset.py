#!/usr/bin/env python3
"""Pack causal executed-candidate 2s windows; never opens reserved test data."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from visualnav_transformer.eat_nav.bidirectional_bank import bank
from visualnav_transformer.eat_nav.fall_stall_targets import body_vector, root_yaw

SCHEMA = 'eatnav_moving_pose2s_dataset_v1'
INPUT_KEYS = ('dem','actor_state','body_state','paths','directions','candidate_mask')
IDS = [('forward' if c.direction == 1 else 'reverse') + ('_direct' if c.arc_angle_deg == 0 else '_%+d' % c.arc_angle_deg) for c in bank()]
BANK = np.stack([c.waypoints for c in bank()]).astype(np.float32)
DEFAULT_SOURCE = '/home/ubuntu/visualnav-transformer-ros2'


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def array_signature(arrays):
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def local_slot(canonical, count):
    if count == 14:
        return canonical
    if count == 4 and canonical in (0,1,2,7):
        return (0,1,2,7).index(canonical)
    raise ValueError('candidate absent in source bank')


def check_dem_stamps(stamps, t0_ns, period_ns=250000000):
    if len(stamps) != 4 or not np.array_equal(np.diff(stamps), [period_ns] * 3) or not 0 <= t0_ns - stamps[-1] < period_ns:
        raise ValueError('noncausal, stale, or incomplete dem stamps')


def check_root(snapshot, trace):
    if not np.array_equal(np.asarray(snapshot), np.asarray(trace)):
        raise ValueError('snapshot root differs from t0 trace pre-state')


def deduplicate(records):
    seen, result = set(), []
    for r in records:
        key = tuple(r[k] for k in ('source_trace_sha256','t0_ns','candidate_id','input_sha256'))
        if key not in seen:
            seen.add(key); result.append(r)
    return result, len(records) - len(result)


def assign_roles(records):
    """Union physical initial-state signatures; preserve all encoder-trained parents."""
    parents = {(r['field'], str(r.get('split_parent_id', r['seed']))) for r in records}
    leader = {p:p for p in parents}
    def find(p):
        while leader[p] != p:
            leader[p] = leader[leader[p]]; p = leader[p]
        return p
    original = defaultdict(set)
    signatures = {}
    for r in records:
        p = (r['field'], str(r.get('split_parent_id', r['seed'])))
        original[p].add(r['original_role'])
        sig = r['initial_state_signature']
        if sig in signatures:
            leader[find(p)] = find(signatures[sig])
        else:
            signatures[sig] = p
    if any(len(v) > 1 for v in original.values()):
        raise ValueError('parent appears in original train and evaluation')
    group_seeds = defaultdict(list)
    for r in records:
        group_seeds[find((r['field'],str(r.get('split_parent_id',r['seed']))))].append(str(r['seed']))
    groups = defaultdict(list)
    for p in parents:
        groups[find(p)].append(p)
    by_field = defaultdict(list); group_role = {}
    for g, ps in groups.items():
        if any('train' in original[p] for p in ps):
            group_role[g] = 'train'
        else:
            # Joining cross-field identical states is conservative: one deterministic group.
            by_field[min(p[0] for p in ps)].append(g)
    for field, gs in by_field.items():
        gs.sort(key=lambda g: hashlib.sha256(('moving-pose2s-v1' + min(group_seeds[g])).encode()).hexdigest())
        for ordinal, g in enumerate(gs):
            group_role[g] = 'validation' if ordinal % 2 == 0 else 'audit_eval'
    return [group_role[find((r['field'], str(r.get('split_parent_id', r['seed']))))] for r in records]


def stack_samples(samples):
    return {k:np.stack([s[k] for s in samples]) for k in samples[0]}


def discover_sources(source_root=DEFAULT_SOURCE):
    """Bind original development indices to raw only, using train/validation allowlist."""
    root = Path(source_root); records = []; bindings = {}
    r4 = root / 'docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/datasets'
    for index_path in sorted(r4.glob('*/sample-index.json')):
        bindings[str(index_path)] = sha256(index_path)
        for group in read_json(index_path):
            if group['role'] not in ('train','validation'):
                raise ValueError('reserved source is forbidden')
            for source in group['sources']:
                p = Path(source); tp = p / 'trace.jsonl'
                if not tp.exists() or not tp.stat().st_size:
                    continue
                manifest = read_json(p / 'input-manifest.json')
                records.append(dict(kind='r4', field=group['field'], level=manifest['level'], seed=group['seed'],
                    original_role=group['role'], sample_id=group['sample_id'], anchor=group['anchor'], source=str(p),
                    trace_path=str(tp), env=0))
    common = root / 'docs/eat-nav/additional-development-batch02-cumulative-20261021/cumulative'
    bindings[str(common/'sample-index.json')] = sha256(common/'sample-index.json')
    bindings[str(common/'targets.jsonl')] = sha256(common/'targets.jsonl')
    index = {r['sample_id'] + ':' + r['batch_id']:r for r in read_json(common/'sample-index.json')}
    for line in (common/'targets.jsonl').open():
        target = json.loads(line)
        if target['role'] not in ('train','validation'):
            raise ValueError('reserved source is forbidden')
        group = index[target['sample_id'] + ':' + target['batch_id']]
        name, envstr = target['sample_id'].split('/env'); env = int(envstr)
        p = Path(target['source_root']) / target['role'] / name / target['candidate'] / 'sim-output'
        manifest = read_json(p/'input-manifest.json')
        records.append(dict(kind='common', field=manifest['terrain_type'], level=manifest['terrain_level'],
            seed=target['seed'], original_role=target['role'], sample_id=target['sample_id'], anchor=0, source=str(p),
            trace_path=str(p / ('trace-env-%d.jsonl' % env)), env=env, candidate_id=target['candidate'],
            bound_trace_sha256=target['trace_sha256'],
            split_parent_id=target['batch_id']+'/'+manifest['terrain_type']+'/slot'+name.rsplit('-slot',1)[1]))
    return dict(records=records, bindings=bindings, source_root=str(root), reserved_test_opened=False)


def discover_fresh_sources(collection_root):
    """Consume production jobs only; preserve roles frozen before outcomes."""
    root=Path(collection_root);records=[];excluded=[];bindings={}
    plan=root/'plan.json'
    if not plan.exists():raise ValueError('fresh frozen plan is required')
    bindings[str(plan)]=sha256(plan)
    parents={p['group_id']:p for p in read_json(plan)['parents']}
    for gp in sorted((root/'production').glob('*/sim-output/group-*/group.json')):
        g=read_json(gp);bindings[str(gp)]=sha256(gp)
        if g['group_id'] not in parents:raise ValueError('fresh parent absent from frozen plan')
        parent=parents[g['group_id']]
        role=parent['role'].replace('-','_')
        if g['role'].replace('-','_')!=role or g['seed']!=parent['seed']:
            raise ValueError('fresh parent/role differs from frozen plan')
        if not g.get('anchor_available'):
            excluded.append(dict(source=str(gp),reason='anchor_unavailable'));continue
        for branch in sorted(gp.parent.glob('candidate-*')):
            slot=int(branch.name.split('-')[-1]);tp=branch/'trace.jsonl'
            if not tp.exists() or not tp.stat().st_size:
                excluded.append(dict(source=str(branch),reason='branch_unavailable'));continue
            if not (branch/'complete.json').exists():
                excluded.append(dict(source=str(branch),reason='branch_not_complete'));continue
            records.append(dict(kind='fresh',field=parent['field'],level=parent['level'],seed=parent['seed'],
                original_role='fresh',fixed_role=role,sample_id=g['group_id'],anchor=0,env=0,source=str(branch),
                trace_path=str(tp),input_path=str(branch/'input.npz'),group_path=str(gp),candidate_id=IDS[slot],
                split_parent_id='fresh/'+g['group_id'],geometry_sha256=(g.get('actual_tile_metadata') or {}).get('geometry_sha256'),
                collection_state=g['state'],counterfactual_complete=g.get('counterfactual_complete',False),
                restoration_gate_passed=g.get('restoration_gate_passed')))
    return dict(records=records,bindings=bindings,source_root=str(root),discovery_exclusions=excluded,reserved_test_opened=False)


def load_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k:f[k] for k in f.files}


def endpoint(row):
    return ((row.get('terminal') or {}).get('pre_reset_state') or {}).get('root_state', row.get('post_root_state'))


def current_input(desc, rows, start):
    p = Path(desc['source']); row = rows[start]; env = desc['env']; kind = desc['kind']
    root = row['root_state']
    if kind == 'common':
        raw_path = p/'raw-dem.npz'; input_path = p/'prepath.npz'; raw = load_npz(raw_path); pre = load_npz(input_path)
        check_root(pre['root'][env], root); check_root(raw['root_state_w'][-1,env], root)
        stamps = read_json(p/'raw-dem-metadata.json')['logical_stamp_ns']
        a = dict(dem=pre['dem'][env], actor_state=pre['actor_pre_noise'][env],
            body_state=np.concatenate([body_vector(root, v) for v in ([0,0,-1],root[7:10],root[10:13])]).astype(np.float32),
            paths=BANK.copy(), directions=np.array([c.direction for c in bank()],dtype=np.float32)[:,None],
            candidate_mask=np.ones(14,dtype=bool))
        candidate_id = desc['candidate_id']; slot = IDS.index(candidate_id)
        selection_path = p/'raw-dem-metadata.json'
    elif kind == 'fresh':
        input_path=Path(desc['input_path']);raw_path=input_path;selection_path=Path(desc['group_path'])
        payload=load_npz(input_path);a={k:payload[k][env] for k in INPUT_KEYS}
        check_root(payload['root_state_w'][env],root)
        if int(payload['t0_ns'])!=row['logical_sim_time_ns']:raise ValueError('fresh snapshot t0 mismatch')
        stamps=payload['dem_stamps_ns'].tolist();candidate_id=desc['candidate_id'];slot=IDS.index(candidate_id)
        if row.get('candidate_id')!=candidate_id:raise ValueError('fresh executed candidate mismatch')
        if not np.array_equal(a['paths'],BANK):raise ValueError('fresh bank mismatch')
    else:
        global_step = int(desc['anchor']) * 20 + start
        input_path = Path(desc.get('input_path',p/('input-%04d.npz'%global_step)))
        raw_path = Path(desc.get('raw_path',p/('raw-dem-%04d.npz'%global_step)))
        selection_path = Path(desc.get('selection_path',p/('selection-%04d.json'%global_step)))
        raw = load_npz(raw_path); a = {k:v[env] for k,v in load_npz(input_path).items()}
        j = read_json(selection_path); stamps = j['dem_stamps_ns']; check_root(raw['root_state_w'][env],root)
        if j['control_step'] != global_step or j['relative_step'] != start or j['input_file'] != input_path.name or j.get('prefix'):
            raise ValueError('snapshot global anchor/time binding mismatch')
        if not np.allclose(j['pose'][:2],root[:2],atol=2e-6,rtol=0) or abs(np.arctan2(np.sin(j['pose'][2]-root_yaw(root)),np.cos(j['pose'][2]-root_yaw(root)))) > 2e-6:
            raise ValueError('selection pose differs from current pre-state')
        candidate_id = row['candidate_id']; slot = local_slot(IDS.index(candidate_id),len(a['paths']))
        if j['candidate'] != slot:
            raise ValueError('executed candidate differs from selection')
        expected = BANK if len(a['paths']) == 14 else BANK[[0,1,2,7]]
        if not np.array_equal(a['paths'],expected):
            raise ValueError('noncanonical candidate paths')
    check_dem_stamps(stamps,row['logical_sim_time_ns'])
    for k in INPUT_KEYS:
        if not np.isfinite(a[k]).all():
            raise ValueError('nonfinite input: '+k)
    if a['dem'].shape != (4,2,20,20) or a['actor_state'].shape != (22,) or a['body_state'].shape != (9,):
        raise ValueError('input shape mismatch')
    expected_body = np.concatenate([body_vector(root,v) for v in ([0,0,-1],root[7:10],root[10:13])]).astype(np.float32)
    if not np.allclose(a['body_state'],expected_body,atol=2e-6,rtol=0):
        raise ValueError('body state differs from current root')
    if not np.allclose(a['actor_state'][3:6],a['body_state'][:3],atol=2e-6,rtol=0) or not np.allclose(a['actor_state'][:3],a['body_state'][6:9]*.25,atol=2e-6,rtol=0):
        raise ValueError('actor angular/gravity differs from current body')
    if not np.isin(a['dem'][:,1],[0,1]).all() or (a['dem'][:,1].mean(axis=(-2,-1)) < .25).any():
        raise ValueError('invalid dem coverage/mask')
    if start and (rows[start-1].get('terminal') or not np.array_equal(endpoint(rows[start-1]),root)):
        raise ValueError('previous post differs from current pre')
    direction = float(a['directions'][slot,0])
    if direction != row.get('follower_diagnostic',{}).get('travel_direction',direction):
        raise ValueError('candidate direction differs from follower')
    selected = {k:np.asarray(a[k]).astype(np.float32) for k in ('dem','actor_state','body_state')}
    selected.update(paths=np.asarray(a['paths'][slot:slot+1],np.float32), directions=np.asarray(a['directions'][slot:slot+1],np.float32),
        candidate_mask=np.asarray(a['candidate_mask'][slot:slot+1],bool), current_quat_wxyz=np.asarray(root[3:7],np.float32),
        initial_body_velocity=np.asarray(a['body_state'][3:6],np.float32), initial_body_angular_velocity=np.asarray(a['body_state'][6:9],np.float32),
        canonical_paths=BANK.copy(),canonical_directions=np.asarray([c.direction for c in bank()],np.float32)[:,None])
    provenance=dict(input_path=str(input_path),input_sha256=sha256(input_path),raw_path=str(raw_path),raw_sha256=sha256(raw_path),
        selection_path=str(selection_path),selection_sha256=sha256(selection_path),dem_stamps_ns=stamps)
    return selected, dict(root_state=root,path_body_xy=selected['paths'][0], direction=direction,t0_ns=row['logical_sim_time_ns'],candidate_id=candidate_id), provenance


def geometry_binding(desc):
    if desc.get('geometry_sha256'):return desc['geometry_sha256']
    p = Path(desc['source'])/'terrain.json'
    if not p.exists():
        identity = Path(desc['source'])/'prepath-identity.json'
        if identity.exists():
            tiles = read_json(identity).get('terrain_tiles', [])
            if desc['env'] < len(tiles):
                return tiles[desc['env']].get('geometry_sha256')
        return None
    a = read_json(p); tiles = a.get('selected_tile_metadata', [])
    try:
        return tiles[int(desc['level'])]['geometry_sha256']
    except (KeyError,IndexError,TypeError):
        return None


def verify_dataset(output):
    p=Path(output); manifest=read_json(p/'manifest.json')
    if manifest['schema'] != SCHEMA:
        raise ValueError('dataset schema mismatch')
    for name, expected in manifest['files'].items():
        if sha256(p/name) != expected:
            raise ValueError('dataset hash mismatch: '+name)
    # Sources are part of resume verification, never silently rebind changed raw.
    for source, expected in read_json(p/'source-binding.json')['files'].items():
        if sha256(source) != expected:
            raise ValueError('source hash mismatch: '+source)
    return manifest


def build_dataset(source_manifest, output):
    """Build once; existing output is verified byte-for-byte without overwriting."""
    from visualnav_transformer.eat_nav.moving_pose2s_targets import build_window
    out=Path(output)
    if out.exists():
        return verify_dataset(out)
    records=[]; samples=[]; reasons=Counter(); excluded=list(source_manifest.get('discovery_exclusions',[])); sources=dict(source_manifest.get('bindings',{})); pre_counts=Counter()
    for di, desc in enumerate(source_manifest['records']):
        tp=Path(desc['trace_path']); rows=[json.loads(l) for l in tp.open() if l.strip()]
        if not rows:
            continue
        trace_hash=sha256(tp)
        for name in ('input-manifest.json','terrain.json','prepath-identity.json'):
            binding=Path(desc['source'])/name
            if binding.exists():sources[str(binding)]=sha256(binding)
        if desc.get('bound_trace_sha256',trace_hash) != trace_hash:
            raise ValueError('bound trace SHA mismatch: '+str(tp))
        sources[str(tp)]=trace_hash
        starts = [0] if desc['kind'] in ('common','fresh') else range(0,len(rows),40)
        for start in starts:
            chunk=rows[start:start+40]
            if any(r.get('append_events') for r in chunk):
                reasons['append_excluded']+=1; excluded.append(dict(source=str(tp),start=start,reason='append_excluded')); continue
            a,snapshot,provenance=current_input(desc,rows,start)
            t=build_window(rows,start,snapshot)
            if not t['primary_eligible'] and desc['kind']=='fresh':
                reason=';'.join(t['reasons']['window']);reasons[reason]+=1;excluded.append(dict(source=str(tp),start=start,reason=reason));continue
            if not t['primary_eligible']:
                raise ValueError('unexpected primary-ineligible nonappend source: '+str((tp,start,t['reasons'])))
            candidate=snapshot['candidate_id']; root=snapshot['root_state']; is_initial=start==0 and desc['kind']!='fresh'
            key='fresh' if desc['kind']=='fresh' else 'common_initial' if desc['kind']=='common' else 'r4_initial' if is_initial else 'r4_moving'
            pre_counts[key]+=1; pre_counts['pose']+=int(t['pose_mask']);pre_counts['partial']+=int(not t['pose_mask'])
            a.update(position_targets=np.asarray(t['pos3'],np.float32)[None],rotation_targets=np.asarray(t['rot6'],np.float32)[None],
                event_targets=np.asarray(t['events4'],np.float32)[None],pose_mask=np.asarray([t['pose_mask']],bool),event_mask=np.asarray(t['event_mask4'],bool)[None],
                trajectory_position=np.asarray(t['trajectory_position'],np.float32)[None],trajectory_quaternion=np.asarray(t['trajectory_quaternion'],np.float32)[None],
                trajectory_mask=np.asarray(t['trajectory_mask'],bool)[None],trajectory_time_offset_sec=np.asarray(t['trajectory_time_offset_sec'],np.float32)[None])
            init_root=np.asarray(rows[0]['root_state'],np.float32)
            signature=array_signature(dict(root=init_root))
            q=root[3:7];pitch=float(np.arcsin(np.clip(2*(q[0]*q[2]-q[3]*q[1]),-1,1))*180/np.pi)
            rec={k:desc[k] for k in ('field','level','seed','original_role','sample_id','anchor','kind','env')}
            rec.update(source=str(tp.parent),source_trace=str(tp),source_trace_sha256=trace_hash,t0_ns=int(snapshot['t0_ns']),start=int(start),
                global_step=int(desc['anchor'])*20+start,candidate_id=candidate,candidate_slot=IDS.index(candidate),direction=snapshot['direction'],
                parent_id=desc['field']+'/'+str(desc.get('split_parent_id',desc['seed'])),split_parent_id=desc.get('split_parent_id',str(desc['seed'])),episode_id=desc['field']+'/'+str(desc['seed'])+'/env'+str(desc['env']),
                prefix_id=desc['sample_id'],branch_id=desc['sample_id']+'/'+candidate,initial_state_signature=signature,
                current_input_signature=array_signature({k:a[k] for k in INPUT_KEYS}),
                state_signature=array_signature(dict(dem=a['dem'],actor_state=a['actor_state'],body_state=a['body_state'],root=np.asarray(root,np.float32))),
                geometry_sha256=geometry_binding(desc),
                is_initial=is_initial,moving=not is_initial,speed_mps=float(np.linalg.norm(a['initial_body_velocity'])),pitch_deg=pitch,
                yawrate_radps=float(a['initial_body_angular_velocity'][2]),teacher_reasons=t['reasons'],teacher_metadata=t['metadata'],**provenance)
            if desc['kind']=='fresh':
                rec.update(fixed_role=desc['fixed_role'],collection_state=desc['collection_state'],counterfactual_complete=desc['counterfactual_complete'],restoration_gate_passed=desc['restoration_gate_passed'])
            rec['_array_index']=len(samples); records.append(rec);samples.append(a)
            for k in ('input_path','raw_path','selection_path'):
                sources[rec[k]]=rec[k.replace('_path','_sha256')]
        if di%100==0:
            print('processed sources',di,'windows',len(records),flush=True)
    unique, duplicates=deduplicate(records)
    roles=([r['fixed_role'] for r in unique] if unique and all('fixed_role' in r for r in unique) else assign_roles(unique))
    if not unique:raise ValueError('no eligible observed windows')
    arrays=stack_samples([samples[r.pop('_array_index')] for r in unique])
    for i,(r,role) in enumerate(zip(unique,roles)):
        r['index']=i;r['role']=role
    split=dict(schema='eatnav_moving_pose2s_split_v1',roles={role:[i for i,r in enumerate(unique) if r['role']==role] for role in ('train','validation','audit_eval')},reserved_test_opened=False)
    expected=source_manifest.get('expected_pre_dedup')
    if expected and any(pre_counts[k]!=v for k,v in expected.items()):
        raise ValueError('pre-dedup counts differ: '+str(dict(pre_counts)))
    out.mkdir(parents=True)
    np.savez_compressed(out/'samples.npz',**arrays)
    write_json(out/'sample-index.json',unique);write_json(out/'split.json',split)
    write_json(out/'teacher-reasons.json',[dict(index=r['index'],reasons=r['teacher_reasons']) for r in unique])
    write_json(out/'source-binding.json',dict(files=sources,source_root=source_manifest.get('source_root'),reserved_test_opened=False))
    write_json(out/'exclusions.json',excluded)
    manifest=dict(schema=SCHEMA,input_schema='eatnav_moving_pose2s_input_v1',horizon_sec=2.,pose_supervision='complete_2s_only',
        counts=dict(pre_dedup=dict(pre_counts),post_dedup=len(unique),duplicates_removed=duplicates,exclusions=dict(reasons)),
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},normalization_fitted=False,reserved_test_opened=False,
        files={name:sha256(out/name) for name in ('samples.npz','sample-index.json','split.json','teacher-reasons.json','source-binding.json','exclusions.json')})
    write_json(out/'manifest.json',manifest)
    return manifest


def merge_datasets(datasets, output):
    """Concatenate frozen components with exact dedup and frozen-role leakage checks."""
    out=Path(output)
    if out.exists():return verify_dataset(out)
    records=[];sources={};components={};arrays_by_key=defaultdict(list)
    for dataset in datasets:
        p=Path(dataset);m=verify_dataset(p);components[str(p/'manifest.json')]=sha256(p/'manifest.json')
        sources.update(read_json(p/'source-binding.json')['files'])
        a=load_npz(p/'samples.npz');index=read_json(p/'sample-index.json')
        if arrays_by_key and set(a)!=set(arrays_by_key):raise ValueError('component array schemas differ')
        offset=len(records)
        for r in index:
            r['_array_index']=offset+r['index'];records.append(r)
        for k,v in a.items():arrays_by_key[k].append(v)
    arrays={k:np.concatenate(v) for k,v in arrays_by_key.items()}
    unique,removed=deduplicate(records);ids=[r.pop('_array_index') for r in unique]
    arrays={k:v[ids] for k,v in arrays.items()}
    parent_roles=defaultdict(set);signature_roles=defaultdict(set);terrain_seed_roles=defaultdict(set)
    for i,r in enumerate(unique):
        r['index']=i;parent_roles[r['parent_id']].add(r['role']);signature_roles[r['initial_state_signature']].add(r['role']);terrain_seed_roles[(r['field'],str(r['seed']))].add(r['role'])
    if any(len(v)>1 for v in parent_roles.values()) or any(len(v)>1 for v in signature_roles.values()) or any(len(v)>1 for v in terrain_seed_roles.values()):
        raise ValueError('combined frozen roles conflict across parents/initial states')
    out.mkdir(parents=True)
    np.savez_compressed(out/'samples.npz',**arrays)
    write_json(out/'sample-index.json',unique)
    write_json(out/'split.json',dict(schema='eatnav_moving_pose2s_split_v1',roles={role:[r['index'] for r in unique if r['role']==role] for role in ('train','validation','audit_eval')},reserved_test_opened=False))
    write_json(out/'teacher-reasons.json',[dict(index=r['index'],reasons=r['teacher_reasons']) for r in unique])
    write_json(out/'exclusions.json',[])
    sources.update(components);write_json(out/'source-binding.json',dict(files=sources,components=components,reserved_test_opened=False))
    manifest=dict(schema=SCHEMA,input_schema='eatnav_moving_pose2s_input_v1',horizon_sec=2.,pose_supervision='complete_2s_only',
        counts=dict(pre_dedup=len(records),post_dedup=len(unique),duplicates_removed=removed,pose=int(arrays['pose_mask'].sum()),partial=int((~arrays['pose_mask']).sum())),
        arrays={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in arrays.items()},normalization_fitted=False,reserved_test_opened=False,
        components=components,files={name:sha256(out/name) for name in ('samples.npz','sample-index.json','split.json','teacher-reasons.json','source-binding.json','exclusions.json')})
    write_json(out/'manifest.json',manifest);return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--source-root',default=DEFAULT_SOURCE)
    parser.add_argument('--source-manifest');parser.add_argument('--fresh-collection');parser.add_argument('--merge',nargs='+');args=parser.parse_args()
    if args.merge:
        manifest=merge_datasets(args.merge,args.output);print(json.dumps(manifest['counts'],indent=2));return
    source=discover_fresh_sources(args.fresh_collection) if args.fresh_collection else read_json(args.source_manifest) if args.source_manifest else discover_sources(args.source_root)
    if not args.source_manifest and not args.fresh_collection:
        source['expected_pre_dedup']=dict(r4_initial=1290,common_initial=512,r4_moving=5095,pose=6880,partial=17)
    manifest=build_dataset(source,args.output);print(json.dumps(manifest['counts'],indent=2))

if __name__=='__main__':
    main()
