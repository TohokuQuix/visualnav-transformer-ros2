"""Re-label actual R4star candidate traces without changing sealed sources."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.r6_targets import future_targets, TIMES_SEC, TRAJECTORY_LAYOUT
from visualnav_transformer.eat_nav.fall_stall_targets import body_vector


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_group_roles(index):
    groups={}
    for row in index:
        if row['role'] not in ('train','validation','test'):raise ValueError('unknown role')
        key=(row.get('field',row.get('terrain')),int(row['seed']))
        if key in groups and groups[key]!=row['role']:raise ValueError('seed group leaks across roles')
        groups[key]=row['role']


def validate_input_times(header,anchor_ns):
    stamps=header.get('dem_stamps_ns')
    if stamps is None:return 'unavailable'
    if len(stamps)!=4 or any(not isinstance(t,int) or t>anchor_ns for t in stamps) or sorted(stamps)!=stamps:
        raise ValueError('input DEM timestamp is future or malformed')
    return 'checked'


def pack_group(initial,candidates):
    """Host interface: initial old NPZ arrays + actual branch {slot,path,rows}.

    initial keys dem(4,2,20,20), actor_state(22), body_state(9).
    candidates must contain only executed branches, with slot 0..13 and rows.
    For new collector inputs remove singleton batch axes before calling.
    """
    shapes=dict(dem=(4,2,20,20),actor_state=(22,),body_state=(9,),paths=(14,8,2),directions=(14,1),
        candidate_mask=(14,),event_targets=(14,4),event_mask=(14,4),trajectory=(14,20,9),trajectory_mask=(14,20))
    bools={'candidate_mask','event_mask','trajectory_mask'}
    result={k:np.zeros(v,bool if k in bools else np.float32) for k,v in shapes.items()}
    for k in ('dem','actor_state','body_state'):
        result[k]=np.asarray(initial[k],np.float32).copy()
        if result[k].shape!=shapes[k] or not np.isfinite(result[k]).all():raise ValueError('invalid initial '+k)
    seen=set();contacts={}
    for candidate in candidates:
        slot=candidate['slot']
        if slot not in range(14) or slot in seen:raise ValueError('invalid/duplicate slot')
        seen.add(slot);path=np.asarray(candidate['path'],np.float32)
        if path.shape!=(8,2):raise ValueError('path shape')
        rows=[dict(r) for r in candidate['rows']]
        if rows:rows[0]['candidate_path']=path.tolist()
        target=future_targets(rows)
        result['paths'][slot]=path;result['directions'][slot]=candidate['direction'];result['candidate_mask'][slot]=True
        for k in ('event_targets','event_mask','trajectory','trajectory_mask'):result[k][slot]=target[k]
        contacts[str(slot)]=target['contact_metadata']
    if not seen:raise ValueError('empty actual candidate group')
    return result,contacts


def build_dataset(source_root,output,*,registry_paths=()):
    source_root=Path(source_root).resolve();output=Path(output).resolve()
    if output.exists():raise ValueError('fresh output directory required')
    # Registries contain membership only; never open reserved-test labels/traces.
    excluded=set()
    for p in registry_paths:
        excluded.update(int(g['terrain_seed']) for g in read(p)['groups'] if g['role']=='reserved_test')
    groups=[];index=[];evidence=[];seen_groups=set()
    for idxpath in sorted((source_root/'datasets').glob('*/sample-index.json')):
        with np.load(idxpath.parent/'samples.npz',allow_pickle=False) as stored:
            arrays={k:stored[k] for k in ('dem','actor_state','body_state','paths','directions')}
        for i,member in enumerate(read(idxpath)):
            if member['role'] not in ('train','validation'):continue
            if member['seed'] in excluded:raise ValueError('reserved-test seed in development source')
            key=(member['field'],member['seed'],member['anchor'])
            if key in seen_groups:raise ValueError('duplicate StateGroup')
            seen_groups.add(key);sources=member['sources']
            # Original 4 branches were embedded before the 10 added branches.
            slots=[0,1,2,7,3,4,5,6,8,9,10,11,12,13]
            if len(sources)!=14:raise ValueError('expected fourteen actually recorded branches')
            candidates=[];provenance=[];timestamp_checks=[]
            for slot,source in zip(slots,sources):
                folder=Path(source)
                if not folder.is_absolute():folder=source_root.parents[2]/folder
                if 'reserved_test' in folder.parts:raise ValueError('reserved-test source prohibited')
                trace=folder/'trace.jsonl';rows=[json.loads(line) for line in trace.read_text().splitlines()]
                header=read(folder/'anchor.json')
                timestamp_checks.append(validate_input_times(header,rows[0]['logical_sim_time_ns']))
                candidates.append(dict(slot=slot,path=arrays['paths'][i,slot],direction=float(arrays['directions'][i,slot,0]),rows=rows))
                provenance.append(dict(slot=slot,trace=str(trace),sha256=sha(trace)))
            packed,contacts=pack_group({k:arrays[k][i] for k in ('dem','actor_state','body_state')},candidates)
            groups.append(packed)
            index.append(dict(member,sample_index=len(index),episode_group=f"{member['field']}/{member['seed']}",
                terrain_seed=member['seed'],level=4,contact_metadata=contacts,input_timestamp_checks=timestamp_checks))
            evidence.extend(provenance)
    if not groups:raise ValueError('no source groups')
    validate_group_roles(index)
    arrays={k:np.stack([g[k] for g in groups]) for k in groups[0]}
    output.mkdir(parents=True)
    np.savez_compressed(output/'samples.npz',**arrays)
    write(output/'sample-index.json',index)
    roles={role:[i for i,g in enumerate(index) if g['role']==role] for role in ('train','validation')}
    write(output/'split.json',dict(schema='eatnav_r6_seed_group_split_v1',roles=roles,reserved_test_included=False,
        group_key=['field','terrain_seed'],test='new independent collection required'))
    coverage={}
    for role,ids in roles.items():
        y=arrays['event_targets'][ids];m=arrays['event_mask'][ids]
        coverage[role]=dict(groups=len(ids),candidate_branches=int(arrays['candidate_mask'][ids].sum()),
            event_positive=((y==1)&m).sum((0,1)).tolist(),event_negative=((y==0)&m).sum((0,1)).tolist(),
            event_unknown=(~m&arrays['candidate_mask'][ids,:,None]).sum((0,1)).tolist(),trajectory_samples=int(arrays['trajectory_mask'][ids].sum()))
    write(output/'coverage.json',coverage);write(output/'source-sha256.json',evidence)
    manifest=dict(schema='eatnav_r6_observed_2s_pose_dataset_v1',event_horizon_sec=2,times_sec=TIMES_SEC.tolist(),
        trajectory_layout=list(TRAJECTORY_LAYOUT),trajectory_frame='initial yaw aligned, initial xyz removed, world up retained',
        samples=len(index),source_root=str(source_root),shapes={k:list(v.shape) for k,v in arrays.items()},
        source_reserved_test_opened=False,files_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    write(output/'manifest.json',manifest)
    return manifest,coverage


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-root',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--registry',action='append',type=Path,default=[]);args=parser.parse_args()
    print(json.dumps(build_dataset(args.source_root,args.output,registry_paths=args.registry),indent=2))

if __name__=='__main__':main()
