"""Resume-safe exact-stamp Stage B preflight runner. Freeze original state and candidate IDs."""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, subprocess, time
from pathlib import Path
import numpy as np
import eat_nav_sim_clock_run as base

ROOT=Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon')
OLD=Path('/tmp/eatnav-candidate-coverage/sim-clock-long-horizon')
base.ROOT=ROOT;base.CONTAINER=str(ROOT);base.SIM=str(ROOT/'exact_sim.py')

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def prepare():
    base.docker('exec','isaac-lab-onix','mkdir','-p',str(ROOT))
    base.docker('cp','scripts/eat_nav_exact_stamp_sim.py',f'isaac-lab-onix:{ROOT}/exact_sim.py')
    base.docker('cp','src/visualnav_transformer/eat_nav/sim_clock_protocol.py',f'isaac-lab-onix:{ROOT}/sim_clock_protocol.py')
    for i in (1,3,4,6):
        dst=f'{ROOT}/long-horizon-inputs/state-{i:03d}'
        base.docker('exec','isaac-lab-onix','mkdir','-p',dst)
        for name in ('config-120.yaml','suffix-actions-120.npz'):
            base.docker('cp',str(base.OLD/'long-horizon-inputs'/f'state-{i:03d}'/name),f'isaac-lab-onix:{dst}/{name}')

def frozen():
    original=base.read(OLD/'clock-corrected-run-identities.json')
    if len(original['runs'])!=20:raise ValueError('not twenty frozen runs')
    if [x['order'] for x in original['runs']]!=list(range(20)):raise ValueError('frozen order mismatch')
    for entry in original['runs']:
        path=base.BASE/'pilot-inputs'/f"state-{int(entry['state_id'][-3:]):03d}"/f"{entry['candidate_id']}-map-path.json"
        if base.sha(path)!=entry['map_path_sha256']:raise ValueError('Path SHA mismatch '+str(path))
    if base.sha(OLD/'clock-corrected-run-identities.json')!='b678cc6b0ee0f326c86a189e838deeb4e305b3d3fa358ebe53e8c1c347c69eb3':
        raise ValueError('frozen run identity SHA mismatch')
    return original

def inspect_run(folder,steps,t0):
    trace=base.trace(folder/'live-0.trace.json.gz')
    with gzip.open(folder/'live-0.clock.json.gz','rt') as f: clock=json.load(f)
    for k,row in enumerate(trace):
        ns=t0+k*50_000_000
        if row['logical_sim_time_ns']!=ns or row['tf_stamp_ns']!=ns or row['follower_command_stamp_ns']!=ns:
            raise ValueError(f'logical/TF/command mismatch step {k}')
        if row['bridge_rejected_count']!=[0]*8:raise ValueError(f'command rejected step {k}')
        events=row['bridge_command_events']
        if len(events)!=8 or any(not event['accepted'] or event['stamp_ns']!=ns or event['tf_stamp_ns']!=ns for event in events):
            raise ValueError(f'not exactly 8 accepted commands at step {k}')
        for name in ('actor_input','actor_latent','actor_action','critic_value','reward','root_state','contact_force'):
            v=np.asarray(row[name],dtype=float)
            if not np.all(np.isfinite(v)):raise ValueError(f'nonfinite {name} step {k}')
    return {'steps':len(trace),'first_command':trace[0]['follower_command'],
            'last_command':trace[-1]['follower_command'], 'clock_trace_type':type(clock).__name__}

def parity_40(identity, folder):
    old=base.BASE/'pilot/runs'/identity['source_run_id']
    if base.sha(old/'live-0.trace.json.gz')!=identity['source_trace_sha256']:
        raise ValueError('source pilot trace SHA mismatch')
    left=base.trace(old/'live-0.trace.json.gz')
    right=base.trace(folder/'live-0.trace.json.gz')
    signals=('command','pre_noise','post_noise','actor_input','critic_observation',
             'actor_latent','actor_action','critic_value','processed_action',
             'joint_target','joint_position','joint_velocity','root_state',
             'contact_force','raycast_hits','dem','reward','terminal','post_root_state')
    if len(left)!=40 or len(right)!=40:raise ValueError('40-step parity trace length')
    for step,(a,b) in enumerate(zip(left,right)):
        for signal in signals:
            if a[signal]!=b[signal]:
                delta=None
                try:delta=float(np.max(np.abs(np.asarray(a[signal],dtype=float)-np.asarray(b[signal],dtype=float))))
                except (ValueError,TypeError):pass
                return {'pass':False,'first_difference':{'step':step,'signal':signal,'previous':a[signal],
                     'exact':b[signal],'max_abs':delta},'old_trace_sha256':base.sha(old/'live-0.trace.json.gz')}
    return {'pass':True,'signals':list(signals),'steps':40,'old_trace_sha256':base.sha(old/'live-0.trace.json.gz')}

def run_t4c(ids, started):
    identity=ids['runs'][0];t0=11_000_000_000
    atomic(ROOT/'status.json',{'stage':'T4C_RUNNING','run_id':'coverage001-nominal-40','started_unix':time.time()})
    result=base.run_one('t4c','coverage001-nominal-40',identity,40,t0)
    folder=ROOT/'t4c/runs/coverage001-nominal-40'
    inspect_run(folder,40,t0)
    parity=parity_40(identity,folder)
    atomic(ROOT/'t4c-parity.json',parity)
    if not parity['pass']:
        first=parity['first_difference']
        raise RuntimeError(f"40-step parity first difference step={first['step']} signal={first['signal']} max_abs={first['max_abs']}")
    atomic(ROOT/'t4c.done.json',{'status':'pass','run':result,'parity_sha256':base.sha(ROOT/'t4c-parity.json')})
    atomic(ROOT/'status.json',{'stage':'T4C_PASS','completed':1,'elapsed_sec':time.monotonic()-started})
    print('EXACT_T4C_PASS',flush=True)

def run_t4d(ids, started):
    from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120
    entries=[x for x in ids['runs'] if x['state_id']=='coverage-003' and x['kind']=='nominal']
    if len(entries)!=3 or [x['repeat'] for x in entries]!=[0,1,2]:
        raise ValueError('coverage003 frozen duplicate identities mismatch')
    folders=[]
    for j,identity in enumerate(entries):
        t0=(51+10*j)*1_000_000_000
        run_id=f'coverage003-nominal-repeat-{j}'
        atomic(ROOT/'status.json',{'stage':'T4D_RUNNING','run_id':run_id,'completed':j,
            'started_unix':time.time(),'elapsed_sec':time.monotonic()-started})
        base.run_one('t4d',run_id,identity,120,t0)
        folder=ROOT/'t4d/runs'/run_id
        inspect_run(folder,120,t0)
        folders.append(folder)
        atomic(ROOT/f't4d-repeat-{j}.done.json',{'status':'pass','run_id':run_id,
            'run_marker_sha256':base.sha(folder/'run-complete.json')})
    summaries=[base.read(f/'live-0.json') for f in folders]
    traces=[base.trace(f/'live-0.trace.json.gz') for f in folders]
    duplicate=duplicate_120(summaries,traces,.0105)
    atomic(ROOT/'t4d-duplicate.json',duplicate)
    if not duplicate['pass'] or not duplicate['dem_sha_equal']:
        raise RuntimeError(f"coverage003 duplicate failed rms={duplicate.get('xy_rms_m')}")
    atomic(ROOT/'t4d.done.json',{'status':'pass','duplicate_sha256':base.sha(ROOT/'t4d-duplicate.json'),
            'elapsed_sec':time.monotonic()-started})
    atomic(ROOT/'status.json',{'stage':'T4D_PASS','completed':3,'elapsed_sec':time.monotonic()-started})
    print('EXACT_T4D_PASS',duplicate,flush=True)

def audit_t4e(ids):
    old_path=base.BASE/'command-preview-long-horizon/long-horizon-run-identities.json'
    old=base.read(old_path)
    if base.sha(old_path)!=ids['old_run_identity_sha256']:
        raise ValueError('old twenty-run identity SHA mismatch')
    for j,(before,after) in enumerate(zip(old['runs'],ids['runs'])):
        same={k:v for k,v in after.items() if k!='logical_t0_ns'}
        if before!=same or after['order']!=j or after['logical_t0_ns']!=(81+10*j)*1_000_000_000:
            raise ValueError(f'run identity changed at order {j}')
    if len(old['runs'])!=20:raise ValueError('old Stage B run count mismatch')
    states=['coverage-001','coverage-003','coverage-004','coverage-006']
    for block,state in enumerate(states):
        group=ids['runs'][5*block:5*(block+1)]
        if [r['state_id'] for r in group]!=[state]*5:
            raise ValueError('state order changed')
        if [r['kind'] for r in group]!=['nominal']*3+['nomad_oracle','bank_winner']:
            raise ValueError('candidate order changed')
    report={'status':'pass','run_count':20,'states':states,'steps_per_run':120,
        'duration_sec':6.0,'old_identity_sha256':base.sha(old_path),
        'new_identity_sha256':base.sha(OLD/'clock-corrected-run-identities.json'),
        'duplicate_xy_rms_limit_m':.0105,'wall_limit_sec':5400,'storage_limit_bytes':250000000,
        'source_sha256':old.get('source_sha256'), 'stage_b_config_sha256':old.get('stage_b_config_sha256')}
    atomic(ROOT/'t4e-input-audit.json',report)
    atomic(ROOT/'t4e.done.json',{'status':'pass','audit_sha256':base.sha(ROOT/'t4e-input-audit.json')})
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['t4b','t4c','t4d','t4e'],required=True);args=parser.parse_args()
    started=time.monotonic()
    try:
        ids=frozen();prepare()
        if args.stage=='t4c':
            run_t4c(ids,started)
            return
        if args.stage=='t4d':
            run_t4d(ids,started)
            return
        if args.stage=='t4e':
            if not (ROOT/'t4d.done.json').exists():raise RuntimeError('T4D gate pending')
            audit_t4e(ids)
            atomic(ROOT/'status.json',{'stage':'T4E_PASS','completed':20,'elapsed_sec':time.monotonic()-started})
            print('EXACT_T4E_PASS',flush=True)
            return
        identity=ids['runs'][0]
        # Dedicated logical epoch so this pilot cannot collide with old artifacts.
        t0=1_000_000_000
        atomic(ROOT/'status.json',{'stage':'T4B_RUNNING','run_id':'coverage001-nominal-20','started_unix':time.time()})
        result=base.run_one('t4b','coverage001-nominal-20',identity,20,t0)
        folder=ROOT/'t4b/runs/coverage001-nominal-20'
        checked=inspect_run(folder,20,t0)
        atomic(ROOT/'t4b.done.json',{'status':'pass','run':result,'inspection':checked,
             'elapsed_sec':time.monotonic()-started,'output_sha256':base.sha(folder/'run-complete.json')})
        atomic(ROOT/'status.json',{'stage':'T4B_PASS','completed':1,'elapsed_sec':time.monotonic()-started})
        print('EXACT_T4B_PASS',checked,flush=True)
    except BaseException as exc:
        atomic(ROOT/'status.json',{'stage':'STOPPED','at':args.stage,'reason':repr(exc),'elapsed_sec':time.monotonic()-started})
        print('EXACT_T4B_STOP',repr(exc),flush=True)
        raise
if __name__=='__main__':main()
