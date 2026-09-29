"""Detached, resume-safe seven-terrain pilot; never starts formal 56/20 states."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time
import yaml

ROOT=Path('/tmp/eatnav-vnt/docs/quality-identifiability-gate/feasible-set/seven-state-pilot')
CONTAINER_ROOT='/tmp/eatnav-quality-identifiability-gate/feasible-set/seven-state-pilot'
SCRIPT='/tmp/eatnav-vnt/scripts/eat_nav_seven_state_case.sh'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    tmp.replace(path)


def case_dir(index,name):return ROOT/'pilot-runs'/f'state-{index:03d}'/name


def verified_case(index,name,stage,repeat):
    folder=case_dir(index,name)
    marker=folder/f'{stage}-{repeat}.marker.json'
    if not marker.exists():return False
    data=json.loads(marker.read_text())
    for source_path,digest in data['outputs_sha256'].items():
        host=folder/Path(source_path).name
        if not host.exists() or sha(host)!=digest:raise ValueError(f'case marker output SHA mismatch: {host}')
    return True


def run_case(index,name,stage,repeat):
    if verified_case(index,name,stage,repeat):
        print('PILOT_RESUME_SKIP',index,name,stage,repeat,flush=True)
        return 0.0
    started=time.monotonic()
    process=subprocess.Popen([SCRIPT,str(index),name,stage,str(repeat)])
    while process.poll() is None:
        atomic(ROOT/'status.json',{'stage':'seven_state_pilot','state':'running','state_index':index,
               'maneuver':name,'run_stage':stage,'repeat_id':repeat,
               'run_elapsed_sec':time.monotonic()-started,'pilot_started':True,
               'formal_56_train_started':False,'validation_20_started':False})
        time.sleep(10)
    elapsed=time.monotonic()-started
    if process.returncode:raise RuntimeError(f'case failed state={index} name={name} stage={stage} repeat={repeat} code={process.returncode}')
    if not verified_case(index,name,stage,repeat):raise RuntimeError('case completed without valid SHA marker')
    if elapsed>200:raise RuntimeError(f'unexpected case runtime {elapsed:.1f}s')
    return elapsed


def load_case(index,name,stage,repeat):
    folder=case_dir(index,name)
    summary=json.loads((folder/f'{stage}-{repeat}.json').read_text())
    with gzip.open(folder/f'{stage}-{repeat}.trace.json.gz','rt') as stream:trace=json.load(stream)
    return summary,trace


def xy_rms(traces):
    if len(traces)<2 or any(len(t)!=40 for t in traces):raise ValueError('incomplete 40-step duplicate')
    def xy(row):
        terminal=row.get('terminal')
        return (terminal['root_state'] if terminal and 'root_state' in terminal else row['post_root_state'])[:2]
    a=traces[0];values=[]
    for b in traces[1:]:
        for k in range(40):
            for j in range(2):
                delta=(xy(b[k])[j]-b[0]['root_state'][j])-(xy(a[k])[j]-a[0]['root_state'][j])
                values.append(delta*delta)
    return (sum(values)/len(values))**.5


def duplicate_check(index):
    cases=[load_case(index,'nominal','live',i) for i in range(3)]
    summaries=[x[0] for x in cases]
    if len({s['fixed_map_path_sha256'] for s in summaries})!=1:
        raise ValueError('duplicate Path SHA mismatch')
    critical=('root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
              'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
              'critic_observation','critic_value','action_manager_action','latency_substeps',
              'latency_seconds','latency_policy_obs_prev','action_delay_buffers','dem_history_sha256')
    if any(any(s['anchor'][key]!=summaries[0]['anchor'][key] for key in critical) for s in summaries[1:]):
        raise ValueError('duplicate anchor differs')
    if any(any(row['command']!=cases[0][1][k]['command'] for k,row in enumerate(trace))
           for _,trace in cases[1:]):
        raise ValueError('duplicate command series differs')
    rms=xy_rms([trace for _,trace in cases])
    if rms>.0105:raise ValueError(f'duplicate XY RMS {rms:.6f} > .0105m')
    return {'state_index':index,'duplicate_xy_rms_m':rms,'nominal_path_sha256':summaries[0]['fixed_map_path_sha256'],
            'run_summary_sha256':[sha(case_dir(index,'nominal')/f'live-{i}.json') for i in range(3)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--gate',type=Path,required=True);a=p.parse_args()
    gate=json.loads(a.gate.read_text())
    if gate.get('classification')!='valid':raise ValueError('State 42 gate is not valid; pilot forbidden')
    config=yaml.safe_load((ROOT/'pilot-config.yaml').read_text())
    selection=json.loads((ROOT/'pilot-state-selection.json').read_text())
    indices=[x['state_index'] for x in selection['selected']]
    if indices!=config['source_state_indices'] or len(indices)!=7:raise ValueError('frozen selected indices mismatch')
    if sha(ROOT/'pilot-state-selection.json')!=config['selection_sha256']:raise ValueError('selection SHA mismatch')
    if sha(ROOT/'pilot-input-freeze.json')!=config['input_freeze_sha256']:raise ValueError('input freeze SHA mismatch')
    outputs=[];invalid=0;started=time.monotonic()
    for index in indices:
        state_marker=ROOT/'pilot-runs'/f'state-{index:03d}'/'state-complete.json'
        invalid_marker=ROOT/'pilot-runs'/f'state-{index:03d}'/'state-invalid.json'
        if state_marker.exists():
            outputs.append(json.loads(state_marker.read_text()));continue
        if invalid_marker.exists():
            outputs.append(json.loads(invalid_marker.read_text()));invalid+=1
            if invalid>=2:break
            continue
        try:
            run_case(index,'nominal','anchor',0)
            folder=f'{CONTAINER_ROOT}/pilot-inputs/state-{index:03d}'
            subprocess.run(['docker','exec','isaac-lab-onix','python3','/tmp/eat_nav_duplicate_recovery_smoke_paths.py',
                            '--state-folder',folder,'--nominal-map',folder+'/nominal-map-path.json'],check=True)
            subprocess.run(['docker','cp',f'isaac-lab-onix:{folder}/.',
                            str(ROOT/'pilot-inputs'/f'state-{index:03d}')+'/'],check=True)
            for repeat in range(3):run_case(index,'nominal','live',repeat)
            duplicate=duplicate_check(index)
            controls_path=ROOT/'duplicate-controls.json'
            controls=json.loads(controls_path.read_text()) if controls_path.exists() else {'states':{}}
            controls['states'][str(index)]=duplicate;atomic(controls_path,controls)
            for name in config['candidate_order_by_state'][index]:
                repeat=3 if name=='nominal' else 0
                run_case(index,name,'live',repeat)
            completed={'state_index':index,'classification':'measurement_valid_preanalysis',
                       'duplicate_xy_rms_m':duplicate['duplicate_xy_rms_m'],
                       'candidate_order':config['candidate_order_by_state'][index]}
            atomic(state_marker,completed);outputs.append(completed)
        except BaseException as exc:
            invalid+=1
            failed={'state_index':index,'classification':'measurement_invalid','error':repr(exc)}
            atomic(invalid_marker,failed);outputs.append(failed)
            print('PILOT_STATE_INVALID',index,repr(exc),flush=True)
            if invalid>=2:break
    result={'state_count_completed':len(outputs),'measurement_invalid_count':invalid,
            'states':outputs,'elapsed_sec':time.monotonic()-started,
            'formal_56_train_started':False,'validation_20_started':False,
            'stopped_early':invalid>=2}
    atomic(ROOT/'pilot-run-status.json',result)
    atomic(ROOT/'status.json',{'stage':'seven_state_pilot','state':'stopped_early' if invalid>=2 else 'raw_runs_complete',
                              'pilot_started':True,'measurement_invalid_count':invalid,
                              'formal_56_train_started':False,'validation_20_started':False})
    print('PILOT_RAW_RUNS_COMPLETE',len(outputs),'invalid',invalid,'elapsed',round(result['elapsed_sec'],1),flush=True)

if __name__=='__main__':main()
