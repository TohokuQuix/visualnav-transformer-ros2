"""Frozen, sequential, resume-safe E1 continuation; no extra candidate processes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

import eat_nav_e1_run_one as replay
from eat_nav_e1_prepare import BANK,G0E,ROOT,sha,validate_frozen_identity,verify_anchor_reuse
from eat_nav_e1_monitor import gate_sample,limits,sample
from eat_nav_g0e_formal_gate import baseline_gate
from eat_nav_g0e_gpu_probe import Nvml,atomic,proc_table
from eat_nav_g0e_gpu_run import container_identity
from eat_nav_e1_lifecycle import protocol_valid_count
from eat_nav_e1_process_ownership import ProcessRegistry

REPO=Path('/tmp/eatnav-vnt-e1-feasibility')
SESSION='eatnav-e1-feasibility'
REMOTE=str(ROOT)
DOCKER='isaac-lab-onix'


def read(path):return json.loads(Path(path).read_text())


def docker(*args,check=True):
    return subprocess.run(['docker',*args],capture_output=True,text=True,check=check)


def execution_entry(row):
    index=int(row['state_group_id'][-3:])
    if index not in (0,1):raise ValueError('unapproved StateGroup')
    states=read(ROOT/'inputs/selected-states.json')['train']
    state=states[index]
    if state['state_id']!=row['state_group_id']:raise ValueError('state identity differs')
    folder=ROOT/'inputs'/f'state-{index:03d}'
    name=row['process_id']
    return {'order':row['order'],'run_id':name,'kind':
        'duplicate' if row['role']=='nominal_duplicate' else 'candidate',
        'state_id':row['state_group_id'],'state_index':index,
        'candidate_id':name,'path_id':row['path_id'],'seed':row['seed'],
        'anchor_step':state['step'],'episode_key':state['episode_key'],
        'logical_t0_ns':2_300_000_000_000+index*100_000_000_000,
        'local_path_sha256':row['path_sha256'],'map_path_sha256':row['map_path_sha256'],
        'base_path_sha256':sha(folder/f'{name}-base-path.json'),
        'map_path_file_sha256':sha(folder/f'{name}-map-path.json'),
        'goal_metadata_sha256':sha(folder/f'{name}-goal-metadata.json')}


def next_pending(rows,completed):
    order=[r['process_id'] for r in rows]
    if any(order[i] in completed and any(name not in completed for name in order[:i])
           for i in range(len(order))):
        raise ValueError('completed marker gap breaks frozen order')
    return next((r for r in rows if r['process_id'] not in completed),None)


def require_duplicate_before_lateral(row,passed):
    sid=row['state_group_id']
    if sid=='g0e-train-001' and passed.get('g0e-train-000')!='pass':
        raise ValueError('first StateGroup duplicate/group gate pending')
    if row['role']=='candidate_outcome' and passed.get(sid)!='pass':
        raise ValueError('StateGroup duplicate gate pending before lateral Path')


def source_paths():
    cpp=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/src/simple_trajectory_follower.cpp')
    header=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/include/simple_trajectory_follower/simple_trajectory_follower.hpp')
    return {'simulator':REPO/'scripts/eat_nav_e1_sim.py',
        'run_one':REPO/'scripts/eat_nav_e1_run_one.py',
        'schedule':REPO/'scripts/eat_nav_e1_schedule.py',
        'monitor':REPO/'scripts/eat_nav_e1_monitor.py',
        'process_ownership':REPO/'scripts/eat_nav_e1_process_ownership.py',
        'attempt3_launcher':REPO/'scripts/eat_nav_e1_attempt3.py',
        'anchor_gate':REPO/'scripts/eat_nav_e1_anchor_gate.py',
        'formal_gate':REPO/'scripts/eat_nav_g0e_formal_gate.py',
        'dem_gate':REPO/'scripts/eat_nav_goal_subgoal_dem_gate.py',
        'bridge':REPO/'scripts/eat_nav_command_delivery_bridge.py',
        'host_exact_stamp_protocol':REPO/'src/visualnav_transformer/eat_nav/exact_stamp_protocol.py',
        'follower_cpp':cpp,'follower_header':header,
        'follower_binary':replay.PROD,'follower_params':replay.PARAM}


def freeze_sources():
    original=read(G0E/'e1-process-manifest.json')
    continuation=read(ROOT/'continuation-manifest.json')
    validate_frozen_identity(original,continuation)
    verify_anchor_reuse()
    if read(G0E/'g0e-final-decision.json')['status']!='G0-E pass / E1 awaiting separate approval':
        raise ValueError('G0-E final decision differs')
    provenance=read(G0E/'formal-anchor-provenance.json')
    frozen={key:sha(path) for key,path in source_paths().items()}
    if frozen['follower_binary']!='616097e44d199b8d33f6534aedd299f2d20127ed8e100c18777f9ca7a2da389d':
        raise ValueError('exact-stamp production follower binary differs')
    if frozen['follower_params']!='ddedff40d13e20416a43309669ad19e7f3a06e2df8cc8c9d3248369e14d40c90':
        raise ValueError('production follower params differ')
    if sha(BANK/'g0e-selected-states.npz')!=sha(ROOT/'inputs/g0e-selected-states.npz'):
        raise ValueError('selected DEM NPZ differs')
    container_sha={path:docker('exec',DOCKER,'sha256sum',path).stdout.split()[0]
        for path in provenance['container_input_sha256']}
    if container_sha!=provenance['container_input_sha256']:
        raise ValueError('container manifest/checkpoint/reference SHA differs')
    rows=continuation['pending_rows']
    ids=[execution_entry(r) for r in rows]
    for original_row,derived in zip(rows,ids):
        if derived['local_path_sha256']!=original_row['path_sha256'] or \
                derived['map_path_sha256']!=original_row['map_path_sha256']:
            raise ValueError('Path identity differs')
    forecast=read(G0E/'e1-runtime-forecast.json')
    resources=read(G0E/'e1-resource-forecast.json')
    anchor_sec=sum(x['total_sec'] for x in forecast['formal_anchor_measurements'])
    estimated_continuation_sec=forecast['point_total_sec']-anchor_sec+18*10.5
    p95_margin_continuation_sec=forecast['p95_plus_20_percent_sec']-anchor_sec+18*10.5
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    status=subprocess.check_output(['git','status','--short'],cwd=REPO,text=True)
    audit={'branch':branch,'worktree':str(REPO),'HEAD':head,'dirty':bool(status),
        'git_status_short':status.splitlines(),
        'G0E_final_report_sha256':sha(G0E/'g0e-final-report.md'),
        'G0E_final_decision_sha256':sha(G0E/'g0e-final-decision.json'),
        'formal_anchor_result_sha256':[sha(G0E/f'formal-anchor-{i:03d}-result.json') for i in range(2)],
        'E1_original_manifest_sha256':sha(G0E/'e1-process-manifest.json'),
        'E1_continuation_manifest_sha256':sha(ROOT/'continuation-manifest.json'),
        'E1_gates_sha256':sha(ROOT/'e1-gates.yaml'),
        'E1_safety_stops_sha256':sha(ROOT/'e1-safety-stops.yaml'),
        'map_path_sha256':[{'state_group_id':r['state_group_id'],'path_id':r['path_id'],
            'sha256':r['map_path_sha256']} for r in rows if r['oracle_denominator']],
        'collection_manifest_sha256':provenance['container_input_sha256']['/tmp/eatnav-extended/output/manifest.json'],
        'split_hash':provenance['split_hash'],
        'RL_checkpoint_sha256':provenance['container_input_sha256']['/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'],
        'reference_config_sha256':provenance['container_input_sha256']['/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml'],
        'source_binary_sha256':frozen,'container_frozen_sha256':container_sha,
        'completed_anchor_only':2,'pending_fresh_processes':18,'run_order':[x['run_id'] for x in ids],
        'estimated_18_run_sec_including_10_5s_baseline_each':estimated_continuation_sec,
        'p95_plus_20_percent_18_run_sec_including_baselines':p95_margin_continuation_sec,
        'GPU_device_peak_forecast_mib':resources['formal_anchor_peak']['device_total_used_mib'],
        'CPU_anon_shmem_peak_forecast_bytes':resources['formal_anchor_peak']['anon_shmem_peak_bytes'],
        'storage_prediction_bytes':resources['projected_total_20_process_bytes'],
        'tmux_session':SESSION,'output_root':str(ROOT),
        'wall_clock_start':'continuation-started.marker.json creation, excluding G0-E anchor time',
        'E1_after_training_started':False,'learning_after_E1_authorized':False,
        'full_collection_authorized':False,'hardware_authorized':False}
    if (ROOT/'source-freeze.json').exists():
        prior=read(ROOT/'source-freeze.json')
        for key in ('source_binary_sha256','E1_continuation_manifest_sha256','E1_gates_sha256','E1_safety_stops_sha256'):
            if prior[key]!=audit[key]:raise ValueError('prelaunch E1 source freeze changed '+key)
    else:atomic(ROOT/'source-freeze.json',audit)
    return audit


def setup_container():
    docker('exec',DOCKER,'mkdir','-p',REMOTE)
    for name,path in source_paths().items():
        if name not in ('simulator','anchor_gate','formal_gate','dem_gate'):continue
        target=REMOTE+'/'+{'simulator':'eat_nav_e1_sim.py',
            'anchor_gate':'eat_nav_e1_anchor_gate.py','formal_gate':'eat_nav_g0e_formal_gate.py',
            'dem_gate':'eat_nav_goal_subgoal_dem_gate.py'}[name]
        docker('cp',str(path),f'{DOCKER}:{target}')
        if docker('exec',DOCKER,'sha256sum',target).stdout.split()[0]!=sha(path):
            raise ValueError('container E1 source SHA differs '+name)
    if docker('exec',DOCKER,'test','-e',REMOTE+'/inputs',check=False).returncode:
        docker('cp',str(ROOT/'inputs'),f'{DOCKER}:{REMOTE}/inputs')
    for local in (ROOT/'inputs').rglob('*'):
        if local.is_file():
            remote=REMOTE+'/'+str(local.relative_to(ROOT))
            if docker('exec',DOCKER,'sha256sum',remote).stdout.split()[0]!=sha(local):
                raise ValueError('container E1 input SHA differs '+str(local))


def prelaunch_baseline(row,container,nvml):
    run_id=row['process_id'];baseline_path=ROOT/'baselines'/f'{run_id}.jsonl'
    if baseline_path.exists():raise FileExistsError('run baseline already attempted')
    baseline_path.parent.mkdir(parents=True,exist_ok=True)
    prior=proc_table();baseline_ids={(pid,p['start_ticks']) for pid,p in prior.items()}
    residual=[]
    for registry_path in (ROOT/'runs').glob('*/process-registry.json') if (ROOT/'runs').exists() else []:
        registry=ProcessRegistry.load(registry_path)
        if registry.run_id!=registry_path.parent.name:raise ValueError('stale process registry run ID')
        if registry.run_id==run_id:continue
        residual.extend(registry.residuals())
    safety=yaml.safe_load((ROOT/'e1-safety-stops.yaml').read_text())
    samples=[];start=time.monotonic()
    while True:
        snap=sample(nvml,container,baseline_ids,str(ROOT/'runs'/run_id))
        with baseline_path.open('a') as stream:
            stream.write(json.dumps(snap,sort_keys=True,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        samples.append(gate_sample(snap,snap['cgroup_memory']['events'].get('oom',0)))
        if time.monotonic()-start>=10.5:break
        time.sleep(max(0,.5-(time.monotonic_ns()-snap['monotonic_ns'])/1e9))
    reason=baseline_gate([{**x,'residual_processes':residual} for x in samples],limits(safety))
    if max(x['device_total_used_mib'] for x in samples)>safety['infrastructure_stop']['gpu_device_total_memory_over_mib']-1000:
        reason=reason or 'insufficient_GPU_headroom'
    result={'status':'pass' if reason is None else 'not_started','reason':reason,
        'sample_count':len(samples),'duration_sec':time.monotonic()-start,
        'baseline_sha256':sha(baseline_path),'device_peak_mib':max(x['device_total_used_mib'] for x in samples),
        'residual_count':len(residual)}
    atomic(ROOT/'baselines'/f'{run_id}.gate.json',result)
    return result


def completed_rows(rows):
    completed=set()
    for row in rows:
        folder=ROOT/'runs'/row['process_id']
        if not folder.exists():continue
        marker=folder/'run-complete.json'
        if not marker.exists():raise ValueError('partial run; no automatic retry '+row['process_id'])
        validity=folder/'protocol-valid.json'
        if not validity.exists():raise ValueError('cleanup gate incomplete; no automatic retry '+row['process_id'])
        result=read(marker)
        if result['identity']!=execution_entry(row):raise ValueError('completed run identity differs')
        for name,digest in result['output_sha256'].items():
            if sha(folder/name)!=digest:raise ValueError('completed run output SHA differs')
        done=read(folder/'timing-monitor.done.json')
        verifier=read(folder/'finalization-verifier-result.json')
        registry=ProcessRegistry.load(folder/'process-registry.json')
        if registry.run_id!=row['process_id']:raise ValueError('completed process registry run ID')
        registry.require_complete()
        valid=read(validity)
        if (done['status']!='stopped' or done['runtime_monitor']!='pass'
                or verifier['status']!='pass' or verifier['run_complete_sha256']!=sha(marker)
                or verifier['process_registry_sha256']!=sha(folder/'process-registry.json')
                or valid['status']!='valid' or valid['identity']!=execution_entry(row)
                or valid['run_complete_sha256']!=sha(marker)
                or valid['monitor_done_sha256']!=sha(folder/'timing-monitor.done.json')
                or valid['finalization_verifier_sha256']!=sha(folder/'finalization-verifier-result.json')):
            raise ValueError('completed monitor marker differs')
        completed.add(row['process_id'])
    return completed


def process_counts(root=None):
    runs=Path(root or ROOT)/'runs'
    folders=[p for p in runs.iterdir() if p.is_dir()] if runs.exists() else []
    traces=sum((p/'live-0.trace.json.gz').exists() for p in folders)
    valid=protocol_valid_count(runs)
    return {'completed_trace_count':traces,'protocol_valid_process_count':valid,
            'measurement_invalid_count':len(folders)-valid}


def run():
    audit=freeze_sources()
    continuation=read(ROOT/'continuation-manifest.json')
    rows=continuation['pending_rows']
    safety=yaml.safe_load((ROOT/'e1-safety-stops.yaml').read_text())
    if safety['infrastructure_stop']['schedule_wall_clock_over_sec']!=3600 or \
            safety['infrastructure_stop']['automatic_retry']!='forbidden':
        raise ValueError('E1 cap or retry rule differs')
    if not (ROOT/'continuation-started.marker.json').exists():
        atomic(ROOT/'continuation-started.marker.json',{'start_wall_ns':time.time_ns(),
            'source_freeze_sha256':sha(ROOT/'source-freeze.json'),'pending_processes':18})
    started=read(ROOT/'continuation-started.marker.json')['start_wall_ns']/1e9
    completed=completed_rows(rows)
    setup_container()
    container=container_identity();nvml=Nvml()
    passed={}
    from eat_nav_e1_analyze import duplicate_gate,final_analysis
    for sid in ('g0e-train-000','g0e-train-001'):
        marker=ROOT/f'duplicate-gate-{sid}.json'
        if marker.exists():
            gate=read(marker)
            if gate['status']!='pass':raise ValueError('saved duplicate gate failure')
            passed[sid]='pass'
    while (row:=next_pending(rows,completed)) is not None:
        require_duplicate_before_lateral(row,passed)
        elapsed=time.time()-started
        if elapsed>=3600:raise TimeoutError('continuation wall cap exceeded')
        durations=[read(ROOT/'runs'/name/'run-complete.json')['elapsed_sec'] for name in completed]
        per_run=max(150.,float(np.mean(durations)) if durations else 150.)
        projected=elapsed+(18-len(completed))*per_run*1.2
        if projected>3600:
            atomic(ROOT/'forecast-stop.json',{'elapsed_sec':elapsed,'completed':len(completed),
                'projected_total_sec':projected,'reason':'60min_cap_projection'})
            raise TimeoutError('continuation projected over 60 minutes')
        atomic(ROOT/'status.json',{'stage':'E1_BASELINE','run_id':row['process_id'],
            'completed_new_processes':len(completed),'elapsed_sec':elapsed})
        baseline=prelaunch_baseline(row,container,nvml)
        if baseline['status']!='pass':raise RuntimeError('prelaunch baseline '+repr(baseline))
        entry=execution_entry(row)
        atomic(ROOT/'status.json',{'stage':'E1_RUNNING','run_id':row['process_id'],
            'completed_new_processes':len(completed),'elapsed_sec':time.time()-started})
        replay.run_one(entry,row['process_id'],timeout_sec=600,timing_telemetry=True,
            timing_monitor_script=REPO/'scripts/eat_nav_e1_monitor.py',
            timing_monitor_config=ROOT/'e1-safety-stops.yaml',
            timing_monitor_extra_args=('--sim-source',replay.REMOTE_CODE),
            monitor_cleanup_protocol=True)
        completed=completed_rows(rows)
        atomic(ROOT/'run-stage-markers'/f"{row['process_id']}.json",{'status':'complete',
            'run_complete_sha256':sha(ROOT/'runs'/row['process_id']/'run-complete.json'),
            'completed_new_processes':len(completed)})
        group=[r for r in rows if r['state_group_id']==row['state_group_id']]
        if row==group[2]:
            result=duplicate_gate(group[:3],ROOT,yaml.safe_load((ROOT/'e1-gates.yaml').read_text()))
            atomic(ROOT/f"duplicate-gate-{row['state_group_id']}.json",result)
            if result['status']!='pass':
                raise RuntimeError('duplicate gate failed; E1 measurement_invalid')
            passed[row['state_group_id']]='pass'
        if row==group[-1]:
            atomic(ROOT/f"group-complete-{row['state_group_id']}.json",{
                'status':'complete','run_marker_sha256':[sha(ROOT/'runs'/r['process_id']/'run-complete.json') for r in group]})
    final_analysis(ROOT)
    atomic(ROOT/'status.json',{'stage':'E1_COMPLETE','completed_new_processes':18,
        'elapsed_sec':time.time()-started,'learning_started':False})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only:
        audit=freeze_sources()
        print('E1_START_AUDIT_READY',ROOT/'source-freeze.json',flush=True)
    else:
        try:run()
        except BaseException as exc:
            atomic(ROOT/'status.json',{'stage':'E1_STOPPED','reason':repr(exc),
                'completed_new_processes':len([p for p in (ROOT/'runs').glob('*/run-complete.json')]) if (ROOT/'runs').exists() else 0,
                'learning_started':False})
            print('E1_STOP',repr(exc),flush=True)
            raise
