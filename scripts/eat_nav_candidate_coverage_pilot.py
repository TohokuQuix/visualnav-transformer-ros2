"""Resume-safe 119-run fresh-process train pilot with hard four-hour wall clock."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
import yaml
from visualnav_transformer.eat_nav.candidate_coverage_pilot import duplicate_gate,map_paths_from_anchor,may_start_next
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest

ROOT=Path('/tmp/eatnav-candidate-coverage')
HOST_INPUT=ROOT/'pilot-inputs'
CONTAINER_INPUT='/tmp/eatnav-candidate-coverage/pilot-inputs'
CASE=Path('/tmp/eatnav-vnt-candidate-coverage/scripts/eat_nav_candidate_coverage_case.sh')


def status_write(value):
    path=ROOT/'pilot/status.json';path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('status.json.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def verified_case(run):
    stage='anchor' if run['kind']=='anchor' else 'live'
    folder=ROOT/'pilot/runs'/run['run_id']
    marker=folder/f'{stage}-0.marker.json'
    if not marker.exists():return False
    data=json.loads(marker.read_text())
    for source_path,expected in data['outputs_sha256'].items():
        host=folder/Path(source_path).name
        if not host.exists() or digest(host)!=expected:
            raise ValueError(f'run marker output SHA mismatch: {host}')
    return True


def case_elapsed(run):
    log=ROOT/'pilot/runs'/run['run_id']/'runner.log'
    if not log.exists():return None
    text=log.read_text(errors='replace')
    begin=re.search(r'RUN_BEGIN_UNIX (\d+)',text)
    end=re.search(r'RUN_EXIT_UNIX (\d+) id='+re.escape(run['run_id'])+r' code=0',text)
    return float(int(end.group(1))-int(begin.group(1))) if begin and end else None


def expected_seconds(run):
    return 46.10692857959084+.8296659776629849*run['full_prefix_steps']


def gpu_sampler(stop_event,path,values):
    with path.open('a') as stream:
        while not stop_event.is_set():
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.used',
                '--format=csv,noheader,nounits'],capture_output=True,text=True)
            if result.returncode==0:
                try:
                    used=int(result.stdout.splitlines()[0].strip());values.append(used)
                    stream.write(f'{time.time():.3f},{used}\n');stream.flush()
                except (ValueError,IndexError):pass
            stop_event.wait(2.)


def startup_crash_count(log):
    """Count Isaac Kit crashes before simulation starts across resumed attempts."""
    attempts=log.split('RUN_BEGIN_UNIX ')[1:]
    return sum("Segmentation fault" in attempt and
               ("appState = 'startup'" in attempt or "UptimeSeconds = '1'" in attempt)
               for attempt in attempts)


def run_case(run,*,deadline,config):
    if verified_case(run):
        print('PILOT_RESUME_SKIP',run['run_id'],flush=True)
        return case_elapsed(run) or expected_seconds(run),None
    stage='anchor' if run['kind']=='anchor' else 'live'
    name=('bank-nominal' if run['kind'] in ('anchor','duplicate') else
          f"nomad-seed-{run['seed']}" if run['kind']=='nomad' else f"bank-{run['name']}")
    index=int(run['state_id'].rsplit('-',1)[1])
    if stage=='live':
        with socket.create_connection(('127.0.0.1',47819),timeout=2.):pass
    prior_log=ROOT/'pilot/runs'/run['run_id']/'runner.log'
    errors=startup_crash_count(prior_log.read_text(errors='replace')) if prior_log.exists() else 0
    while True:
        remaining=deadline-time.time()
        if remaining<=0:raise TimeoutError('four-hour wall-clock deadline reached')
        stop=threading.Event();values=[]
        gpu_path=ROOT/'pilot/runs'/run['run_id']/'gpu-memory.csv';gpu_path.parent.mkdir(parents=True,exist_ok=True)
        thread=threading.Thread(target=gpu_sampler,args=(stop,gpu_path,values),daemon=True);thread.start()
        started=time.monotonic()
        process=subprocess.Popen([str(CASE),run['run_id'],str(index),name,stage],start_new_session=True)
        timed_out=False
        try:code=process.wait(timeout=min(remaining,2*expected_seconds(run)))
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code=124
        finally:stop.set();thread.join(timeout=3.)
        elapsed=time.monotonic()-started
        peak=max(values) if values else None
        if peak is not None and peak>config['gpu_memory_unexpected_mib']:
            raise RuntimeError(f'unexpected GPU memory peak {peak} MiB')
        if elapsed>config['runtime_over_estimate_stop_ratio']*expected_seconds(run) or timed_out:
            raise RuntimeError(f'unexpected runtime {elapsed:.1f}s for {run["run_id"]}, expected {expected_seconds(run):.1f}s')
        if code==0 and verified_case(run):return elapsed,peak
        log=(ROOT/'pilot/runs'/run['run_id']/'runner.log').read_text(errors='replace').rsplit('RUN_BEGIN_UNIX ',1)[-1]
        startup_crash=startup_crash_count('RUN_BEGIN_UNIX '+log)>0
        infra=startup_crash or any(term in log for term in ('infrastructure_invalid','follower command timeout',
                                          'ConnectionRefusedError','stale follower command'))
        errors+=1
        if not infra or errors>config['infrastructure_retry_max_per_run']:
            raise RuntimeError(f'run {run["run_id"]} failed code={code}, infrastructure={infra}, retries={errors-1}')
        print('PILOT_SAME_CANDIDATE_RETRY',run['run_id'],errors,flush=True)


def freeze_map_paths(run,freeze):
    index=int(run['state_id'].rsplit('-',1)[1]);folder=HOST_INPUT/f'state-{index:03d}'
    anchor=ROOT/'pilot/runs'/run['run_id']/'anchor-0.json'
    if not anchor.exists():raise ValueError('anchor summary missing')
    summary=json.loads(anchor.read_text())
    expected=freeze['states'][run['state_id']]['base_path_sha256']
    base={name:json.loads((folder/f'{name}-base-path.json').read_text()) for name in expected}
    for name,sha in expected.items():
        if digest(folder/f'{name}-base-path.json')!=sha:raise ValueError('frozen candidate Path changed')
    mapped=map_paths_from_anchor(summary,base)
    hashes={}
    for name,points in mapped.items():
        if name=='bank-nominal':points=summary['map_path']
        target=folder/f'{name}-map-path.json';atomic_once(target,points)
        hashes[name]=digest(target)
        container=f'{CONTAINER_INPUT}/state-{index:03d}/{target.name}'
        check=subprocess.run(['docker','exec','isaac-lab-onix','sha256sum',container],capture_output=True,text=True)
        if check.returncode==0:
            if check.stdout.split()[0]!=hashes[name]:raise ValueError('container map Path differs')
        else:
            subprocess.run(['docker','cp',str(target),f'isaac-lab-onix:{container}'],check=True)
    result={'anchor_run_id':run['run_id'],'anchor_summary_sha256':digest(anchor),
            'fixed_map_path_sha256':hashes,'source_base_path_sha256':expected}
    atomic_once(folder/'map-path-freeze.json',result)


def compare_duplicates(state_id,schedule,config):
    rows=[r for r in schedule if r['state_id']==state_id and r['kind']=='duplicate']
    summaries=[];traces=[];sha=[]
    for run in rows:
        folder=ROOT/'pilot/runs'/run['run_id']
        if not verified_case(run):raise ValueError('duplicate marker missing')
        summary_path=folder/'live-0.json';trace_path=folder/'live-0.trace.json.gz'
        summaries.append(json.loads(summary_path.read_text()))
        with gzip.open(trace_path,'rt') as stream:traces.append(json.load(stream))
        sha.append({'summary':digest(summary_path),'trace':digest(trace_path)})
    result=duplicate_gate(summaries,traces,xy_rms_limit_m=config['duplicate_xy_rms_max_m'])
    result.update({'state_id':state_id,'source_run_sha256':sha,'measurement_valid':True})
    target=ROOT/'pilot/duplicate-gates'/f'{state_id}.json'
    atomic_once(target,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    config_path=Path('/tmp/eatnav-vnt-candidate-coverage/docs/quality-identifiability-gate/candidate-coverage/pilot-config.yaml')
    selection_path=ROOT/'stage0/frozen/pilot-selection.json'
    schedule_path=ROOT/'stage0/frozen/pilot-schedule.json'
    budget=json.loads((ROOT/'pilot/budget-start.json').read_text())
    if budget['input_sha256']!={'config':digest(config_path),'selection':digest(selection_path),'schedule':digest(schedule_path)}:
        raise ValueError('budget input SHA mismatch')
    previous_status=json.loads((ROOT/'pilot/status.json').read_text())
    if previous_status.get('state') in ('stopped','budget_stop'):
        review_path=ROOT/'pilot/reviewed-startup-crash.json'
        if previous_status.get('state')!='stopped' or not review_path.exists():
            raise ValueError('pilot was stopped by a gate; resume requires review')
        review=json.loads(review_path.read_text())
        crash_log=ROOT/'pilot/runs/pilot-013/runner.log'
        allowed=(review.get('run_id')=='pilot-013' and
                 review.get('status_sha256')==digest(ROOT/'pilot/status-at-kit-startup-crash.json') and
                 review.get('crash_log_sha256')==digest(crash_log) and
                 previous_status['next_run_id']=='pilot-013' and
                 startup_crash_count(crash_log.read_text(errors='replace'))==1 and
                 all((ROOT/'pilot/completed'/f'pilot-{i:03d}.json').exists() for i in range(13)))
        if not allowed:
            raise ValueError('stopped pilot review does not match one pre-simulation Kit crash')
    config=yaml.safe_load(config_path.read_text());schedule=json.loads(schedule_path.read_text())['runs']
    freeze=json.loads((HOST_INPUT/'input-freeze.json').read_text())
    if len(schedule)!=119 or freeze['schedule_sha256']!=digest(schedule_path):raise ValueError('frozen schedule/input mismatch')
    deadline=budget['deadline_unix_sec'];started=budget['started_unix_sec']
    completed=[];observed=[];last_peak=None
    def forecast(index):
        baseline=sum(expected_seconds(r) for r in schedule[index:])
        ratio=max(1.,sum(observed)/sum(expected_seconds(r) for r in completed)) if completed else 1.
        return baseline*ratio
    def update(state,index,reason=None):
        elapsed=time.time()-started;remaining=forecast(index)
        status_write({'stage':'train_pilot','state':state,'completed_runs':len(completed),
            'total_runs':119,'next_run_id':schedule[index]['run_id'] if index<len(schedule) else None,
            'elapsed_sec':elapsed,'deadline_unix_sec':deadline,'forecast_remaining_sec':remaining,
            'forecast_finish_unix_sec':time.time()+remaining,'three_hour_review_due':elapsed>=10800,
            'last_gpu_peak_mib':last_peak,'stop_reason':reason,
            'selection_sha256':digest(selection_path),'schedule_sha256':digest(schedule_path),
            'formal_validation_started':False,'retraining_started':False,'real_robot_started':False})
    try:
        for index,run in enumerate(schedule):
            if run['kind'] in ('nomad','bank'):
                gate=ROOT/'pilot/duplicate-gates'/f"{run['state_id']}.json"
                if not gate.exists() or not json.loads(gate.read_text())['measurement_valid']:
                    raise ValueError('candidate run without passing duplicate gate')
            if not verified_case(run):
                remaining=forecast(index)
                if not may_start_next(elapsed_sec=time.time()-started,predicted_remaining_sec=remaining,
                                      approved_budget_sec=config['approved_wall_clock_sec']):
                    update('budget_stop',index,'completion forecast exceeds four-hour budget');return
            update('running',index)
            elapsed,peak=run_case(run,deadline=deadline,config=config)
            observed.append(elapsed);completed.append(run)
            if peak is not None:last_peak=peak
            if run['kind']=='anchor':freeze_map_paths(run,freeze)
            if run['kind']=='duplicate' and run['name']==2:
                gate=compare_duplicates(run['state_id'],schedule,config)
                print('PILOT_DUPLICATE_GATE',run['state_id'],gate['duplicate_xy_rms_m'],flush=True)
            marker=ROOT/'pilot/runs'/run['run_id']/('anchor-0.marker.json' if run['kind']=='anchor' else 'live-0.marker.json')
            completion=ROOT/'pilot/completed'/f"{run['run_id']}.json"
            if completion.exists():
                prior=json.loads(completion.read_text())
                if prior['run']!=run or prior['case_marker_sha256']!=digest(marker):
                    raise ValueError('completed run marker/source mismatch')
            else:
                atomic_once(completion,{'run':run,'case_marker_sha256':digest(marker),
                    'wall_elapsed_sec':elapsed,'gpu_peak_mib':peak,
                    'selection_sha256':digest(selection_path),'schedule_sha256':digest(schedule_path)})
            print('PILOT_RUN_COMPLETE',run['run_id'],run['terrain_type'],run['kind'],round(elapsed,1),flush=True)
        update('raw_runs_complete',119)
    except BaseException as exc:
        update('stopped',len(completed),repr(exc))
        print('PILOT_STOP',repr(exc),flush=True)
        raise

if __name__=='__main__':main()
