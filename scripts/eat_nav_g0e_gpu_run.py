"""One approved G0-E startup probe; never produces a formal anchor measurement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from eat_nav_g0e_gpu_probe import (MIB, Nvml, account, atomic, cgroup_memory,
    descendants, host_available_bytes, nvidia_smi_processes, proc_info, proc_table, sha)

ROOT=Path('/tmp/eatnav-g0e-gpu-attribution')
CONFIG=ROOT/'gpu-attribution-config.yaml'
PREVIOUS=Path('/tmp/eatnav-candidate-bank-g0e')
DOCKER='isaac-lab-onix'
REMOTE=str(ROOT)
ENV='PYTHONPATH=/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src:/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:/tmp/eatnav-extended/isaac-source'


def docker(*args,check=True):
    return subprocess.run(['docker',*args],capture_output=True,text=True,check=check)


def append(path,row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def source_config():
    import yaml
    cfg=yaml.safe_load(CONFIG.read_text())
    if cfg['scope']!='one_startup_probe_no_formal_anchor' or cfg['GPU_cap_mib']!=6656 or cfg['max_runs']!=1:
        raise ValueError('diagnostic scope/cap drift')
    for path,digest in cfg['frozen_sha256'].items():
        if sha(path)!=digest:raise ValueError(f'frozen SHA drift: {path}')
    return cfg


def container_identity():
    data=docker('inspect','-f','{{.Id}} {{.State.Pid}} {{.State.Status}}',DOCKER).stdout.split()
    if len(data)!=3 or data[2]!='running':raise RuntimeError('Docker container unavailable')
    info=proc_info(int(data[1]))
    if info is None or info['namespace_pid_chain'][-1]!=1 or data[0] not in info['cgroup']:
        raise RuntimeError('container host/container PID or cgroup mapping unavailable')
    return {'container_id':data[0], 'init_host_pid':int(data[1]),
            'init_container_pid':1,'init_start_ticks':info['start_ticks'],
            'cgroup':info['cgroup']}


def own_roots(table, container_id):
    return {pid for pid,info in table.items() if container_id in info['cgroup']
            and '/tmp/eatnav-g0e-gpu-attribution/diag_sim.py' in info['command_line']
            and '--output /tmp/eatnav-g0e-gpu-attribution/run' in info['command_line']}


def stage_events():
    result=docker('exec',DOCKER,'cat',f'{REMOTE}/run/timing-events.jsonl',check=False)
    if result.returncode:return []
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def container_proc_details(container_pids):
    if not container_pids:return {}
    script=('import json,os,pathlib,sys; out={}; '
        'pids=[int(x) for x in sys.argv[1:]]; '
        '\nfor pid in pids:\n'
        ' p=pathlib.Path("/proc")/str(pid)\n'
        ' try:\n'
        '  lines=(p/"smaps_rollup").read_text().splitlines()\n'
        '  pss=next((int(x.split()[1])*1024 for x in lines if x.startswith("Pss:")),None)\n'
        '  fds=list((p/"fd").iterdir())\n'
        '  sockets=sum("socket:[" in os.readlink(f) for f in fds)\n'
        '  out[pid]={"executable":os.readlink(p/"exe"),"pss_bytes":pss,"fd_count":len(fds),"socket_count":sockets}\n'
        ' except (OSError,StopIteration):out[pid]={"error":"process_exited_or_proc_unavailable"}\n'
        'print(json.dumps(out))')
    result=docker('exec',DOCKER,'/usr/bin/python3','-c',script,*map(str,container_pids),check=False)
    return json.loads(result.stdout) if result.returncode==0 else {'error':result.stderr}


def snapshot(nvml, baseline_ids, container, docker_client_pid=None):
    now_mono=time.monotonic_ns();now_wall=time.time_ns()
    devices,gpu_rows=nvml.sample();table=proc_table()
    roots=own_roots(table,container['container_id'])
    acct=account(devices,gpu_rows,table,roots,baseline_ids,container['container_id'])
    tree=acct['simulator_tree_host_pids']
    mem=cgroup_memory(container['container_id'])
    result={'monotonic_ns':now_mono,'wall_ns':now_wall,'devices':devices,
        'gpu_accounting':acct,'tree_processes':[table[p] for p in tree if p in table],
        'container_tree_details':container_proc_details([table[p]['container_pid'] for p in tree if p in table]),
        'docker_exec_client_host_pid':docker_client_pid,
        'cgroup_memory':mem,'host_mem_available_bytes':host_available_bytes(),
        'stage_events':stage_events() if docker_client_pid is not None else []}
    return result


def stop_reason(s,cfg,baseline_oom):
    a=s['gpu_accounting']; mem=s['cgroup_memory']
    if a['device_total_used_bytes']>cfg['GPU_cap_mib']*MIB:return 'device_GPU_cap'
    if a['unresolved_pid_bytes']:return 'GPU_PID_unresolved'
    if abs(a['unattributed_bytes'])>cfg['accounting_unattributed_max_mib']*MIB:return 'GPU_accounting_gap'
    if mem['anon_bytes']+mem['shmem_bytes']>cfg['anon_shmem_cap_bytes']:return 'anon_shmem_cap'
    if s['host_mem_available_bytes']<cfg['host_mem_available_min_bytes']:return 'host_available_floor'
    if mem['events'].get('oom',0)>baseline_oom or mem['events'].get('oom_kill',0)>0:return 'OOM_event'
    return None


def signal_owned_container(identities, container_id, sig):
    """Verify host identity, then signal the corresponding container PID as Docker root."""
    sent=[];skipped=[]
    flag='-TERM' if sig==signal.SIGTERM else '-KILL'
    for pid,start in sorted(identities,reverse=True):
        current=proc_info(pid)
        if current is None or current['start_ticks']!=start or container_id not in current['cgroup']:
            skipped.append({'host_pid':pid,'reason':'exited_reused_or_not_this_container'});continue
        container_pid=current['container_pid']
        result=docker('exec',DOCKER,'kill',flag,str(container_pid),check=False)
        sent.append({'host_pid':pid,'container_pid':container_pid,'signal':flag,
                     'exit_code':result.returncode,'stderr':result.stderr})
    return {'sent':sent,'skipped':skipped}


def cleanup_owned(identities, docker_client, log, container_id):
    record={'TERM':signal_owned_container(identities,container_id,signal.SIGTERM),'KILL':None,'docker_client_signal':None}
    if docker_client.poll() is None:
        docker_client.terminate();record['docker_client_signal']='SIGTERM'
    try:docker_client.wait(timeout=10)
    except subprocess.TimeoutExpired:
        remaining={(p,t) for p,t in identities if (x:=proc_info(p)) and x['start_ticks']==t}
        record['KILL']=signal_owned_container(remaining,container_id,signal.SIGKILL)
        docker_client.kill();record['docker_client_signal']='SIGKILL';docker_client.wait(timeout=10)
    log.flush();os.fsync(log.fileno());log.close()
    remaining={(p,t) for p,t in identities if (x:=proc_info(p)) and x['start_ticks']==t}
    if remaining:
        record['KILL']=signal_owned_container(remaining,container_id,signal.SIGKILL)
        time.sleep(1)
    record['remaining_owned']=[p for p,t in identities if (x:=proc_info(p)) and x['start_ticks']==t]
    return record


def run():
    cfg=source_config();ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'run-started.marker.json').exists():raise FileExistsError('single probe already launched; no retry')
    container=container_identity();nvml=Nvml()
    for remote,digest in cfg['container_frozen_sha256'].items():
        actual=docker('exec',DOCKER,'sha256sum',remote).stdout.split()[0]
        if actual!=digest:raise ValueError('container input SHA drift '+remote)
    atomic(ROOT/'status.json',{'stage':'D0_frozen','E1_started':False,'formal_anchor_started':False})
    before=proc_table();preexisting={(pid,p['start_ticks']) for pid,p in before.items()}
    atomic(ROOT/'prelaunch-process-table.json',{'container':container,
        'gpu_processes':[p for p in before.values() if 'nvidia' in p['command_line'].lower() or
            'Xorg' in p['command_line'] or 'gnome-shell' in p['command_line']],
        'previous_g0e_residual':[p for p in before.values() if '/tmp/eatnav-candidate-bank-g0e/eat_nav_g0e_anchor_sim.py' in p['command_line']],
        'tmux':subprocess.run(['tmux','list-sessions'],capture_output=True,text=True).stdout})
    baseline=[]
    start=time.monotonic()
    while time.monotonic()-start<cfg['baseline_seconds']:
        s=snapshot(nvml,preexisting,container)
        append(ROOT/'prelaunch-gpu-baseline.jsonl',s)
        baseline.append(s)
        time.sleep(max(0,cfg['sample_period_sec']-(time.monotonic_ns()-s['monotonic_ns'])/1e9))
    atomic(ROOT/'raw-nvidia-smi-baseline.json',nvidia_smi_processes())
    max_base=max(s['gpu_accounting']['device_total_used_bytes'] for s in baseline)
    baseline_oom=baseline[-1]['cgroup_memory']['events'].get('oom',0)
    reason=None
    if len(baseline)<10:reason='baseline_too_short'
    elif max_base>=cfg['GPU_cap_mib']*MIB or max_base>cfg['GPU_cap_mib']*MIB-cfg['required_headroom_mib']*MIB:reason='external_GPU_occupancy'
    elif any(s['gpu_accounting']['unresolved_pid_bytes'] or abs(s['gpu_accounting']['unattributed_bytes'])>cfg['accounting_unattributed_max_mib']*MIB for s in baseline):reason='baseline_GPU_accounting_unresolved'
    elif any(s['cgroup_memory']['anon_bytes']+s['cgroup_memory']['shmem_bytes']>cfg['anon_shmem_cap_bytes'] or s['host_mem_available_bytes']<cfg['host_mem_available_min_bytes'] for s in baseline):reason='baseline_memory_abnormal'
    elif own_roots(proc_table(),container['container_id']):reason='residual_diagnostic_simulator'
    elif any('/tmp/eatnav-candidate-bank-g0e/eat_nav_g0e_anchor_sim.py' in p['command_line'] for p in proc_table().values()):reason='previous_G0E_residual'
    atomic(ROOT/'prelaunch-gate.json',{'status':'pass' if reason is None else 'not_started',
        'reason':reason,'baseline_samples':len(baseline),'max_device_used_mib':max_base/MIB,
        'last_gpu_accounting':baseline[-1]['gpu_accounting']})
    if reason:
        atomic(ROOT/'status.json',{'stage':'D2_not_started','reason':reason,'E1_started':False})
        print('G0E_GPU_NOT_STARTED',reason,flush=True)
        return
    # Diagnostic source and frozen inputs are copied only after all prelaunch gates pass.
    for host,remote in [(ROOT/'diag_sim.py',f'{REMOTE}/diag_sim.py'),
        (PREVIOUS/'inputs/selected-states.json',f'{REMOTE}/inputs/selected-states.json'),
        (PREVIOUS/'inputs/state-000/prefix-actions.npz',f'{REMOTE}/inputs/state-000/prefix-actions.npz'),
        (PREVIOUS/'inputs/state-000/logged-anchor.json',f'{REMOTE}/inputs/state-000/logged-anchor.json'),
        (PREVIOUS/'inputs/state-000/anchor-config.yaml',f'{REMOTE}/inputs/state-000/anchor-config.yaml')]:
        docker('exec',DOCKER,'mkdir','-p',str(Path(remote).parent))
        docker('cp',str(host),f'{DOCKER}:{remote}')
        digest=docker('exec',DOCKER,'sha256sum',remote).stdout.split()[0]
        if digest!=sha(host):raise ValueError('Docker input copy SHA mismatch '+str(host))
    docker('exec',DOCKER,'mkdir','-p',f'{REMOTE}/run')
    command=['docker','exec','-e',ENV,DOCKER,'/isaac-sim/python.sh',f'{REMOTE}/diag_sim.py',
        '--headless','--stage','anchor','--repeat-id','0','--logical-t0-ns','2300000000000',
        '--manifest','/tmp/eatnav-extended/output/manifest.json',
        '--prefix',f'{REMOTE}/inputs/state-000/prefix-actions.npz',
        '--logged-anchor',f'{REMOTE}/inputs/state-000/logged-anchor.json',
        '--state',f'{REMOTE}/inputs/selected-states.json',
        '--config',f'{REMOTE}/inputs/state-000/anchor-config.yaml',
        '--reference-config','/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
        '--rl-checkpoint','/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
        '--output',f'{REMOTE}/run','--timing-telemetry']
    atomic(ROOT/'run-started.marker.json',{'scope':'one_startup_probe_no_formal_anchor','command':command,
        'diagnostic_source_sha256':sha(ROOT/'diag_sim.py'),'start_wall_ns':time.time_ns()})
    log=(ROOT/'runner-simulator.log').open('x')
    sim=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(ROOT/'status.json',{'stage':'D3_running','docker_exec_client_host_pid':sim.pid,'E1_started':False})
    runtime_start=time.monotonic();owned=set();stop=None;peak={'device':0,'simulator':0,'other':0,'unattributed':0}
    last_stage=None;last_stage_time=runtime_start;stable=0;seen_root=False;sample_index=0
    try:
        while True:
            s=snapshot(nvml,preexisting,container,sim.pid)
            sample_index+=1
            a=s['gpu_accounting']
            for k,source in [('device','device_total_used_bytes'),('simulator','simulator_tree_used_bytes'),('other','other_processes_used_bytes')]:
                peak[k]=max(peak[k],a[source])
            peak['unattributed']=max(peak['unattributed'],abs(a['unattributed_bytes']))
            owned.update((p['host_pid'],p['start_ticks']) for p in s['tree_processes'])
            if s['tree_processes']:seen_root=True
            append(ROOT/'gpu-process-timeseries.jsonl',s)
            append(ROOT/'simulator-process-tree.jsonl',{'monotonic_ns':s['monotonic_ns'],'tree':s['tree_processes']})
            if sample_index==1 or sample_index%5==0:
                append(ROOT/'raw-nvidia-smi-timeseries.jsonl',nvidia_smi_processes())
            events=s['stage_events'];stage=events[-1]['event'] if events else None
            if stage!=last_stage:
                last_stage=stage;last_stage_time=time.monotonic();stable=0
                if stage:atomic(ROOT/'stage-current.json',{'stage':stage,'wall_ns':s['wall_ns']})
            if stage=='initial_reset_complete':stable+=1
            stop=stop_reason(s,cfg,baseline_oom)
            if stop is None and seen_root and not s['tree_processes'] and sim.poll() is None:
                stop='simulator_PID_ownership_lost'
            if stop is None and not seen_root and time.monotonic()-runtime_start>15:
                stop='simulator_PID_not_found'
            if stop is None and stage and time.monotonic()-last_stage_time>cfg['stage_stall_sec']:
                stop='startup_stage_stall'
            if stop:
                atomic(ROOT/'resource-stop-snapshot.json',{'reason':stop,'snapshot':s,
                    'previous_snapshot_sha256':sha(ROOT/'gpu-process-timeseries.jsonl')})
                break
            if any(e['event']=='first_control_step_complete' for e in events):stop='first_control_step_complete';break
            if stable>=5:stop='post_reset_five_stable_samples';break
            if time.monotonic()-runtime_start>=cfg['startup_wall_limit_sec']:stop='startup_wall_limit';break
            if sim.poll() is not None:stop='process_exited';break
            time.sleep(max(0,cfg['sample_period_sec']-(time.monotonic_ns()-s['monotonic_ns'])/1e9))
    except BaseException as exc:
        stop='monitor_error_'+type(exc).__name__
        atomic(ROOT/'monitor-failure.marker.json',{'stop':stop,'error':repr(exc),
            'partial_timeseries_sha256':sha(ROOT/'gpu-process-timeseries.jsonl') if (ROOT/'gpu-process-timeseries.jsonl').exists() else None})
    # Record ownership once more, then clean up only this run's verified PID identities.
    table=proc_table();roots=own_roots(table,container['container_id'])
    owned.update((p,table[p]['start_ticks']) for p in descendants(table,roots) if p in table)
    cleanup=cleanup_owned(owned,sim,log,container['container_id'])
    docker('cp',f'{DOCKER}:{REMOTE}/run/.',str(ROOT/'run'),check=False)
    atomic(ROOT/'cleanup-result.json',{'stop_reason':stop,'cleanup':cleanup,
        'simulator_client_exit_code':sim.returncode,'own_process_residual':cleanup['remaining_owned'],
        'other_processes_signaled':False,'raw_log_sha256':sha(ROOT/'runner-simulator.log')})
    atomic(ROOT/'status.json',{'stage':'D3_complete' if not cleanup['remaining_owned'] else 'D5_cleanup_failed',
        'stop_reason':stop,'peak_bytes':peak,'E1_started':False,'formal_anchor_result':False})
    atomic(ROOT/'probe-complete.marker.json',{'stop_reason':stop,'peak_bytes':peak,
        'cleanup_sha256':sha(ROOT/'cleanup-result.json'),
        'timeseries_sha256':sha(ROOT/'gpu-process-timeseries.jsonl'),
        'formal_anchor_result':False})
    print('G0E_GPU_PROBE_COMPLETE',stop,{k:round(v/MIB,1) for k,v in peak.items()},flush=True)


if __name__=='__main__':run()
