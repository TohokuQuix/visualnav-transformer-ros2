"""External per-PID monitor for one E1 fresh simulator and its cleanup marker."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from eat_nav_g0e_formal_gate import runtime_gate
from eat_nav_g0e_gpu_probe import (MIB,Nvml,account,atomic,cgroup_memory,descendants,
    host_available_bytes,nvidia_smi_processes,proc_table,sha)
from eat_nav_g0e_gpu_run import container_identity,container_proc_details


def append(path,row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


def roots(table,container_id,remote_output):
    source='/tmp/eatnav-e1-feasibility-20261003/eat_nav_e1_sim.py'
    return {pid for pid,p in table.items() if container_id in p['cgroup']
        and source in p['command_line'] and '--output '+remote_output in p['command_line']}


def sample(nvml,container,baseline_ids,remote_output):
    devices,rows=nvml.sample();table=proc_table()
    root=roots(table,container['container_id'],remote_output)
    accounting=account(devices,rows,table,root,baseline_ids,container['container_id'])
    tree=[table[pid] for pid in accounting['simulator_tree_host_pids'] if pid in table]
    cgroup=cgroup_memory(container['container_id'])
    detail=container_proc_details([p['container_pid'] for p in tree]) if tree else {}
    return {'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
        'devices':devices,'gpu_accounting':accounting,'simulator_tree':tree,
        'container_tree_details':detail,'cgroup_memory':cgroup,
        'host_available_bytes':host_available_bytes(),
        'simulator_tree_rss_bytes':sum(p['rss_bytes'] for p in tree),
        'simulator_tree_pss_bytes':sum(x.get('pss_bytes') or 0 for x in detail.values() if isinstance(x,dict)),
        'simulator_tree_fd_count':sum(p['fd_count'] or 0 for p in tree),
        'simulator_tree_thread_count':sum(p['thread_count'] for p in tree)}


def gate_sample(snapshot,baseline_oom):
    a=snapshot['gpu_accounting'];m=snapshot['cgroup_memory']
    return {'device_total_used_mib':a['device_total_used_bytes']/MIB,
        'simulator_process_tree_used_mib':a['simulator_tree_used_bytes']/MIB,
        'other_processes_used_mib':a['other_processes_used_bytes']/MIB,
        'unattributed_used_mib':a['unattributed_bytes']/MIB,
        'unresolved_pid_mib':a['unresolved_pid_bytes']/MIB,
        'anon_shmem_bytes':m['anon_bytes']+m['shmem_bytes'],
        'host_available_bytes':snapshot['host_available_bytes'],
        'oom_events_delta':m['events'].get('oom',0)-baseline_oom}


def limits(safety):
    s=safety['infrastructure_stop']
    return {'GPU_cap_mib':s['gpu_device_total_memory_over_mib'],
        'unattributed_cap_mib':s['gpu_unattributed_memory_over_mib'],
        'anon_shmem_cap_bytes':s['cgroup_anon_plus_shmem_over_bytes'],
        'host_memory_floor_bytes':s['host_mem_available_under_bytes']}


def remote_progress(path):
    result=subprocess.run(['docker','exec','isaac-lab-onix','cat',path],capture_output=True,text=True)
    return json.loads(result.stdout) if result.returncode==0 else None


def disk_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def verify_complete(run):
    complete=json.loads((run/'run-complete.json').read_text())
    final=json.loads((run/'final-status.json').read_text())
    if final['stage']!='COMPLETE' or final['residual_processes'] or final['sim_exit']!=0:
        raise ValueError('cleanup or simulator completion missing')
    for name,digest in complete['output_sha256'].items():
        if sha(run/name)!=digest:raise ValueError('complete marker output SHA mismatch '+name)
    return sha(run/'run-complete.json')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--remote-progress',required=True)
    parser.add_argument('--sim-host-pid',type=int,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--target-step',type=int,required=True)
    parser.add_argument('--target-episode',type=int,required=True)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    run=args.run;root=run.parents[1];remote=str(root/'runs'/args.run_id)
    safety=yaml.safe_load(args.config.read_text())
    container=container_identity();nvml=Nvml()
    table=proc_table();baseline_ids={(pid,p['start_ticks']) for pid,p in table.items()}
    baseline_oom=cgroup_memory(container['container_id'])['events'].get('oom',0)
    start=time.monotonic();last_progress=start;last_step=None;seen_root=False
    sample_count=0;exit_seen=None;reason=None
    while True:
        snap=sample(nvml,container,baseline_ids,remote)
        sample_count+=1
        append(run/'gpu-process-timeseries.jsonl',snap)
        if sample_count==1 or sample_count%10==0:
            append(run/'raw-nvidia-smi.jsonl',nvidia_smi_processes())
        gate=gate_sample(snap,baseline_oom)
        if snap['simulator_tree']:seen_root=True
        stop=runtime_gate(gate,limits(safety))
        if stop is None and not seen_root and time.monotonic()-start>15:
            stop='simulator_PID_not_found'
        proc=Path('/proc')/str(args.sim_host_pid)
        if not proc.exists() and exit_seen is None:exit_seen=time.monotonic()
        if stop is None and seen_root and not snap['simulator_tree'] and proc.exists():
            stop='simulator_PID_ownership_lost'
        progress=remote_progress(args.remote_progress)
        if progress is not None:
            step=progress['last_completed_step']
            if last_step!=step:last_step=step;last_progress=time.monotonic()
            if (time.monotonic_ns()-progress['monotonic_ns'])/1e9>safety['infrastructure_stop']['heartbeat_stale_over_sec']:
                stop=stop or 'simulator_heartbeat_stale'
        if stop is None and exit_seen is None and time.monotonic()-last_progress>safety['infrastructure_stop']['step_stall_over_sec']:
            stop='simulator_step_stall'
        if stop is None and disk_bytes(root)>safety['infrastructure_stop']['storage_over_bytes']:
            stop='storage_cap'
        heartbeat={'run_id':args.run_id,'wall_ns':snap['wall_ns'],'monotonic_ns':snap['monotonic_ns'],
            'global_step':last_step,'target_step':args.target_step,
            'target_episode':args.target_episode,'simulator_tree_pid_count':len(snap['simulator_tree']),
            'resource':gate,'simulator_rss_bytes':snap['simulator_tree_rss_bytes'],
            'simulator_pss_bytes':snap['simulator_tree_pss_bytes'],
            'simulator_fd_count':snap['simulator_tree_fd_count'],
            'simulator_thread_count':snap['simulator_tree_thread_count'],
            'output_bytes':disk_bytes(root),'stop_reason':stop}
        atomic(run/'timing-heartbeat.json',heartbeat)
        if stop:
            atomic(run/'resource-stop.failure.json',{'reason':stop,'snapshot':snap,
                'gpu_timeseries_sha256':sha(run/'gpu-process-timeseries.jsonl')})
            reason=stop;break
        if (run/'run-complete.json').exists():
            try:
                digest=verify_complete(run)
                atomic(run/'timing-monitor.done.json',{'status':'complete','lifecycle':'COMPLETE_MARKER',
                    'complete_marker_sha256':digest,'sample_count':sample_count})
                return
            except BaseException as exc:
                reason='complete_marker_invalid_'+repr(exc);break
        if (run/'monitor.stop').exists():
            atomic(run/'timing-monitor.done.json',{'status':'stopped','lifecycle':'MONITOR_STOP',
                'sample_count':sample_count})
            return
        if exit_seen is not None and time.monotonic()-exit_seen>120:
            reason='cleanup_marker_timeout';break
        time.sleep(.5)
    atomic(run/'timing-monitor.done.json',{'status':'failure','reason':reason,'sample_count':sample_count})
    raise RuntimeError(reason)


if __name__=='__main__':main()
