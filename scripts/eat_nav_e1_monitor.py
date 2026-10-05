"""External per-PID monitor for one E1 fresh simulator and its cleanup marker."""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import time

import yaml

from eat_nav_g0e_formal_gate import runtime_gate
from eat_nav_g0e_gpu_probe import (MIB,Nvml,account,atomic,cgroup_memory,descendants,
    host_available_bytes,nvidia_smi_processes,proc_table,sha)
from eat_nav_g0e_gpu_run import container_identity,container_proc_details
from eat_nav_e1_lifecycle import LifecycleMachine, marker_matches_run, read_complete_json
from eat_nav_e1_process_ownership import (container_owned_roots,ProcessRegistry,
    registered_alive)


def append(path,row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())


DEFAULT_SIM_SOURCE='/tmp/eatnav-e1-feasibility-20261003/eat_nav_e1_sim.py'


def roots(table,container_id,remote_output,source=DEFAULT_SIM_SOURCE):
    return {pid for pid,p in table.items() if container_id in p['cgroup']
        and source in p['command_line'] and '--output '+remote_output in p['command_line']}


def sample(nvml,container,baseline_ids,remote_output,source=DEFAULT_SIM_SOURCE,
           container_baseline=None):
    sample_start=time.monotonic_ns()
    devices,rows=nvml.sample()
    table_start=time.monotonic_ns();table=proc_table();table_end=time.monotonic_ns()
    devices_after=nvml.memory_snapshot()
    classify_start=time.monotonic_ns()
    root=(container_owned_roots(table,container['container_id'],container_baseline)
          if container_baseline is not None else roots(table,container['container_id'],remote_output,source))
    accounting=account(devices,rows,table,root,baseline_ids,container['container_id'])
    before_by_uuid={device['uuid']:device['used_bytes'] for device in devices}
    after_by_uuid={device['uuid']:device['used_bytes'] for device in devices_after}
    if len(before_by_uuid)!=len(devices) or set(before_by_uuid)!=set(after_by_uuid):
        raise RuntimeError('GPU UUID changed during bracketed sample')
    process_by_uuid={uuid:0 for uuid in before_by_uuid}
    for process in accounting['processes']:
        process_by_uuid[process['gpu_uuid']]+=process['used_gpu_memory_bytes']
    bracket=[]
    for uuid,before in before_by_uuid.items():
        after=after_by_uuid[uuid];total=process_by_uuid[uuid]
        gap=(min(before,after)-total if total<min(before,after) else
             max(before,after)-total if total>max(before,after) else 0)
        bracket.append({'gpu_uuid':uuid,'device_before_bytes':before,
                        'device_after_bytes':after,'process_sum_bytes':total,
                        'attribution_gap_bytes':gap})
    accounting['bracketed_unattributed_bytes']=max(
        (row['attribution_gap_bytes'] for row in bracket),key=abs)
    accounting['device_total_peak_bracket_bytes']=sum(
        max(row['device_before_bytes'],row['device_after_bytes']) for row in bracket)
    classify_end=time.monotonic_ns()
    tree=[table[pid] for pid in accounting['simulator_tree_host_pids'] if pid in table]
    cgroup=cgroup_memory(container['container_id'])
    detail=container_proc_details([p['container_pid'] for p in tree]) if tree else {}
    return {'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),
        'devices':devices,'gpu_accounting':accounting,'simulator_tree':tree,
        'gpu_acquisition':{'device_before':devices,'device_after':devices_after,
            'raw_process_rows':rows,'bracket':bracket,
            'query_timing':{**getattr(nvml,'last_query_timing',{}),
                'process_table_start_ns':table_start,'process_table_end_ns':table_end,
                **getattr(nvml,'last_memory_snapshot_timing',{}),
                'classification_start_ns':classify_start,
                'classification_end_ns':classify_end,
                'sample_start_ns':sample_start,
                'sample_end_ns':time.monotonic_ns()}},
        'container_tree_details':detail,'cgroup_memory':cgroup,
        'host_available_bytes':host_available_bytes(),
        'simulator_tree_rss_bytes':sum(p['rss_bytes'] for p in tree),
        'simulator_tree_pss_bytes':sum(x.get('pss_bytes') or 0 for x in detail.values() if isinstance(x,dict)),
        'simulator_tree_fd_count':sum(p['fd_count'] or 0 for p in tree),
        'simulator_tree_thread_count':sum(p['thread_count'] for p in tree)}


def gate_sample(snapshot,baseline_oom):
    a=snapshot['gpu_accounting'];m=snapshot['cgroup_memory']
    return {'device_total_used_mib':a.get('device_total_peak_bracket_bytes',
                                        a['device_total_used_bytes'])/MIB,
        'simulator_process_tree_used_mib':a['simulator_tree_used_bytes']/MIB,
        'other_processes_used_mib':a['other_processes_used_bytes']/MIB,
        'unattributed_used_mib':a.get('bracketed_unattributed_bytes',
                                    a['unattributed_bytes'])/MIB,
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


def heartbeat_record(progress, *, target_step, sample_monotonic_ns, host_pid, host_start_ticks):
    step=int(progress['last_completed_step'])
    return {'phase':'prefix' if step<=target_step else 'suffix',
        'global_step':step,'local_step':progress.get('episode_local_step'),
        'episode':progress.get('current_episode'),
        'simulator_container_pid':progress.get('pid'),
        'docker_exec_host_pid':host_pid,'docker_exec_start_ticks':host_start_ticks,
        'source_monotonic_ns':progress['monotonic_ns'],
        'sample_monotonic_ns':sample_monotonic_ns}


def disk_bytes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def finite(value):
    if isinstance(value,dict):return all(finite(x) for x in value.values())
    if isinstance(value,list):return all(finite(x) for x in value)
    return not isinstance(value,float) or math.isfinite(value)


def remote_completion(remote,expected_map_sha):
    marker_path=remote+'/live-0.marker.json'
    result=subprocess.run(['docker','exec','isaac-lab-onix','cat',marker_path],
                          capture_output=True,text=True)
    if result.returncode:return None
    try:marker=json.loads(result.stdout)
    except json.JSONDecodeError:return None
    if not marker_matches_run(marker,remote,expected_map_sha):
        return {'valid':False,'trace_closed':False,'trace_sha_match':False,
                'step_count':None,'normal_completion':False}
    sha_match=True
    for path,digest in marker['outputs_sha256'].items():
        output=subprocess.run(['docker','exec','isaac-lab-onix','sha256sum',path],
                              capture_output=True,text=True)
        if output.returncode or output.stdout.split()[0]!=digest:sha_match=False
    trace=remote+'/live-0.trace.json.gz'
    closed=subprocess.run(['docker','exec','isaac-lab-onix','gzip','-t',trace],
                          capture_output=True).returncode==0
    summary=subprocess.run(['docker','exec','isaac-lab-onix','cat',remote+'/live-0.json'],
                           capture_output=True,text=True)
    try:data=json.loads(summary.stdout) if summary.returncode==0 else {}
    except json.JSONDecodeError:data={}
    failed=subprocess.run(['docker','exec','isaac-lab-onix','test','-e',
                           remote+'/failure-0.marker.json'],capture_output=True).returncode==0
    return {'valid':sha_match and closed and not failed,'trace_closed':closed,
            'trace_sha_match':sha_match,'step_count':data.get('suffix_steps'),
            'normal_completion':not failed and data.get('stage')=='live'}


def follower_exit_status(run):
    path=run/'follower-logs/bridge-events.jsonl'
    if not path.exists():return None
    try:events=[json.loads(row) for row in path.read_text().splitlines()]
    except json.JSONDecodeError:return None
    exits=[x for x in events if x.get('event')=='follower_exit']
    if len(exits)!=8:return None
    return all(x.get('returncode')==0 for x in exits)


def zombie_count(run):
    path=run/'follower-logs/bridge-events.jsonl'
    if not path.exists():return 0
    try:events=[json.loads(row) for row in path.read_text().splitlines()]
    except json.JSONDecodeError:return 0
    pids={x['pid'] for x in events if x.get('event')=='follower_exit' and 'pid' in x}
    count=0
    for pid in pids:
        stat=Path('/proc')/str(pid)/'stat'
        if stat.exists() and stat.read_text().rsplit(')',1)[-1].strip().startswith('Z'):
            count+=1
    return count


def finalization_status(run):
    returned=read_complete_json(run/'docker-exec-return.json')
    final=read_complete_json(run/'final-status.json')
    complete=read_complete_json(run/'run-complete.json')
    followers=follower_exit_status(run)
    result={'docker_exit':returned.get('exit_code') if returned else None,
            'bridge_exit':final.get('bridge_exit') if final else None,
            'followers_exit_ok':followers,
            'run_complete_exists':complete is not None,
            'output_sha_match':None,'residual_count':len(final['residual_processes']) if final else 0,
            'zombie_count':zombie_count(run),
            'cleanup_marker':(run/'bridge-shutdown.json').exists() and final is not None}
    if complete is not None:
        try:verify_complete(run);result['output_sha_match']=True
        except (ValueError,KeyError,FileNotFoundError,json.JSONDecodeError):result['output_sha_match']=False
    return result


def verify_complete(run,*,source_run=None):
    complete=json.loads((run/'run-complete.json').read_text())
    final=json.loads((run/'final-status.json').read_text())
    returned=json.loads((run/'docker-exec-return.json').read_text())
    if final['stage']!='COMPLETE' or final['residual_processes'] or final['sim_exit']!=0 or final['bridge_exit']!=0 or returned['exit_code']!=0:
        raise ValueError('cleanup or simulator completion missing')
    if not (run/'bridge-shutdown.json').exists() or follower_exit_status(run) is not True or zombie_count(run):
        raise ValueError('bridge/follower cleanup missing')
    for name,digest in complete['output_sha256'].items():
        if sha(run/name)!=digest:raise ValueError('complete marker output SHA mismatch '+name)
    marker=json.loads((run/'live-0.marker.json').read_text())
    started=read_complete_json(run/'started.json') or {}
    expected_map_sha=complete['identity'].get('map_path_file_sha256') or started.get('input_sha256',{}).get('map_path')
    if not marker_matches_run(marker,str(source_run or run),expected_map_sha):
        raise ValueError('simulator marker identity mismatch')
    for name,digest in marker['outputs_sha256'].items():
        if sha(run/Path(name).name)!=digest:raise ValueError('simulator output SHA mismatch')
    with gzip.open(run/'live-0.trace.json.gz','rt') as stream:trace=json.load(stream)
    if len(trace)!=120 or not finite(trace):raise ValueError('120-step finite trace required')
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
    parser.add_argument('--sim-source',default=DEFAULT_SIM_SOURCE)
    parser.add_argument('--container-baseline',type=Path)
    parser.add_argument('--process-registry',type=Path)
    args=parser.parse_args()
    run=args.run;root=run.parents[1];remote=str(root/'runs'/args.run_id)
    safety=yaml.safe_load(args.config.read_text())
    container=container_identity();nvml=Nvml()
    table=proc_table();baseline_ids={(pid,p['start_ticks']) for pid,p in table.items()}
    baseline_oom=cgroup_memory(container['container_id'])['events'].get('oom',0)
    start=time.monotonic();last_progress=start;last_step=None;last_recorded_step=None;seen_root=False
    sample_count=0;reason=None;completion=None
    machine=LifecycleMachine(expected_steps=120,grace_ns=30_000_000_000)
    entry=json.loads((run/'started.json').read_text())['entry']
    registry=ProcessRegistry.load(args.process_registry) if args.process_registry else None
    bridge_entry=registry.by_role('bridge')[0] if registry else None
    router_entry=registry.by_role('zenoh_router')[0] if registry else None
    container_baseline=(json.loads(args.container_baseline.read_text())['process_ids']
                        if args.container_baseline is not None else None)
    while True:
        snap=sample(nvml,container,baseline_ids,remote,args.sim_source,container_baseline)
        if snap['simulator_tree']:
            atomic(run/'simulator-processes.json',{'run_id':args.run_id,
                'processes':snap['simulator_tree'],'sample_monotonic_ns':snap['monotonic_ns']})
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
        table=proc_table();host=table.get(args.sim_host_pid)
        progress=remote_progress(args.remote_progress)
        if progress is not None:
            step=progress['last_completed_step']
            if last_step!=step:last_step=step;last_progress=time.monotonic()
            if last_recorded_step!=step:
                append(run/'timing-progress-timeseries.jsonl',heartbeat_record(progress,
                    target_step=args.target_step,sample_monotonic_ns=snap['monotonic_ns'],
                    host_pid=args.sim_host_pid,host_start_ticks=host['start_ticks'] if host else None))
                last_recorded_step=step
            if (time.monotonic_ns()-progress['monotonic_ns'])/1e9>safety['infrastructure_stop']['heartbeat_stale_over_sec']:
                stop=stop or 'simulator_heartbeat_stale'
        stalled=machine.state!='FINALIZING' and time.monotonic()-last_progress>safety['infrastructure_stop']['step_stall_over_sec']
        if stop is None and disk_bytes(root)>safety['infrastructure_stop']['storage_over_bytes']:
            stop='storage_cap'
        if completion is None and last_step is not None and last_step>=args.target_step+120:
            completion=remote_completion(remote,entry.get('map_path_file_sha256') or
                                         read_complete_json(run/'started.json')['input_sha256']['map_path'])
        suffix_started=last_step is not None and last_step>args.target_step+1
        bridge_alive=(registered_alive(bridge_entry) if bridge_entry else False)
        router_alive=(registered_alive(router_entry) if router_entry else False)
        journal=run/'follower-logs/bridge-events.jsonl'
        journal_text=journal.read_text(errors='replace') if journal.exists() else ''
        follower_spawns=[json.loads(line) for line in journal_text.splitlines()
            if '"event":"follower_spawn"' in line]
        follower_alive=(len(follower_spawns)==8 and all(registered_alive(x['process'])
            for x in follower_spawns))
        protocol_error=any(x in journal_text for x in ('"event":"tcp_error"',
            '"event":"executor_exception"','"event":"waiter_error"'))
        observed={'now_ns':snap['monotonic_ns'],'simulator_tree':bool(snap['simulator_tree']),
            'host_alive':proc.exists(),'host_start_ticks':host['start_ticks'] if host else None,
            'step_count':max(0,(last_step or 0)-args.target_step),
            'completion':completion,'resource_stop':stop,'heartbeat_stale':stalled,
            'protocol_error':protocol_error,'router_alive':router_alive,
            'bridge_alive':bridge_alive,'followers_alive':follower_alive if suffix_started else True,
            'nonfinite':False,'final':None}
        machine.observe(observed)
        heartbeat={'run_id':args.run_id,'wall_ns':snap['wall_ns'],'monotonic_ns':snap['monotonic_ns'],
            'global_step':last_step,'target_step':args.target_step,
            'target_episode':args.target_episode,'simulator_tree_pid_count':len(snap['simulator_tree']),
            'resource':gate,'simulator_rss_bytes':snap['simulator_tree_rss_bytes'],
            'simulator_pss_bytes':snap['simulator_tree_pss_bytes'],
            'simulator_fd_count':snap['simulator_tree_fd_count'],
            'simulator_thread_count':snap['simulator_tree_thread_count'],
            'output_bytes':disk_bytes(root),'stop_reason':machine.reason,
            'lifecycle_state':machine.state,'lifecycle_transitions':machine.transitions}
        atomic(run/'timing-heartbeat.json',heartbeat)
        if machine.state=='FAIL':
            atomic(run/'resource-stop.failure.json',{'reason':machine.reason,'snapshot':snap,
                'gpu_timeseries_sha256':sha(run/'gpu-process-timeseries.jsonl')})
            reason=machine.reason;break
        if (run/'monitor.stop').exists():
            if machine.state!='FINALIZING':
                reason='control_cleanup_failure: monitor stopped before finalizing';break
            atomic(run/'runtime-monitor-result.json',{'status':'pass','transitions':machine.transitions,
                'sample_count':sample_count,'resource_limits':limits(safety),
                'last_sample_monotonic_ns':snap['monotonic_ns']})
            atomic(run/'timing-monitor.done.json',{'status':'stopped','lifecycle':'FINALIZING',
                'sample_count':sample_count,'runtime_monitor':'pass',
                'last_sample_monotonic_ns':snap['monotonic_ns']})
            return
        time.sleep(.5)
    atomic(run/'timing-monitor.done.json',{'status':'failure','reason':reason,'sample_count':sample_count})
    raise RuntimeError(reason)


if __name__=='__main__':main()
