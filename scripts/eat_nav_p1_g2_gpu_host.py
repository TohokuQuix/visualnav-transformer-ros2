"""One-shot G2 timing host with G1 ownership, NVML and finalization gates."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import shutil
import signal
import socket
import subprocess
import time
from eat_nav_g0e_gpu_probe import Nvml, proc_info, proc_table
from eat_nav_p1_gpu_attribution import capture_live
from eat_nav_p1_longhorizon_gpu import capture_sample
from eat_nav_p1_owner_host import initial_graphics_identities
from eat_nav_p1_g1_lifecycle import (
    G1Lifecycle, diagnose_owner_resolution, residual_identities,
    seal_after_writers, verify_seal,
)
from eat_nav_ros_subprocess_env import build_ros_host_env, env_audit
from visualnav_transformer.eat_nav.p1_longhorizon_protocol import SameEpisodeGate

CONTAINER = 'isaac-lab-onix'
FOLLOWER=pathlib.Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node')
PARAMS=pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')
ROUTER=pathlib.Path('/home/kojima/eatnav-router-logs/eatnav-p1-paired-noise-20261010-v7')
ROOT = None
REMOTE = None
BASE = ':'.join(('/tmp/eatnav-train/src', '/tmp/eatnav-extended/isaac-source',
                 '/workspace/isaaclab/source/isaaclab', '/workspace/isaaclab/source/isaaclab_rl',
                 '/workspace/isaaclab/source/isaaclab_tasks',
                 '/workspace/isaaclab_onix/source/onix_isaac_lab'))


def build_wrapper_command(*, remote: str, container_id: str, token: str,
                          source_sha: str, config_sha: str) -> list[str]:
    """Keep the simulator paused until its exact host PID is monitored."""
    return ['docker','exec','-e',f'PYTHONPATH={remote}/src:{BASE}',CONTAINER,
            'timeout','600s','/isaac-sim/python.sh',f'{remote}/owner-wrapper.py',
            '--coord',f'{remote}/coord','--run-token',token,
            '--container-id',container_id,'--source',f'{remote}/source.py',
            '--source-sha',source_sha,'--config',f'{remote}/input/freeze.json',
            '--config-sha',config_sha,'--output',f'{remote}/output','--mode','smoke',
            '--','--headless','--input',f'{remote}/input',
            '--output',f'{remote}/output','--anchor-id','first-differential',
            '--batch-index','0']


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = pathlib.Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def command(args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


GPU_UUID = 'GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26'
GPU_CONTRACT = {'required_gpu_uuid': GPU_UUID,
                'required_total_memory_mib_min': 16000,
                'gpu_device_total_hard_cap_mib': 7680,
                'bracket_nonatomic_sampling_gap_hard_cap_mib': 256,
                'absolute_attribution_gap_hard_cap_mib': 256,
                'unresolved_gpu_bytes_hard_cap': 0}


def process_tick(pid):
    try:return int(pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except FileNotFoundError:return None


def owned_group_members(pgid):
    members=[]
    for entry in pathlib.Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0]!='Z':members.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError,ValueError):pass
    return sorted(members)


def stop_owned_bridge(process,tick):
    if process_tick(process.pid)!=tick:raise ValueError('bridge PID/start tick changed')
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if process_tick(process.pid)!=tick:raise ValueError('bridge PID/start tick changed during cleanup')
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    remaining=owned_group_members(process.pid)
    if remaining:raise ValueError('owned bridge/follower residual:'+repr(remaining))
    return remaining


def router_audit():
    launcher_pid=int((ROUTER/'router-launch.pid').read_text().strip())
    pid=int((ROUTER/'router.pid').read_text().strip())
    def identity(target):
        fields=pathlib.Path(f'/proc/{target}/stat').read_text().rsplit(')',1)[1].split()
        command=pathlib.Path(f'/proc/{target}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        if fields[0] in ('Z','X') or not command:raise ValueError('router process not alive')
        return {'pid':target,'ppid':int(fields[1]),'start_tick':int(fields[19]),
                'state':fields[0],'cmdline':command}
    launcher=identity(launcher_pid);router=identity(pid)
    if ('ros2 run rmw_zenoh_cpp rmw_zenohd' not in launcher['cmdline']
            or '/rmw_zenohd' not in router['cmdline'] or router['ppid']!=launcher_pid):
        raise ValueError('external router process identity mismatch')
    stdout=ROUTER/'router.stdout.log';stderr=ROUTER/'router.stderr.log'
    data=stdout.read_text()+'\n'+stderr.read_text()
    zid=re.search(r'Started Zenoh router with id ([0-9a-f]+)',data)
    endpoints=re.findall(r'Zenoh can be reached at: tcp/([^\s]+):7447',data)
    if zid is None or not endpoints:raise ValueError('router log ZID or endpoint missing')
    reachable=[]
    for endpoint in endpoints:
        try:
            with socket.create_connection((endpoint,7447),timeout=2):reachable.append(endpoint)
        except OSError:pass
    if not reachable:raise ValueError('external router unreachable')
    return {'pid':pid,'start_tick':router['start_tick'],'launcher_pid':launcher_pid,
            'launcher_start_tick':launcher['start_tick'],'zid':zid.group(1),
            'endpoints':endpoints,'reachable_endpoints':reachable,
            'stdout_sha256':digest(stdout),'stderr_sha256':digest(stderr)}


def host_available():
    for line in pathlib.Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing')


def anon_shmem():
    raw = command(['docker', 'exec', CONTAINER, 'cat', '/sys/fs/cgroup/memory.stat'])
    stats = {row.split()[0]: int(row.split()[1]) for row in raw.splitlines()}
    return stats['anon'] + stats['shmem']


def root_bytes():
    return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())


def resource_row(started, gpu):
    # Keep the bracket evidence small enough to stay inside the frozen storage cap.
    audit = {key: gpu[key] for key in ('pass','failures','gpu_uuid','device_before_bytes',
             'device_after_bytes','device_total_bytes','processes','owned_gpu_pids',
             'bracket_gap_bytes','process_sum_bytes','resolved_owned','wall_ns',
             'sample_finished_monotonic_ns')}
    return {'elapsed_s': time.monotonic() - started,
            'gpu_device_total_mib': max(gpu['device_before_bytes'],gpu['device_after_bytes'])/1048576,
            'gpu_attribution': audit,
            'host_available_bytes': host_available(), 'container_anon_plus_shmem_bytes': anon_shmem(),
            'persistent_root_bytes': root_bytes()}


def sample(started, nvml, owned, baseline, container_id):
    gpu = capture_sample(nvml, owned, baseline, GPU_CONTRACT,
                         container_id=container_id, require_owned_gpu=False)
    return gpu, resource_row(started, gpu)


def validate_sample(row):
    if not row['gpu_attribution']['pass']: return 'gpu_attribution:' + ','.join(row['gpu_attribution']['failures'])
    if row['gpu_device_total_mib'] > 7680: return 'gpu_device_total_cap'
    if row['host_available_bytes'] < 10095421850: return 'host_available_floor'
    if row['container_anon_plus_shmem_bytes'] > 6442450944: return 'container_anon_shmem_cap'
    if row['persistent_root_bytes'] > 1000000000: return 'persistent_storage_cap'
    return None


def append_jsonl(stream, value):
    stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
    stream.flush(); os.fsync(stream.fileno())


def exact_owner_alive(owner):
    if owner is None:return False
    row=proc_info(owner['host_pid'])
    return bool(row and row.get('start_ticks')==owner['start_ticks'])


def stop_exact_owner(owner, evidence):
    if not exact_owner_alive(owner):return
    os.kill(owner['host_pid'],signal.SIGTERM)
    evidence.append({'signal':'SIGTERM','host_pid':owner['host_pid'],
                     'start_ticks':owner['start_ticks']})
    for _ in range(50):
        if not exact_owner_alive(owner):return
        time.sleep(.1)
    if exact_owner_alive(owner):
        os.kill(owner['host_pid'],signal.SIGKILL)
        evidence.append({'signal':'SIGKILL','host_pid':owner['host_pid'],
                         'start_ticks':owner['start_ticks']})


def diagnostic_params(production: pathlib.Path, destination: pathlib.Path, expected: str):
    raw=production.read_text()
    anchor='debug_control_log_period_sec: 0.5'
    if raw.count(anchor)!=1 or raw.count('debug_control_log: true')!=1:
        raise ValueError('production diagnostic parameter anchor changed')
    derived=raw.replace(anchor,'debug_control_log_period_sec: 0.1').encode()
    if hashlib.sha256(derived).hexdigest()!=expected:
        raise ValueError('frozen diagnostic follower params unavailable')
    destination.write_bytes(derived)
    return digest(destination)


def _finite(value):
    if isinstance(value,float):return math.isfinite(value)
    if isinstance(value,list):return all(_finite(item) for item in value)
    if isinstance(value,dict):return all(_finite(item) for item in value.values())
    return True


def validate_g2_output(root: pathlib.Path, remote: str, g2_contract: dict) -> dict:
    output=root/'output'
    marker=json.loads((output/'complete.json').read_text())
    summary=json.loads((output/'pilot-summary.json').read_text())
    if (marker.get('Path_publications')!=1 or marker.get('settling_control_steps')!=20
            or marker.get('control_steps')!=600 or summary.get('recorded_steps')!=[600]*4):
        raise ValueError('G2 completion marker or four-env count invalid')
    for key,name in (('initial_state_sha256','initial-state.json'),
                     ('terrain_sha256','terrain.json'),('prepath_gate_sha256','prepath-gate.json'),
                     ('path_publication_sha256','path-publication.json'),
                     ('pilot_summary_sha256','pilot-summary.json'),
                     ('contact_body_names_sha256','contact-body-names.json')):
        if digest(output/name)!=marker[key]:raise ValueError('G2 referenced SHA mismatch:'+name)
    traces=[]
    for index in range(4):
        path=output/f'trace-slot-{index}.jsonl'
        if digest(path)!=marker['trace_sha256'][index]:
            raise ValueError('G2 trace SHA mismatch')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=600 or [row['control_step'] for row in rows]!=list(range(600)):
            raise ValueError('G2 trace incomplete or noncontiguous')
        gate=SameEpisodeGate(600,initial_episode_length=rows[0]['timing_protocol']['episode_length_before'],
                             episode_cap_steps=800)
        for row in rows:
            if not _finite(row):raise ValueError('G2 nonfinite trace value')
            if row.get('terminal') is not None:raise ValueError('G2 terminal trace')
            gate.accept(row['timing_protocol'])
        if not gate.complete:raise ValueError('G2 same-episode gate incomplete')
        traces.append(rows)
    for name in ('complete.json',*[f'trace-slot-{i}.jsonl' for i in range(4)]):
        result=subprocess.run(['docker','exec',CONTAINER,'sha256sum',f'{remote}/output/{name}'],
                              capture_output=True,text=True,check=True)
        if result.stdout.split()[0]!=digest(output/name):
            raise ValueError('G2 host/container SHA mismatch:'+name)
    stamps=[row['timing_protocol']['stamp_ns'] for row in traces[0]]
    for rows in traces[1:]:
        if [row['timing_protocol']['stamp_ns'] for row in rows]!=stamps:
            raise ValueError('G2 env logical clocks differ')
    return {'trace_steps_per_env':[600]*4,'logical_first_ns':stamps[0],
            'logical_last_ns':stamps[-1],
            'logical_interval_ns':50_000_000,
            'max_episode_length_steps':800,
            'path_publications':1,
            'trace_sha256':marker['trace_sha256']}


def main():
    global ROOT, REMOTE
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=pathlib.Path,required=True)
    parser.add_argument('--remote',required=True)
    args=parser.parse_args()
    ROOT=args.root.resolve();REMOTE=args.remote
    if ROOT.name!='g2-attempt-0' or (ROOT/'output').exists() or (ROOT/'host-seal.json').exists():
        raise ValueError('G2 requires a fresh dedicated persistent root')
    freeze=json.loads((ROOT/'input/freeze.json').read_text())
    g2=json.loads((ROOT/'input/g2-contract.json').read_text())
    cfg=json.loads((ROOT/'input/statistical-config.json').read_text())
    if (g2['schema']!='eatnav_p1_longhorizon_g2_diagnostic_v1'
            or g2['control_steps']!=600 or g2['episode_length_s']!=40.0
            or g2['control_dt_s']!=0.05 or g2['path']!='frozen_0.75m_0deg_direct'
            or digest(ROOT/'source.py')!=g2['g2_simulator_source_sha256']):
        raise ValueError('frozen G2 source/contract mismatch')
    for name,key in (('assignment.json','assignment_sha256'),
                     ('statistical-config.json','config_sha256'),
                     ('source-contract.json','source_contract_sha256')):
        if digest(ROOT/'input'/name)!=freeze[key]:raise ValueError('frozen P1 input SHA mismatch:'+name)
    if digest(FOLLOWER)!=cfg['follower_sha256']:
        raise ValueError('production follower binary SHA mismatch')
    if digest(pathlib.Path(__file__).with_name('eat_nav_p1_paired_bridge.py'))!=cfg['bridge_sha256']:
        raise ValueError('frozen diagnostic bridge SHA mismatch')
    params=ROOT/'follower-debug.yaml'
    diagnostic_params(PARAMS,params,cfg['follower_params_sha256'])
    router_before=router_audit()
    atomic(ROOT/'router-before.json',router_before)
    inspect=json.loads(command(['docker','inspect',CONTAINER]))[0]
    if inspect['State']['Status']!='running':raise RuntimeError('container not running')
    if subprocess.run(['docker','exec',CONTAINER,'test','-d',REMOTE],capture_output=True).returncode:
        raise RuntimeError('persistent root not visible inside container')
    checkpoint=subprocess.run(['docker','exec',CONTAINER,'sha256sum',cfg['rl_checkpoint']],
                              capture_output=True,text=True,check=True)
    if checkpoint.stdout.split()[0]!=cfg['rl_checkpoint_sha256']:
        raise ValueError('frozen RL checkpoint SHA mismatch')
    nvml=Nvml();devices,rows=nvml.sample();table=proc_table()
    baseline_identities=initial_graphics_identities(rows,table)
    baseline_gpu=capture_live(nvml,{},baseline_identities,GPU_CONTRACT,
        container_id=inspect['Id'],expected_simulator_alive=False,device_cap_mib=1024)
    if not baseline_gpu['pass'] or not baseline_identities:
        raise RuntimeError('G2 prelaunch GPU attribution invalid:'+','.join(baseline_gpu['failures']))
    if devices[0]['uuid']!=GPU_UUID or devices[0]['used_bytes']>1024*1048576:
        raise RuntimeError('G2 prelaunch GPU identity/cap')
    if shutil.disk_usage(ROOT).free<2_000_000_000:
        raise RuntimeError('G2 host disk free floor')
    atomic(ROOT/'g2-prelaunch.json',{'container_id':inspect['Id'],'image_id':inspect['Image'],
        'source_sha256':digest(ROOT/'source.py'),'g2_contract_sha256':digest(ROOT/'input/g2-contract.json'),
        'host_launcher_sha256':digest(__file__),
        'frozen_params_sha256':digest(params),'follower_binary_sha256':digest(FOLLOWER),
        'bridge_sha256':cfg['bridge_sha256'],'checkpoint_sha256':checkpoint.stdout.split()[0],
        'gpu_uuid':GPU_UUID,'baseline_attribution':{key:baseline_gpu[key] for key in
            ('processes','bracket_gap_bytes','owned_gpu_pids','device_before_bytes','device_after_bytes')}})
    wrapper=pathlib.Path(__file__).with_name('eat_nav_p1_owner_wrapper.py')
    shutil.copy2(wrapper,ROOT/'owner-wrapper.py')
    coord=ROOT/'coord';coord.mkdir()
    token=secrets.token_hex(32)
    source_sha=digest(ROOT/'source.py');config_sha=digest(ROOT/'input/freeze.json')
    atomic(coord/'launch-intent.json',{'schema':'eatnav_p1_g2_start_v1','run_token':token,
        'container_id':inspect['Id'],'output_root':f'{REMOTE}/output',
        'source_sha256':source_sha,'config_sha256':config_sha,
        'wrapper_sha256':digest(ROOT/'owner-wrapper.py')})
    cmd=build_wrapper_command(remote=REMOTE,container_id=inspect['Id'],token=token,
                              source_sha=source_sha,config_sha=config_sha)
    atomic(ROOT/'g2-command.json',{'argv':cmd,'wrapper_sha256':digest(ROOT/'owner-wrapper.py')})
    ros_env=build_ros_host_env(repo=pathlib.Path(__file__).resolve().parents[1])
    ros_env['RMW_IMPLEMENTATION']='rmw_zenoh_cpp'
    atomic(ROOT/'bridge-environment.json',env_audit(ros_env,'/usr/bin/python3'))
    bridge_cmd=['/usr/bin/python3',str(pathlib.Path(__file__).with_name('eat_nav_p1_paired_bridge.py')),
        '--branches','5','--follower',str(FOLLOWER),'--params',str(params),
        '--diagnostic-dir',str(ROOT/'follower-logs')]
    atomic(ROOT/'bridge-command.json',{'argv':bridge_cmd})
    gate=G1Lifecycle(token=token,container_id=inspect['Id'],source_sha=source_sha,
        config_sha=config_sha,output_root=f'{REMOTE}/output',gpu_uuid=GPU_UUID)
    started=time.monotonic();failure=None;owner=None;client=None;bridge=None;bridge_tick=None
    native_exit=None;brackets=0;cleanup_signals=[];owned_identities={};bridge_residual=[]
    with (ROOT/'runner.log').open('xb') as log, \
         (ROOT/'bridge.log').open('xb') as bridge_log, \
         (ROOT/'resource-series.jsonl').open('x') as resources, \
         (ROOT/'owner-polls.jsonl').open('x') as polls:
        try:
            bridge=subprocess.Popen(bridge_cmd,env=ros_env,stdout=bridge_log,
                                    stderr=subprocess.STDOUT,start_new_session=True)
            bridge_tick=process_tick(bridge.pid)
            if not isinstance(bridge_tick,int):raise RuntimeError('bridge PID/start tick unresolved')
            atomic(ROOT/'bridge-owner.json',{'pid':bridge.pid,'start_tick':bridge_tick})
            deadline=time.monotonic()+20
            while True:
                if bridge.poll() is not None:raise RuntimeError('bridge exited before listening')
                try:
                    with socket.create_connection(('127.0.0.1',47819),timeout=.2):break
                except OSError:
                    if time.monotonic()>deadline:raise TimeoutError('bridge listener timeout')
                    time.sleep(.1)
            client=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+45
            while owner is None:
                if bridge.poll() is not None:raise RuntimeError('bridge exited before simulator authorization')
                if client.poll() is not None:raise RuntimeError('wrapper exited before owner readiness')
                if time.monotonic()>deadline:raise TimeoutError('owner mapping timeout')
                ready_path=coord/'owner-ready.json'
                if not ready_path.is_file():time.sleep(.05);continue
                ready=json.loads(ready_path.read_text());table=proc_table()
                diagnostic=diagnose_owner_resolution(ready,table,inspect['Id'])
                candidates=[{'host_pid':p['host_pid'],'container_pid':p['container_pid'],
                    'start_ticks':p['start_ticks'],'namespace_pid_chain':p['namespace_pid_chain'],
                    'cgroup':p['cgroup'],'command_line':p['command_line']}
                    for p in table.values() if p.get('container_pid')==ready.get('container_pid')
                    or token in p.get('command_line','').split()]
                append_jsonl(polls,{'wall_ns':time.time_ns(),'diagnostic':diagnostic,
                                    'ready':ready,'candidates':candidates})
                try:owner=gate.register_ready(ready,table)
                except ValueError as exc:
                    if str(exc)!='owner_pid_mapping_missing_or_ambiguous':raise
                    time.sleep(.1)
            atomic(coord/'registered-owner.json',owner)
            gpu=capture_live(nvml,{'simulator':owner},baseline_identities,GPU_CONTRACT,
                container_id=inspect['Id'],expected_simulator_alive=True)
            row=resource_row(started,gpu);append_jsonl(resources,row);brackets+=1
            if violation:=validate_sample(row):raise RuntimeError('preauthorization_'+violation)
            gate.pre_authorization(gpu)
            auth=gate.authorize();auth.update({'bracket_monotonic_ns':gpu['monotonic_ns'],
                                               'authorization_wall_ns':time.time_ns()})
            atomic(coord/'owner-authorized.json',auth)
            while client.poll() is None:
                if bridge.poll() is not None:raise RuntimeError('bridge exited during G2')
                gpu,row=sample(started,nvml,{'simulator':owner},baseline_identities,inspect['Id'])
                append_jsonl(resources,row);brackets+=1
                if violation:=validate_sample(row):raise RuntimeError(violation)
                gate.observe(gpu)
                for pid in gpu.get('owned_process_tree_host_pids',[]):
                    proc=gpu['process_table_before'].get(pid)
                    if proc:owned_identities[pid]=proc['start_ticks']
                if row['elapsed_s']>600:raise TimeoutError('G2 process wall cap')
                heartbeat=ROOT/'output/heartbeat.json'
                if heartbeat.is_file() and time.time()-heartbeat.stat().st_mtime>90:
                    raise TimeoutError('G2 heartbeat stale')
                time.sleep(.5)
            native_exit=client.wait(timeout=30)
        except BaseException as exc:
            failure=type(exc).__name__+':'+str(exc)
            if owner is not None:stop_exact_owner(owner,cleanup_signals)
            if client is not None and client.poll() is None:client.terminate()
            if client is not None:
                try:native_exit=client.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    client.kill();native_exit=client.wait(timeout=10)
        finally:
            try:
                if bridge is not None:
                    if bridge.poll() is None:
                        bridge_residual=stop_owned_bridge(bridge,bridge_tick)
                    else:
                        bridge_residual=owned_group_members(bridge.pid)
                        if bridge_residual:raise RuntimeError('bridge group residual')
            except BaseException as exc:
                failure=failure or (type(exc).__name__+':'+str(exc))
                bridge_residual=bridge_residual or ['cleanup_unverified']
            for stream in (polls,resources,log,bridge_log):
                stream.flush();os.fsync(stream.fileno())
    try:
        router_after=router_audit()
        atomic(ROOT/'router-after.json',router_after)
        if any(router_after[k]!=router_before[k] for k in
               ('pid','start_tick','launcher_pid','launcher_start_tick','zid','endpoints')):
            raise ValueError('external router identity changed')
    except BaseException as exc:
        failure=failure or (type(exc).__name__+':'+str(exc))
    final_table=proc_table()
    residual=residual_identities(final_table,owner=owner,known=owned_identities,
                                 token=token,container_id=inspect['Id'])
    writer_state={'simulator':not residual and (client is None or client.poll() is not None),
        'heartbeat':not residual,'resource':True,'log':not bridge_residual}
    validation=None
    if failure is None:
        try:
            validation=validate_g2_output(ROOT,REMOTE,g2)
            gate.complete(native_exit=native_exit,owner_alive=bool(residual),
                          writers_closed=all(writer_state.values()))
        except BaseException as exc:failure=type(exc).__name__+':'+str(exc)
    atomic(ROOT/'cleanup-evidence.json',{'signals':cleanup_signals,'owned_residual':residual,
        'bridge_residual':bridge_residual,'writer_state':writer_state,'native_exit':native_exit})
    atomic(ROOT/'g2-host-status.json',{'failure':failure,'native_exit':native_exit,
        'elapsed_s':time.monotonic()-started,'gpu_bracket_count':brackets,
        'owned_gpu_bracket_count':gate.live_gpu_samples,'owned_simulator':owner,
        'owned_simulator_residual':bool(residual),'lifecycle_phase':gate.phase,
        'validation':validation})
    if all(writer_state.values()):
        stage='G2_protocol_valid' if failure is None else 'G2_failed_or_inconclusive'
        seal_after_writers(ROOT,writer_state=writer_state,stage=stage,reason=failure)
        time.sleep(.1);verify_seal(ROOT)
    else:
        atomic(ROOT/'unsealed-evidence.json',{'failure':failure,'writer_state':writer_state})
        raise RuntimeError('G2 cannot seal while owned writer remains')
    if failure is not None:raise RuntimeError('G2 failed or inconclusive: '+failure)
    print('G2_PASS',str(ROOT/'host-seal.json'))


if __name__=='__main__':
    main()
