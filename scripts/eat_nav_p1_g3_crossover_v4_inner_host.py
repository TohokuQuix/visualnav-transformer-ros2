"""P1 Path crossover host with G1 ownership and pre-Path approval gate."""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import secrets
import stat
import shutil
import signal
import socket
import subprocess
import time
from eat_nav_g0e_gpu_probe import Nvml, proc_info, proc_table
from eat_nav_p1_gpu_attribution import capture_live
from eat_nav_p1_longhorizon_gpu import capture_sample
from eat_nav_p1_g3_runtime_evidence_v4 import classify_gpu_bracket
from eat_nav_p1_owner_host import initial_graphics_identities
from eat_nav_p1_g1_lifecycle import (
    G1Lifecycle, diagnose_owner_resolution, residual_identities,
    seal_after_writers, verify_seal,
)
from eat_nav_ros_subprocess_env import build_ros_host_env, env_audit
from visualnav_transformer.eat_nav.p1_longhorizon_protocol import SameEpisodeGate, G3Exposure
from visualnav_transformer.eat_nav.p1_g3_shared_dem import verify_shared_artifacts
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import validate_four_follower_final
from visualnav_transformer.eat_nav.p1_g3_material_gate_v2 import SCHEMA, MATERIAL_SCHEMA
from visualnav_transformer.eat_nav.p1_g3_crossover_runtime import compare_prepath
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment

CONTAINER = 'isaac-lab-onix'
FOLLOWER=pathlib.Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node')
PARAMS=pathlib.Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')
ROUTER=None  # Set from the read-only runtime router sidecar at launch.
ROOT = None
REMOTE = None
STAGE_ROOT = None
BASE = ':'.join(('/tmp/eatnav-train/src', '/tmp/eatnav-extended/isaac-source',
                 '/workspace/isaaclab/source/isaaclab', '/workspace/isaaclab/source/isaaclab_rl',
                 '/workspace/isaaclab/source/isaaclab_tasks',
                 '/workspace/isaaclab_onix/source/onix_isaac_lab'))


def build_wrapper_command(*, remote: str, container_id: str, token: str,
                          source_sha: str, config_sha: str, anchor_id: str, batch_index: int, execution_index: int,
                          dem_diagnostic: bool = False) -> list[str]:
    """Keep the simulator paused until its exact host PID is monitored."""
    command=['docker','exec','-e',f'PYTHONPATH={remote}/src:{BASE}',CONTAINER,
            'timeout','600s','/isaac-sim/python.sh',f'{remote}/owner-wrapper.py',
            '--coord',f'{remote}/coord','--run-token',token,
            '--container-id',container_id,'--source',f'{remote}/source.py',
            '--source-sha',source_sha,'--config',f'{remote}/input/freeze.json',
            '--config-sha',config_sha,'--output',f'{remote}/output','--mode','smoke',
            '--','--headless','--input',f'{remote}/input',
            '--output',f'{remote}/output','--anchor-id',anchor_id,
            '--batch-index',str(batch_index),'--execution-index',str(execution_index)]
    if dem_diagnostic: command.append('--gate1-probe')
    return command


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


def router_log_bytes(path: pathlib.Path, launcher_pid: int, fd: int,
                     *, proc_root: pathlib.Path=pathlib.Path('/proc')) -> bytes:
    """Preserve access to externally owned logs after a pathname unlink."""
    try:
        if stat.S_ISREG(path.stat().st_mode):
            return path.read_bytes()
    except FileNotFoundError:
        pass
    fd_path=proc_root/str(launcher_pid)/'fd'/str(fd)
    if not stat.S_ISREG(fd_path.stat().st_mode):
        raise ValueError('router log fd is not a regular file')
    return fd_path.read_bytes()


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
    stdout=router_log_bytes(ROUTER/'router.stdout.log',launcher_pid,1)
    stderr=router_log_bytes(ROUTER/'router.stderr.log',launcher_pid,2)
    data=stdout.decode(errors='replace')+'\n'+stderr.decode(errors='replace')
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
            'stdout_sha256':hashlib.sha256(stdout).hexdigest(),
            'stderr_sha256':hashlib.sha256(stderr).hexdigest()}


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
            'sample_started_wall_ns': gpu['wall_ns'],
            'sample_finished_wall_ns': gpu['process_table_after_wall_ns'],
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


def approve_prepath(root: pathlib.Path, anchor_id: str, batch_index: int,
                    execution_index: int) -> dict:
    """Authorize exactly one pre-Path snapshot after checking earlier sealed period."""
    output = root/'output'
    ready = json.loads((output/'prepath-ready.json').read_text())
    if ready.get('Path_publications') != 0 or ready.get('physics_steps_after_clone') != 0 or \
            (output/'path-publication.json').exists():
        raise ValueError('Path/physics advanced before cross-period approval')
    snapshot_path=output/'prepath-crossover-snapshot.json'
    material_path=output/'live-material.json'
    if (digest(snapshot_path)!=ready.get('snapshot_sha256') or
            digest(material_path)!=ready.get('live_material_sha256')):
        raise ValueError('pre-Path snapshot/material SHA mismatch')
    actor_gate=json.loads((output/'direct-actor-gate.json').read_text())
    if not json.loads((output/'prepath-gate.json').read_text())['pass'] or \
            actor_gate['max_abs'] > actor_gate['limit']:
        raise ValueError('within-process pre-Path gate failed')
    current=json.loads(snapshot_path.read_text())
    material=json.loads(material_path.read_text())
    if (material.get('schema')!=MATERIAL_SCHEMA or not material.get('pass') or
            not material.get('terrain_collision_bindings') or
            not material.get('robot_collision_bindings') or
            material['terrain_declared_sha256']!=current['terrain_material_sha256'] or
            material['robot_effective_physx']['canonical_sha256']!=current['robot_material_sha256']):
        raise ValueError('live material audit incomplete')
    if execution_index==0:
        baseline=current
        baseline_material=material
        baseline_sha=digest(snapshot_path)
    else:
        prior=root.parent/f'batch-{anchor_id}-{batch_index}-period-{execution_index-1}'
        if verify_seal(prior).get('stage')!='G3_crossover_batch_protocol_valid':
            raise ValueError('previous crossover period not sealed protocol-valid')
        first=root.parent/f'batch-{anchor_id}-{batch_index}-period-0'
        seal=verify_seal(first)
        if seal.get('stage')!='G3_crossover_batch_protocol_valid':
            raise ValueError('first period not sealed protocol-valid')
        baseline_path=first/'output/prepath-crossover-snapshot.json'
        baseline=json.loads(baseline_path.read_text())
        baseline_material=json.loads((first/'output/live-material.json').read_text())
        baseline_sha=digest(baseline_path)
    gate=compare_prepath(baseline,current)
    material_equal=(
        baseline_material['terrain_collision_bindings']==material['terrain_collision_bindings'] and
        baseline_material['robot_collision_bindings']==material['robot_collision_bindings'] and
        baseline_material['robot_effective_physx']['canonical_sha256']==
        material['robot_effective_physx']['canonical_sha256'])
    gate.update({'baseline_snapshot_sha256':baseline_sha,
                 'current_snapshot_sha256':digest(snapshot_path),
                 'live_material_equal':material_equal,
                 'baseline_period_execution_index':0,
                 'current_execution_index':execution_index,
                 'Path_publications':0})
    gate['pass']=gate['pass'] and material_equal
    atomic(output/'cross-period-prepath-gate.json',gate)
    if not gate['pass']:
        atomic(output/'prepath-rejection.json',{'reason':'cross_period_prepath_gate_failed',
                'gate_sha256':digest(output/'cross-period-prepath-gate.json')})
        raise ValueError('cross-period pre-Path gate failed')
    approval={'pass':True,'snapshot_sha256':digest(snapshot_path),
              'gate_sha256':digest(output/'cross-period-prepath-gate.json'),
              'material_sha256':digest(material_path),'Path_publications':0}
    atomic(output/'prepath-approval.json',approval)
    return approval


def validate_period_assignment(assigned_rows: list[dict], frozen_rows: list[dict]) -> None:
    """Check one period against its frozen row without demanding all three roles."""
    if (len(assigned_rows)!=4 or
            [x['slot'] for x in assigned_rows]!=list(range(4)) or
            [x['candidate_slot'] for x in assigned_rows]!=[f'crossover_{i}' for i in range(4)] or
            {x['condition'] for x in assigned_rows}!={'A'} or
            any(x['path_role'] not in ('direct','left_20','right_20') for x in assigned_rows) or
            assigned_rows!=sorted(frozen_rows,key=lambda x:x['slot'])):
        raise ValueError('crossover four-branch assignment invalid')


def simulator_artifact_failure(output: pathlib.Path, native_exit: int | None,
                               *, diagnostic: bool = False) -> str | None:
    """A simulator diagnostic marker is a protocol failure regardless of shell exit."""
    marker=output/'diagnostic-error.json'
    if marker.is_file():
        try:
            error=json.loads(marker.read_text())
            return 'simulator_diagnostic_error:'+str(error['type'])+':'+str(error['message'])
        except (ValueError, KeyError, TypeError) as exc:
            return 'simulator_diagnostic_error_marker_invalid:'+type(exc).__name__
    if native_exit is not None and native_exit != 0:
        return 'simulator_native_exit:'+str(native_exit)
    marker_name = 'gate1-probe-complete.json' if diagnostic else 'complete.json'
    if native_exit is not None and not (output/marker_name).is_file():
        return 'simulator_completion_marker_missing'
    return None


def classify_monitor_violation(violation: str, client_exit: int | None,
                               output: pathlib.Path, *, diagnostic: bool = False) -> str | None:
    """Only a verified terminal marker can resolve an end-of-process PID race."""
    if client_exit is None or 'simulator_disappeared_in_bracket' not in violation:
        return violation
    artifact=simulator_artifact_failure(output,client_exit,diagnostic=diagnostic)
    return artifact if artifact is not None else None


def terminal_completion(output: pathlib.Path, native_exit: int | None,
                        owner: dict | None, root: pathlib.Path) -> dict | None:
    """Validate a pre-close completion marker before interpreting an exit bracket."""
    marker_path=output/'complete.json'
    if not marker_path.is_file() or owner is None:
        return None
    try:
        marker=json.loads(marker_path.read_text())
        registered=json.loads((root/'coord/registered-owner.json').read_text())
        valid=(registered==owner and marker.get('Path_publications')==1 and
               isinstance(marker.get('control_steps'),int) and
               41<=marker['control_steps']<=640 and
               len(marker.get('trace_sha256',[]))==4 and
               all(digest(output/f'trace-slot-{i}.jsonl')==expected for i,expected in
                   enumerate(marker['trace_sha256'])))
    except (OSError, ValueError, KeyError, TypeError):
        valid=False
    return {'written_wall_ns':marker_path.stat().st_mtime_ns,
            'sha_valid':valid,'native_exit':native_exit,
            'owner_identity_valid':registered==owner if 'registered' in locals() else False}


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


def cross_track_mean(rows, path, origin):
    errors=[]
    for row in rows:
        if not row['label_exposure']:continue
        root=row['root_state']
        x,y=root[0]-origin[0],root[1]-origin[1]
        distances=[]
        for (ax,ay),(bx,by) in zip(path,path[1:]):
            dx,dy=bx-ax,by-ay
            fraction=max(0.0,min(1.0,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy)))
            distances.append(math.hypot(x-ax-fraction*dx,y-ay-fraction*dy))
        errors.append(min(distances))
    return sum(errors)/len(errors)


def duplicate_gate(traces, summary, terrain, publication):
    first,second=traces[:2]
    arrivals=summary['arrivals'][:2]
    failures=[]
    if arrivals[0]['right_censored']!=arrivals[1]['right_censored'] or (
            (arrivals[0]['arrival_step'] is None)!=(arrivals[1]['arrival_step'] is None)):
        failures.append('arrival_or_censor_category')
    if all(x['arrival_step'] is not None for x in arrivals) and abs(
            arrivals[0]['arrival_step']-arrivals[1]['arrival_step'])*0.05>0.10:
        failures.append('arrival_time_spread')
    def recovery(rows):
        return {'linear':any(r['follower_diagnostic']['linear_recovery'] for r in rows if r['label_exposure']),
                'angular':any(r['follower_diagnostic']['angular_recovery'] for r in rows if r['label_exposure']),
                'modes':sorted({r['follower_diagnostic']['mode'] for r in rows if r['label_exposure']})}
    recoveries=[recovery(x) for x in (first,second)]
    if recoveries[0]!=recoveries[1]:failures.append('recovery_category')
    paths=[x['map_xy'] for x in publication['trial_paths']]
    if publication['trial_paths'][0]['path_sha256']!=publication['trial_paths'][1]['path_sha256']:
        failures.append('direct_Path_geometry')
    origins=terrain['world_tile_origins']
    tracking=[cross_track_mean(traces[i],paths[i],origins[i]) for i in (0,1)]
    if abs(tracking[0]-tracking[1])>0.02:failures.append('tracking_mean_spread')
    goal=paths[0][-1]
    progresses=[]
    for i,rows in enumerate((first,second)):
        if len(rows)<40:raise ValueError('duplicate 2s progress unavailable')
        ox,oy=origins[i][:2]
        start,end=rows[0]['root_state'],rows[39]['post_root_state']
        progresses.append(math.hypot(start[0]-ox-goal[0],start[1]-oy-goal[1])-
                          math.hypot(end[0]-ox-goal[0],end[1]-oy-goal[1]))
    if abs(progresses[0]-progresses[1])>=0.025:failures.append('two_second_common_goal_progress_spread')
    return {'pass':not failures,'failures':failures,'arrivals':arrivals,
            'recovery_categories':recoveries,'tracking_mean_m':tracking,
            'two_second_common_goal_progress_m':progresses,
            'thresholds':{'arrival_time_spread_s':0.10,'tracking_mean_spread_m':0.02,
                          'two_second_common_goal_progress_spread_m_strict':0.025}}


def pair_duplicate_gate(traces, summary, terrain, publication, left, right):
    selected=(left,right)
    pair_summary={**summary,'arrivals':[summary['arrivals'][i] for i in selected]}
    pair_terrain={**terrain,'world_tile_origins':[terrain['world_tile_origins'][i] for i in selected]}
    pair_publication={**publication,'trial_paths':[publication['trial_paths'][i] for i in selected]}
    return duplicate_gate([traces[i] for i in selected],pair_summary,pair_terrain,pair_publication)


def validate_g3_output(root: pathlib.Path, remote: str, g3_contract: dict,
                       assigned: list[dict]) -> dict:
    output=root/'output'
    marker=json.loads((output/'complete.json').read_text())
    summary=json.loads((output/'pilot-summary.json').read_text())
    steps=marker.get('control_steps')
    if (marker.get('Path_publications')!=1 or marker.get('settling_control_steps')!=20
            or not isinstance(steps,int) or not 41<=steps<=640
            or summary.get('recorded_steps')!=[steps]*4):
        raise ValueError('G3 completion marker or four-branch count invalid')
    for key,name in (('initial_state_sha256','initial-state.json'),
                     ('terrain_sha256','terrain.json'),('prepath_gate_sha256','prepath-gate.json'),
                     ('process_provenance_sha256','process-provenance.json'),
                     ('prepath_physical_gate_sha256','prepath-physical-state-gate.json'),
                     ('prepath_shared_gate_sha256','prepath-shared-dem-gate.json'),
                     ('prepath_shared_ledger_sha256','prepath-shared-dem-sha-ledger.json'),
                     ('prepath_native_diagnostic_sha256','prepath-native-dem-diagnostic.json'),
                     ('prepath_branch_input_diagnostics_sha256','prepath-branch-input-diagnostics.json'),
                     ('prepath_branch_inputs_sha256','prepath-branch-inputs.npz'),
                     ('prepath_dem_ledger_sha256','prepath-dem-sha-ledger.json'),
                     ('direct_actor_gate_sha256','direct-actor-gate.json'),
                     ('path_publication_sha256','path-publication.json'),
                     ('pilot_summary_sha256','pilot-summary.json'),
                     ('contact_body_names_sha256','contact-body-names.json')):
        if digest(output/name)!=marker[key]:raise ValueError('G3 referenced SHA mismatch:'+name)
    for key,name in (('live_material_sha256','live-material.json'),
                     ('prepath_crossover_snapshot_sha256','prepath-crossover-snapshot.json'),
                     ('prepath_approval_sha256','prepath-approval.json')):
        if digest(output/name)!=marker[key]:raise ValueError('crossover referenced SHA mismatch:'+name)
    approval=json.loads((output/'prepath-approval.json').read_text())
    gate_path=output/'cross-period-prepath-gate.json'
    if not approval['pass'] or digest(gate_path)!=approval['gate_sha256'] or \
            not json.loads(gate_path.read_text())['pass']:
        raise ValueError('cross-period pre-Path approval not valid')
    dem_ledger=json.loads((output/'prepath-dem-sha-ledger.json').read_text())
    if len([name for name in dem_ledger if name.startswith('prepath-dem-env-')])!=16:
        raise ValueError('G3 raw DEM frame count invalid')
    for name,expected in dem_ledger.items():
        if pathlib.Path(name).name!=name or digest(output/name)!=expected:
            raise ValueError('G3 raw DEM evidence SHA mismatch:'+name)
    if marker.get('protocol_version')!=SCHEMA:
        raise ValueError('G3 protocol version mismatch')
    if not json.loads((output/'prepath-gate.json').read_text())['pass']:
        raise ValueError('G3 pre-Path gate failed')
    verify_shared_artifacts(output, require_native=True)
    shared_gate=json.loads((output/'prepath-shared-dem-gate.json').read_text())
    native=json.loads((output/'prepath-native-dem-diagnostic.json').read_text())
    if (not shared_gate['pass'] or shared_gate['meaning']!='shared_experimental_input_equality'
            or shared_gate['native_sensor_equality_claimed'] is not False
            or native['meaning']!='measured_native_sensor_history_equality'
            or len(native.get('frames', []))!=12):
        raise ValueError('G3 shared/native DEM semantics mismatch')
    shared_ledger=json.loads((output/'prepath-shared-dem-sha-ledger.json').read_text())
    provenance_names=set(shared_ledger) | set(dem_ledger) | {
        'prepath-shared-dem-sha-ledger.json','prepath-dem-sha-ledger.json',
        'prepath-shared-dem-gate.json','prepath-native-dem-diagnostic.json',
        'prepath-physical-state-gate.json','prepath-branch-inputs.npz',
        'prepath-branch-input-diagnostics.json','prepath-gate.json',
        'live-material.json','prepath-crossover-snapshot.json','prepath-ready.json',
        'cross-period-prepath-gate.json','prepath-approval.json'}
    result=subprocess.run(['docker','exec',CONTAINER,'sha256sum',
        *[f'{remote}/output/{name}' for name in sorted(provenance_names)]],
        capture_output=True,text=True,check=True)
    container_sha={path.rsplit('/',1)[-1]:value for value,path,*_ in
                   (line.split() for line in result.stdout.splitlines())}
    for name in provenance_names:
        if container_sha.get(name)!=digest(output/name):
            raise ValueError('G3 shared/native host/container SHA mismatch:'+name)
    traces=[]
    for index in range(4):
        path=output/f'trace-slot-{index}.jsonl'
        if digest(path)!=marker['trace_sha256'][index]:
            raise ValueError('G3 trace SHA mismatch')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=steps or [row['control_step'] for row in rows]!=list(range(steps)):
            raise ValueError('G3 trace incomplete or noncontiguous')
        gate=SameEpisodeGate(640,initial_episode_length=rows[0]['timing_protocol']['episode_length_before'],
                             episode_cap_steps=800)
        exposure=G3Exposure()
        for row in rows:
            if not _finite(row):raise ValueError('G3 nonfinite trace value')
            if row.get('terminal') is not None:raise ValueError('G3 terminal trace')
            gate.accept(row['timing_protocol'])
            active=exposure.observe(row['control_step'],bool(row['follower_diagnostic']['goal_reached']))
            if active!=row['label_exposure']:
                raise ValueError('G3 exposure flag mismatch')
        if gate.next_step!=steps or not exposure.finished_after(steps-1):
            raise ValueError('G3 same-episode or exposure incomplete')
        if summary['arrivals'][index]!={'arrival_step':exposure.arrival_step,
                    'stop_exclusive':exposure.stop_exclusive,'right_censored':exposure.censored}:
            raise ValueError('G3 arrival/censor summary mismatch')
        traces.append(rows)
    for name in ('complete.json',*[f'trace-slot-{i}.jsonl' for i in range(4)]):
        result=subprocess.run(['docker','exec',CONTAINER,'sha256sum',f'{remote}/output/{name}'],
                              capture_output=True,text=True,check=True)
        if result.stdout.split()[0]!=digest(output/name):
            raise ValueError('G3 host/container SHA mismatch:'+name)
    stamps=[row['timing_protocol']['stamp_ns'] for row in traces[0]]
    validate_four_follower_final(summary['bridge_final']['snapshot'],steps,stamps[-1])
    for rows in traces[1:]:
        if [row['timing_protocol']['stamp_ns'] for row in rows]!=stamps:
            raise ValueError('G3 env logical clocks differ')
    terrain=json.loads((output/'terrain.json').read_text())
    process_provenance=json.loads((output/'process-provenance.json').read_text())
    publication=json.loads((output/'path-publication.json').read_text())
    if (process_provenance['anchor_id']!=summary['anchor_id'] or
            process_provenance['batch_index']!=summary['batch_index'] or
            process_provenance['execution_index']!=summary['execution_index'] or
            process_provenance['path_sha256_by_env']!=[row['path_sha256'] for row in publication['trial_paths']]):
        raise ValueError('direct repeat process provenance mismatch')
    if (len(publication.get('trial_paths',[]))!=4 or
            [entry['trial_id'] for entry in publication['trial_paths']]!=summary['trial_ids']):
        raise ValueError('crossover Path publication contract failed')
    validate_frozen_assignment(assigned, publication['trial_paths'], stage='post_path')
    if (publication.get('dem_input_role')!='shared_experimental_history'
            or publication.get('dem_source_env')!=0
            or publication.get('shared_dem_ledger_sha256')!=marker['prepath_shared_ledger_sha256']):
        raise ValueError('G3 Path publication DEM provenance mismatch')
    for env,rows in enumerate(traces):
        declared=publication['trial_paths'][env]['experimental_dem_input']
        if declared.get('role')!='shared_experimental_history' or declared.get('source_env')!=0 or \
                len(declared.get('frames',[]))!=4:
            raise ValueError('G3 per-branch shared DEM declaration invalid')
        for frame_index,frame in enumerate(declared['frames']):
            name=f'prepath-shared-dem-env-{env}-frame-{frame_index}.npy'
            if (frame['array_file']!=name or frame['array_sha256']!=shared_ledger[name] or
                    frame['source_frame_sha256']!=shared_gate['source_frame_sha256'][frame_index]):
                raise ValueError('G3 per-branch shared DEM file/source SHA mismatch')
        for row in rows:
            if (row.get('dem_input_role')!='shared_experimental_history' or
                    row.get('dem_source_env')!=0 or
                    row.get('experimental_dem_input')!=declared or
                    row.get('shared_dem_source_frame_sha256')!=shared_gate['source_frame_sha256']):
                raise ValueError('G3 trace shared DEM provenance mismatch')
    duplicates=[{'env_pair':[left,right],
                 **pair_duplicate_gate(traces,summary,terrain,publication,left,right)}
                for left,right in ((0,1),(2,3))
                if publication['trial_paths'][left]['path_role']==
                   publication['trial_paths'][right]['path_role']=='direct']
    atomic(root/'duplicate-gate.json',{'pairs':duplicates,
        'pass_pairs':sum(row['pass'] for row in duplicates),
        'fail_pairs':sum(not row['pass'] for row in duplicates)})
    return {'trace_steps_per_env':[steps]*4,'logical_first_ns':stamps[0],
            'logical_last_ns':stamps[-1],
            'logical_interval_ns':50_000_000,
            'max_episode_length_steps':800,
            'path_publications':1,
            'direct_duplicate_pairs':duplicates,
            'trace_sha256':marker['trace_sha256']}


def validate_dem_diagnostic_output(root: pathlib.Path, remote: str) -> dict:
    output=root/'output'
    marker=json.loads((output/'gate1-probe-complete.json').read_text())
    heartbeat=json.loads((output/'heartbeat.json').read_text())
    gate=json.loads((output/'prepath-gate.json').read_text())
    shared=json.loads((output/'prepath-shared-dem-gate.json').read_text())
    native=json.loads((output/'prepath-native-dem-diagnostic.json').read_text())
    if (marker.get('stage')!='pre_Path_instrumentation_probe' or
            marker.get('protocol_version')!='eatnav_p1_longhorizon_g3_shared_dem_v2' or
            marker.get('Path_publications')!=0 or heartbeat.get('Path_publications')!=0 or
            heartbeat.get('completed_control_steps')!=20 or not gate.get('pass') or
            not shared.get('pass') or shared.get('native_sensor_equality_claimed') is not False or
            native.get('meaning')!='measured_native_sensor_history_equality' or
            len(native.get('frames', []))!=12 or
            (output/'path-publication.json').exists() or (output/'complete.json').exists()):
        raise ValueError('candidate-free shared DEM diagnostic completion invalid')
    refs={'shared_dem_ledger_sha256':'prepath-shared-dem-sha-ledger.json',
          'shared_dem_gate_sha256':'prepath-shared-dem-gate.json',
          'native_dem_diagnostic_sha256':'prepath-native-dem-diagnostic.json',
          'branch_input_diagnostics_sha256':'prepath-branch-input-diagnostics.json',
          'branch_inputs_sha256':'prepath-branch-inputs.npz'}
    for key,name in refs.items():
        if marker[key]!=digest(output/name):raise ValueError('G3 probe reference SHA mismatch:'+name)
    verified=verify_shared_artifacts(output, require_native=True)
    ledger=json.loads((output/'prepath-dem-sha-ledger.json').read_text())
    frame_files=[name for name in ledger if re.fullmatch(r'prepath-dem-env-[0-3]-frame-[0-3]\.npz',name)]
    diff_files=[name for name in ledger if re.fullmatch(r'prepath-dem-diff-frame-[0-3]\.npz',name)]
    if len(frame_files)!=16 or len(diff_files)!=4:
        raise ValueError('candidate-free native DEM raw/diff file count')
    for name,expected in ledger.items():
        if pathlib.Path(name).name!=name or digest(output/name)!=expected:
            raise ValueError('candidate-free native DEM SHA mismatch:'+name)
    shared_ledger=json.loads((output/'prepath-shared-dem-sha-ledger.json').read_text())
    names=['gate1-probe-complete.json','prepath-gate.json','prepath-physical-state-gate.json',
           'prepath-shared-dem-gate.json','prepath-shared-dem-sha-ledger.json',
           'prepath-dem-sha-ledger.json','prepath-branch-inputs.npz',
           'prepath-branch-input-diagnostics.json',*sorted(ledger),*sorted(shared_ledger)]
    names=sorted(set(names))
    result=subprocess.run(['docker','exec',CONTAINER,'sha256sum',
                           *[f'{remote}/output/{name}' for name in names]],
                          capture_output=True,text=True,check=True)
    container_sha={path.rsplit('/',1)[-1]:value for value,path,*_ in
                   (line.split() for line in result.stdout.splitlines())}
    for name in names:
        if container_sha.get(name)!=digest(output/name):
            raise ValueError('candidate-free DEM host/container SHA mismatch:'+name)
    return {'settling_steps':20,'envs':4,'frames_per_env':4,'Path_publications':0,
            'physical_gate':'pass','shared_input_gate':'pass',
            'native_sensor_gate':'diagnostic_only', 'native_sensor_equal':native.get('pass'),
            'native_files':len(frame_files),'shared_files':verified['raw_shared_frame_count'],
            'shared_ledger_sha256':verified['ledger_sha256']}


def main():
    global ROOT, REMOTE, STAGE_ROOT, ROUTER
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=pathlib.Path,required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--router-log-dir',type=pathlib.Path,required=True)
    parser.add_argument('--anchor-id',choices=('first-differential','second-graded'),required=True)
    parser.add_argument('--batch-index',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--execution-index',type=int,choices=(0,1,2),required=True)
    parser.add_argument('--dem-diagnostic',action='store_true')
    args=parser.parse_args()
    ROOT=args.root.resolve();REMOTE=args.remote;STAGE_ROOT=ROOT.parent
    ROUTER=args.router_log_dir.resolve()
    if ROOT.name != f'batch-{args.anchor_id}-{args.batch_index}-period-{args.execution_index}' or (ROOT/'output').exists() or (ROOT/'host-seal.json').exists():
        raise ValueError('G3 requires a fresh dedicated batch root')
    freeze=json.loads((ROOT/'input/freeze.json').read_text())
    g3=json.loads((ROOT/'input/g3-contract.json').read_text())
    assignment=json.loads((ROOT/'input/g3-assignment.json').read_text())
    cfg=json.loads((ROOT/'input/statistical-config.json').read_text())
    if (g3['schema']!=SCHEMA
            or g3['unreached_control_step_cap']!=600 or g3['post_arrival_hold_steps']!=40
            or g3['max_control_steps']!=640 or g3['episode_length_s']!=40.0
            or g3['control_dt_s']!=0.05
            or digest(ROOT/'source.py')!=g3['g3_simulator_source_sha256']
            or digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py')!=g3['shared_dem_module_sha256']
            or digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_direct_repeat.py')!=g3['direct_repeat_module_sha256']
            or digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_crossover_runtime.py')!=g3['crossover_runtime_module_sha256']
            or digest(ROOT/'src/visualnav_transformer/eat_nav/p1_path_crossover_v2.py')!=g3['path_crossover_module_sha256']
            or digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_material_gate_v2.py')!=g3['material_gate_module_sha256']
            or digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_assignment.py')!=g3['g3_assignment_module_sha256']
            or digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_material_diagnostic.py')!=g3['material_diagnostic_module_sha256']):
        raise ValueError('frozen G3 source/contract mismatch')
    if digest(__file__)!=g3['g3_host_source_sha256'] or \
            digest(ROOT/'host-runner.py')!=g3['g3_host_source_sha256']:
        raise ValueError('G3 host source SHA drift')
    staged_ledger=json.loads((ROOT/'input-sha-ledger.json').read_text())
    for name,expected in staged_ledger.items():
        relative=pathlib.Path(name)
        if relative.is_absolute() or '..' in relative.parts or digest(ROOT/relative)!=expected:
            raise ValueError('G3 staged input SHA mismatch:'+name)
    if args.dem_diagnostic or g3.get('execution_mode')!='path_crossover_four_env':
        raise ValueError('crossover control scope mismatch')
    if digest(ROOT/'input/g3-assignment.json')!=g3['assignment_sha256']:
        raise ValueError('G3 assignment SHA drift')
    matched=[x for x in assignment['processes'] if x['anchor_id']==args.anchor_id and
          x['batch_index']==args.batch_index and x['execution_index']==args.execution_index]
    if len(matched)!=1 or matched[0]['batch_seed']!=json.loads((ROOT/'input/assignment.json').read_text())['batch_seeds'][args.anchor_id][args.batch_index]:
        raise ValueError('crossover process seed invalid')
    assigned_rows=sorted(matched[0]['trials'],key=lambda x:x['slot'])
    frozen_process=next((x for x in g3['processes'] if x['anchor_id']==args.anchor_id and
                         x['batch_index']==args.batch_index and
                         x['execution_index']==args.execution_index),None)
    frozen_rows=[] if frozen_process is None else frozen_process['trials']
    validate_frozen_assignment(frozen_rows, assigned_rows, stage='prelaunch')
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
    router_before=None if args.dem_diagnostic else router_audit()
    if router_before is not None:atomic(ROOT/'router-before.json',router_before)
    inspect=json.loads(command(['docker','inspect',CONTAINER]))[0]
    if inspect['State']['Status']!='running':raise RuntimeError('container not running')
    if subprocess.run(['docker','exec',CONTAINER,'test','-d',REMOTE],capture_output=True).returncode:
        raise RuntimeError('persistent root not visible inside container')
    checkpoint=subprocess.run(['docker','exec',CONTAINER,'sha256sum',cfg['rl_checkpoint']],
                              capture_output=True,text=True,check=True)
    if checkpoint.stdout.split()[0]!=cfg['rl_checkpoint_sha256']:
        raise ValueError('frozen RL checkpoint SHA mismatch')
    nvml=Nvml();devices,nvml_rows=nvml.sample();table=proc_table()
    baseline_identities=initial_graphics_identities(nvml_rows,table)
    baseline_gpu=capture_live(nvml,{},baseline_identities,GPU_CONTRACT,
        container_id=inspect['Id'],expected_simulator_alive=False,device_cap_mib=1024)
    if not baseline_gpu['pass'] or not baseline_identities:
        raise RuntimeError('G2 prelaunch GPU attribution invalid:'+','.join(baseline_gpu['failures']))
    if devices[0]['uuid']!=GPU_UUID or devices[0]['used_bytes']>1024*1048576:
        raise RuntimeError('G2 prelaunch GPU identity/cap')
    if shutil.disk_usage(ROOT).free<2_000_000_000:
        raise RuntimeError('G2 host disk free floor')
    atomic(ROOT/'g3-prelaunch.json',{'container_id':inspect['Id'],'image_id':inspect['Image'],
        'execution_mode':'path_crossover_four_env',
        'external_router_used':not args.dem_diagnostic,
        'source_sha256':digest(ROOT/'source.py'),
        'shared_dem_module_sha256':digest(ROOT/'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py'),
        'g3_contract_sha256':digest(ROOT/'input/g3-contract.json'),
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
    atomic(coord/'launch-intent.json',{'schema':'eatnav_p1_g3_shared_dem_start_v2','run_token':token,
        'container_id':inspect['Id'],'output_root':f'{REMOTE}/output',
        'source_sha256':source_sha,'config_sha256':config_sha,
        'wrapper_sha256':digest(ROOT/'owner-wrapper.py')})
    cmd=build_wrapper_command(remote=REMOTE,container_id=inspect['Id'],token=token,
                              source_sha=source_sha,config_sha=config_sha,
                              anchor_id=args.anchor_id,batch_index=args.batch_index,execution_index=args.execution_index,
                              dem_diagnostic=args.dem_diagnostic)
    atomic(ROOT/'g3-command.json',{'argv':cmd,'wrapper_sha256':digest(ROOT/'owner-wrapper.py')})
    ros_env=None;bridge_cmd=None
    if not args.dem_diagnostic:
        ros_env=build_ros_host_env(repo=pathlib.Path(__file__).resolve().parents[1])
        ros_env['RMW_IMPLEMENTATION']='rmw_zenoh_cpp'
        atomic(ROOT/'bridge-environment.json',env_audit(ros_env,'/usr/bin/python3'))
        bridge_cmd=['/usr/bin/python3',str(pathlib.Path(__file__).with_name('eat_nav_p1_paired_bridge.py')),
            '--branches','4','--follower',str(FOLLOWER),'--params',str(params),
            '--diagnostic-dir',str(ROOT/'follower-logs')]
        atomic(ROOT/'bridge-command.json',{'argv':bridge_cmd})
    gate=G1Lifecycle(token=token,container_id=inspect['Id'],source_sha=source_sha,
        config_sha=config_sha,output_root=f'{REMOTE}/output',gpu_uuid=GPU_UUID)
    started=time.monotonic();failure=None;owner=None;client=None;bridge=None;bridge_tick=None
    native_exit=None;brackets=0;cleanup_signals=[];owned_identities={};bridge_residual=[]
    prepath_approved=False
    monitor_failure=None;terminal_gpu_row=None;terminal_gpu_classification=None
    with (ROOT/'runner.log').open('xb') as log, \
         (ROOT/'bridge.log').open('xb') as bridge_log, \
         (ROOT/'resource-series.jsonl').open('x') as resources, \
         (ROOT/'owner-polls.jsonl').open('x') as polls:
        try:
            if not args.dem_diagnostic:
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
                if bridge is not None and bridge.poll() is not None:
                    raise RuntimeError('bridge exited before simulator authorization')
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
                if bridge is not None and bridge.poll() is not None:
                    raise RuntimeError('bridge exited during G3')
                gpu,row=sample(started,nvml,{'simulator':owner},baseline_identities,inspect['Id'])
                append_jsonl(resources,row);brackets+=1
                if violation:=validate_sample(row):
                    monitor_failure=violation
                    if not violation.startswith('gpu_attribution:') or \
                            not (ROOT/'output/complete.json').is_file():
                        raise RuntimeError(violation)
                    terminal_gpu_row=row
                    break
                gate.observe(gpu)
                for pid in gpu.get('owned_process_tree_host_pids',[]):
                    proc=gpu['process_table_before'].get(pid)
                    if proc:owned_identities[pid]=proc['start_ticks']
                if row['elapsed_s']>600:raise TimeoutError('G3 process wall cap')
                if not prepath_approved and (ROOT/'output/prepath-ready.json').is_file():
                    approve_prepath(ROOT,args.anchor_id,args.batch_index,args.execution_index)
                    prepath_approved=True
                heartbeat=ROOT/'output/heartbeat.json'
                if heartbeat.is_file() and time.time()-heartbeat.stat().st_mtime>90:
                    raise TimeoutError('G3 heartbeat stale')
                time.sleep(.5)
            native_exit=client.wait(timeout=30)
            if terminal_gpu_row is not None:
                bracket={'sample_started_wall_ns':terminal_gpu_row['sample_started_wall_ns'],
                         'sample_finished_wall_ns':terminal_gpu_row['sample_finished_wall_ns'],
                         'failures':terminal_gpu_row['gpu_attribution']['failures'],
                         'device_cap_pass':terminal_gpu_row['gpu_device_total_mib']<=7680 and
                             terminal_gpu_row['host_available_bytes']>=10095421850 and
                             terminal_gpu_row['container_anon_plus_shmem_bytes']<=6442450944 and
                             terminal_gpu_row['persistent_root_bytes']<=1000000000,
                         'attribution_gap_pass':
                             terminal_gpu_row['gpu_attribution']['bracket_gap_bytes']<=256*1048576}
                terminal_gpu_classification=classify_gpu_bracket(
                    bracket,terminal_completion(ROOT/'output',native_exit,owner,ROOT))
                if terminal_gpu_classification['status'] not in (
                        'terminal_expected_exit','terminal_valid'):
                    raise RuntimeError('terminal_gpu_boundary_inconclusive')
                monitor_failure=None
            if not prepath_approved:
                raise RuntimeError('simulator exited without cross-period pre-Path approval')
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
        if router_before is not None:
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
    artifact_failure=simulator_artifact_failure(ROOT/'output',native_exit,
                                                diagnostic=args.dem_diagnostic)
    if artifact_failure is not None:
        failure=artifact_failure if failure is None else artifact_failure+'; host_observation='+failure
    validation=None
    if failure is None:
        try:
            validation=(validate_dem_diagnostic_output(ROOT,REMOTE) if args.dem_diagnostic else
                        validate_g3_output(ROOT,REMOTE,g3,assigned_rows))
            gate.complete(native_exit=native_exit,owner_alive=bool(residual),
                          writers_closed=all(writer_state.values()))
        except BaseException as exc:failure=type(exc).__name__+':'+str(exc)
    atomic(ROOT/'cleanup-evidence.json',{'signals':cleanup_signals,'owned_residual':residual,
        'bridge_residual':bridge_residual,'writer_state':writer_state,'native_exit':native_exit})
    atomic(ROOT/'g3-host-status.json',{'failure':failure,'native_exit':native_exit,
        'monitor_failure':monitor_failure,
        'monitor_terminal_raw_failure':terminal_gpu_row['gpu_attribution']['failures']
            if terminal_gpu_row is not None else None,
        'monitor_terminal_classification':terminal_gpu_classification,
        'elapsed_s':time.monotonic()-started,'gpu_bracket_count':brackets,
        'owned_gpu_bracket_count':gate.live_gpu_samples,'owned_simulator':owner,
        'owned_simulator_residual':bool(residual),'lifecycle_phase':gate.phase,
        'validation':validation})
    if all(writer_state.values()):
        stage=('G3_DEM_diagnostic_protocol_valid' if failure is None else
               'G3_DEM_diagnostic_failed_or_inconclusive') if args.dem_diagnostic else (
               'G3_crossover_batch_protocol_valid' if failure is None else 'G3_crossover_batch_failed_or_inconclusive')
        seal_after_writers(ROOT,writer_state=writer_state,stage=stage,reason=failure)
        time.sleep(.1);verify_seal(ROOT)
    else:
        atomic(ROOT/'unsealed-evidence.json',{'failure':failure,'writer_state':writer_state})
        raise RuntimeError('G3 cannot seal while owned writer remains')
    if failure is not None:raise RuntimeError('G3 failed or inconclusive: '+failure)
    print('G3_BATCH_PASS',str(ROOT/'host-seal.json'))


if __name__=='__main__':
    main()
