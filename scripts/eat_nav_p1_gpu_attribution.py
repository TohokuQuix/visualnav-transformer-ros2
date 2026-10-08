"""P1 v2 live GPU process attribution with exact host PID/start-tick identity."""
from __future__ import annotations

import ctypes
import math
import subprocess
import time
from collections import defaultdict

from eat_nav_g0e_gpu_probe import MIB, UNKNOWN, Nvml, descendants, proc_table


KNOWN_GRAPHICS_NAMES = ('Xorg', 'Xwayland', 'gnome-shell', 'kwin_wayland', 'kwin_x11',
                        'nvidia-persistenced', 'plasmashell')


def map_container_pid(table: dict, container_pid: int, container_id: str) -> dict:
    matches = [row for row in table.values()
               if row.get('container_pid') == container_pid and container_id in row.get('cgroup', '')]
    if len(matches) != 1 or not isinstance(matches[0].get('start_ticks'), int):
        raise ValueError('container_pid_mapping_missing_or_ambiguous')
    return {'host_pid':matches[0]['host_pid'], 'start_ticks':matches[0]['start_ticks']}


def _bracket_gap(left: int, right: int, summed: int) -> int:
    if summed < min(left, right):
        return min(left, right) - summed
    if summed > max(left, right):
        return summed - max(left, right)
    return 0


def classify_bracket(device_before: dict, gpu_rows: list[dict], table_before: dict | None,
                     device_after: dict, table_after: dict | None, owned: dict,
                     baseline_identities: set[tuple[int, int]], contract: dict,
                     *, container_id: str, expected_simulator_alive: bool,
                     health_ok: bool, accounting_mode: str,
                     device_cap_mib: int | None = None) -> dict:
    """Classify a single NVML bracket; no process or cap is inferred from a missing row."""
    failures = []
    if table_before is None or table_after is None or not table_before or not table_after:
        failures.append('missing_process_table')
        table_before, table_after = table_before or {}, table_after or {}
    uuid = device_before.get('uuid')
    if uuid != contract['required_gpu_uuid'] or device_after.get('uuid') != uuid:
        failures.append('gpu_uuid_mismatch')
    if device_before.get('total_bytes', 0) < contract['required_total_memory_mib_min'] * MIB:
        failures.append('gpu_total_memory_mismatch')
    if not health_ok or not math.isfinite(float(device_before.get('temperature_c', float('nan')))):
        failures.append('gpu_health_query')
    cap = (contract['gpu_device_total_hard_cap_mib'] if device_cap_mib is None else device_cap_mib) * MIB
    if max(device_before.get('used_bytes', 0), device_after.get('used_bytes', 0)) > cap:
        failures.append('device_total_cap')
    if (not gpu_rows and max(device_before.get('used_bytes', 0),device_after.get('used_bytes', 0)) >
            contract['bracket_nonatomic_sampling_gap_hard_cap_mib'] * MIB):
        failures.append('empty_gpu_process_table')
    simulator = owned.get('simulator')
    roots=set()
    if simulator:
        pid=simulator['host_pid'];start=simulator['start_ticks']
        observed=table_before.get(pid)
        if observed is None or observed.get('start_ticks') != start:
            failures.append('owned_start_tick_mismatch')
        else:
            roots.add(pid)
        after=table_after.get(pid)
        if expected_simulator_alive and (after is None or after.get('start_ticks') != start):
            failures.append('simulator_disappeared_in_bracket')
    elif expected_simulator_alive:
        failures.append('owned_simulator_identity_missing')
    tree=descendants(table_before,roots) if roots else set()
    merged={}
    for row in gpu_rows:
        pid=row['host_pid'];proc=table_before.get(pid)
        key=(row.get('gpu_uuid'),pid,proc.get('start_ticks') if proc else None)
        if key not in merged:
            merged[key]={**row,'kinds':{row['kind']},'proc':proc}
        else:
            merged[key]['used_gpu_memory_bytes']=max(merged[key]['used_gpu_memory_bytes'],row['used_gpu_memory_bytes'])
            merged[key]['kinds'].add(row['kind'])
    counters=defaultdict(int);details=[];owned_gpu=[]
    for row in merged.values():
        pid=row['host_pid'];proc=row['proc'];size=row['used_gpu_memory_bytes']
        if size == UNKNOWN or size is None or size < 0:
            failures.append('per_pid_gpu_memory_unavailable');size=0
        identity=(pid,proc['start_ticks']) if proc and isinstance(proc.get('start_ticks'),int) else None
        kinds=sorted(row['kinds'])
        if row['gpu_uuid'] != uuid:
            failures.append('gpu_process_uuid_mismatch')
        if proc is None or identity is None:
            classification='unresolved'
        elif pid in tree:
            after=table_after.get(pid)
            if expected_simulator_alive and (after is None or after.get('start_ticks') != proc['start_ticks']):
                failures.append('owned_process_disappeared_in_bracket')
            if container_id not in proc.get('cgroup',''):
                failures.append('owned_container_mismatch')
            classification='owned_simulator_tree';owned_gpu.append(pid)
        elif 'graphics' in kinds and (identity in baseline_identities or
                                       any(name in proc.get('command_line','') for name in KNOWN_GRAPHICS_NAMES)):
            classification='known_graphics'
        elif 'compute' in kinds and identity in baseline_identities:
            classification='known_unrelated_compute'
        elif 'compute' in kinds:
            classification='unexpected_compute';failures.append('unexpected_compute_process')
        else:
            classification='unresolved'
        counters[classification]+=size
        details.append({'host_pid':pid,'start_ticks':identity[1] if identity else None,
                        'gpu_uuid':row['gpu_uuid'],'kinds':kinds,'used_gpu_memory_bytes':size,
                        'classification':classification,'command_line':proc.get('command_line') if proc else None,
                        'executable':proc.get('executable') if proc else None,
                        'container_pid':proc.get('container_pid') if proc else None,
                        'cgroup':proc.get('cgroup') if proc else None})
    unresolved=counters['unresolved']+counters['unexpected_compute']
    if unresolved > contract['unresolved_gpu_bytes_hard_cap']:
        failures.append('unresolved_gpu_process')
    summed=sum(row['used_gpu_memory_bytes'] for row in details)
    gap=_bracket_gap(device_before.get('used_bytes',0),device_after.get('used_bytes',0),summed)
    if gap > contract['bracket_nonatomic_sampling_gap_hard_cap_mib'] * MIB:
        failures.append('bracket_gap_cap')
    if gap > contract['absolute_attribution_gap_hard_cap_mib'] * MIB:
        failures.append('absolute_attribution_gap_cap')
    return {'pass':not failures,'failures':sorted(set(failures)),'gpu_uuid':uuid,
            'device_before_bytes':device_before.get('used_bytes'),'device_after_bytes':device_after.get('used_bytes'),
            'device_total_bytes':device_before.get('total_bytes'),
            'temperature_c':device_before.get('temperature_c'),'accounting_mode':accounting_mode,
            'processes':details,'owned_gpu_pids':sorted(owned_gpu),
            'known_graphics_bytes':counters['known_graphics'],
            'known_unrelated_compute_bytes':counters['known_unrelated_compute'],
            'unresolved_gpu_bytes':unresolved,'process_sum_bytes':summed,'bracket_gap_bytes':gap,
            'container_id':container_id,'owned_process_tree_host_pids':sorted(tree)}


def capture_live(nvml: Nvml, owned: dict, baseline_identities: set[tuple[int,int]],
                 contract: dict, *, container_id: str, expected_simulator_alive: bool,
                 device_cap_mib: int | None = None, owner_resolver=None) -> dict:
    """Take two memory brackets around a live NVML and /proc ownership sample."""
    started=time.monotonic_ns();wall=time.time_ns()
    before,rows=nvml.sample()
    before_table=proc_table();table_time=time.monotonic_ns()
    resolved_owned=owned
    if owner_resolver is not None and not owned.get('simulator'):
        try:
            resolved_owned=owner_resolver(before_table)
        except ValueError:
            resolved_owned=owned
    after=nvml.memory_snapshot();after_table=proc_table()
    finished=time.monotonic_ns()
    health=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,temperature.gpu,driver_version,mig.mode.current',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    health_ok=health.returncode==0 and bool(health.stdout.strip())
    if len(before)!=1 or len(after)!=1:
        raise RuntimeError('expected one GPU for P1 resource contract')
    result=classify_bracket(before[0],rows,before_table,after[0],after_table,resolved_owned,baseline_identities,
                            contract,container_id=container_id,
                            expected_simulator_alive=expected_simulator_alive or bool(resolved_owned.get('simulator')),
                            health_ok=health_ok,accounting_mode='not_required_live_attribution',
                            device_cap_mib=device_cap_mib)
    result.update({'wall_ns':wall,'monotonic_ns':started,'sample_finished_monotonic_ns':finished,
                   'process_table_monotonic_ns':table_time,'raw_nvml_rows':rows,
                   'process_table_before':before_table,'process_table_after':after_table,
                   'resolved_owned':resolved_owned,
                   'health_query_stdout':health.stdout,'health_query_stderr':health.stderr,
                   'query_timing':{**nvml.last_query_timing,**nvml.last_memory_snapshot_timing}})
    return result
