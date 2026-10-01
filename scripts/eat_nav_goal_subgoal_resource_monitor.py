"""External resource-aware heartbeat for one full-prefix timing probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
import subprocess
import time


STAT_KEYS = ('anon', 'file', 'shmem', 'slab', 'kernel', 'sock', 'pagetables',
             'inactive_file', 'active_file', 'inactive_anon', 'active_anon')
EVENT_KEYS = ('low', 'high', 'max', 'oom', 'oom_kill')


def atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_failure_marker(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    atomic(path, value)


def heartbeat_stale(path: Path, now_unix: float, limit_sec: float) -> bool:
    return path.exists() and now_unix - path.stat().st_mtime > limit_sec


def parse_pairs(raw: str, *, wanted: tuple[str, ...] | None = None) -> dict[str, int]:
    result = {}
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            try:
                result[parts[0].rstrip(':')] = int(parts[1])
            except ValueError:
                continue
    if wanted is not None:
        missing = set(wanted) - result.keys()
        if missing:
            raise ValueError(f'missing resource fields: {sorted(missing)}')
        return {key: result[key] for key in wanted}
    return result


def anon_shmem(stat: dict[str, int]) -> int:
    return stat['anon'] + stat['shmem']


def completed_suffix_step(log_path: Path, anchor_step: int) -> int | None:
    """Read the existing simulator step marker without changing its control path."""
    if not log_path.exists():
        return None
    tail = log_path.read_text(errors='replace')[-65536:]
    matches = re.findall(r'^DIAG_ENV_STEP_RETURN (\d+) \d+$', tail, re.MULTILINE)
    if not matches:
        return None
    return anchor_step + 1 + int(matches[-1])


def event_delta(events: dict[str, int], baseline: dict[str, int]) -> dict[str, int]:
    result = {key: events[key] - baseline[key] for key in EVENT_KEYS}
    if any(value < 0 for value in result.values()):
        raise ValueError('cgroup memory.events counters moved backwards')
    return result


def evaluate(snapshot: dict, config: dict, state: dict) -> tuple[str | None, list[str]]:
    """Return hard-stop reason and warnings; mutate only the host-low streak."""
    warnings = []
    delta = event_delta(snapshot['memory_events'], config['memory_events_baseline'])
    if delta['oom'] > 0 or delta['oom_kill'] > 0:
        return 'cgroup OOM event increased', warnings
    if snapshot['kernel_oom'] or snapshot['docker_oom']:
        return 'kernel or Docker OOM recorded', warnings
    if snapshot['memory_max_bytes'] is not None and snapshot['memory_current_bytes'] >= snapshot['memory_max_bytes']:
        return 'finite Docker memory.max reached', warnings
    if anon_shmem(snapshot['memory_stat']) > config['anon_shmem_cap_bytes']:
        return 'anonymous plus shared memory cap exceeded', warnings
    if snapshot['mem_available_bytes'] < config['host_mem_available_min_bytes']:
        state['host_low_streak'] = state.get('host_low_streak', 0) + 1
        warnings.append('host MemAvailable below minimum')
        if state['host_low_streak'] >= 3:
            return 'host MemAvailable below minimum for three heartbeats', warnings
    else:
        state['host_low_streak'] = 0
    if snapshot['gpu_used_mib'] > config['gpu_cap_mib']:
        return 'GPU memory cap exceeded', warnings
    if snapshot['gpu_xid']:
        return 'NVIDIA Xid recorded', warnings
    if snapshot['simulator_alive'] and snapshot['seconds_since_progress'] >= config['step_stall_sec']:
        return 'no completed simulator step for 120 seconds', warnings
    if snapshot['output_bytes'] >= config['storage_cap_bytes'] or snapshot['projected_output_bytes'] >= config['storage_cap_bytes']:
        return 'probe output reached or is projected above storage cap', warnings
    if not snapshot['router_alive']:
        return 'router process stopped', warnings
    if not snapshot['docker_exec_alive'] and not state.get('cleanup_grace_active'):
        return 'Docker exec process stopped before cleanup', warnings
    logical_time = snapshot['simulator_logical_time_sec']
    if logical_time is not None and not math.isfinite(logical_time):
        return 'nonfinite simulator logical time', warnings
    if snapshot['memory_stat']['file'] > config['file_cache_warn_bytes']:
        warnings.append('file cache above warning threshold; no hard stop')
    return None, warnings


def call(*args: str, timeout: int = 8) -> str:
    return subprocess.check_output(args, text=True, timeout=timeout, stderr=subprocess.DEVNULL).strip()


def remote(*args: str) -> str:
    return call('docker', 'exec', 'isaac-lab-onix', *args)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def maybe(fn, default=None):
    try:
        return fn()
    except (OSError, ValueError, subprocess.SubprocessError, json.JSONDecodeError):
        return default


def process_rows(needle: str):
    rows = call('ps', '-eo', 'pid,ppid,rss,nlwp,args').splitlines()
    return [row for row in rows if needle in row and 'ps -eo' not in row]


def read_pss_kib(pid: int | None):
    if pid is None:
        return None
    raw = maybe(lambda: remote('cat', f'/proc/{pid}/smaps_rollup'))
    if raw is None:
        return None
    match = re.search(r'^Pss:\s+(\d+)\s+kB$', raw, re.MULTILINE)
    return int(match.group(1)) if match else None


def process_detail(pid: int | None):
    if pid is None:
        return None
    raw = maybe(lambda: remote('ps', '-o', 'pid=,rss=,nlwp=,time=', '-p', str(pid)))
    if not raw:
        return None
    fields = raw.split()
    fd = maybe(lambda: len(remote('sh', '-c', f'ls /proc/{pid}/fd').splitlines()))
    sockets = maybe(lambda: int(remote('sh', '-c', f"ls -l /proc/{pid}/fd 2>/dev/null | grep -Fc 'socket:['")), 0)
    return {'pid': pid, 'rss_bytes': int(fields[1]) * 1024, 'threads': int(fields[2]),
            'cpu_time_ps': fields[3], 'pss_bytes': (value * 1024 if (value := read_pss_kib(pid)) is not None else None),
            'fd_count': fd, 'socket_count': sockets}


def kernel_flags(start_unix: float):
    stamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_unix))
    raw = maybe(lambda: call('journalctl', '-k', '--since', stamp, '--no-pager', '-o', 'cat', timeout=10))
    if raw is None:
        return None, None
    return bool(re.search(r'\bNVRM: Xid\b', raw, re.IGNORECASE)), bool(re.search(r'out of memory|oom-kill|killed process', raw, re.IGNORECASE))


def docker_oom(start_unix: float):
    raw = maybe(lambda: call('docker', 'events', '--since', str(int(start_unix)), '--until', str(int(time.time())),
                             '--filter', 'container=isaac-lab-onix', '--filter', 'event=oom', '--format', '{{.Action}}'))
    return None if raw is None else bool(raw)


def gpu_snapshot():
    line = call('nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits').splitlines()[0]
    used, utilization, temperature = [int(part.strip()) for part in line.split(',')]
    compute = maybe(lambda: call('nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'), '')
    return {'used_mib': used, 'utilization_percent': utilization, 'temperature_c': temperature,
            'compute_processes': compute.splitlines() if compute else []}


def output_bytes(path: Path):
    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def verified_complete_marker(folder: Path) -> bool:
    """Accept cleanup only after host output copies match the atomic marker."""
    path = folder / 'run-complete.json'
    if not path.exists():
        return False
    complete = json.loads(path.read_text())
    if complete.get('protocol') != 'fresh_bridge_per_run':
        raise ValueError('unexpected complete marker protocol')
    outputs = complete.get('output_sha256')
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError('missing completed output SHA ledger')
    for name, digest in outputs.items():
        if Path(name).name != name or not (folder / name).is_file():
            raise ValueError('completed output missing: ' + name)
        h = hashlib.sha256()
        with (folder / name).open('rb') as stream:
            for part in iter(lambda: stream.read(1 << 20), b''):
                h.update(part)
        if h.hexdigest() != digest:
            raise ValueError('completed output SHA mismatch: ' + name)
    return True


def v2_storage_projection(*, root: Path, active: Path, remote_root_bytes: int,
                          remote_active_bytes: int, completed_remote_bytes: dict[str, int],
                          schedule: dict, completed_step: int,
                          target_step: int, initial_active_bytes: int,
                          baseline_bytes: int) -> dict:
    """Project the fixed schedule using control steps and completed run sizes.

    Host and Docker output roots are distinct storage locations.  Count both
    actual footprints, including the preserved invalid attempt.  The frozen
    pre-run estimate remains a floor until at least one live run completes.
    """
    host_root_bytes = output_bytes(root)
    host_active_bytes = output_bytes(active)
    actual = host_root_bytes + remote_root_bytes
    active_raw = host_active_bytes + remote_active_bytes
    runs = schedule['runs']
    if len(runs) != 48 or baseline_bytes != 239_907_072:
        raise ValueError('v2 schedule or frozen storage baseline differs')
    completed = []
    completed_live = []
    completed_host_bytes = 0
    completed_physical_bytes = 0
    compressed_logs = 0
    for entry in runs:
        name = (entry['run_id'] + '-attempt-1' if entry['order'] == 1 else entry['run_id'])
        folder = root / 'runs' / name
        if (folder / 'run-complete.json').exists() or (folder / 'recovered_complete_after_monitor_cleanup_race.json').exists():
            completed.append(entry)
            host_size = output_bytes(folder)
            completed_host_bytes += host_size
            if name not in completed_remote_bytes:
                raise ValueError('completed Docker run size missing: ' + name)
            physical_size = host_size + completed_remote_bytes[name]
            completed_physical_bytes += physical_size
            if entry['kind'] != 'anchor':
                completed_live.append(physical_size)
            compressed_logs += sum(p.stat().st_size for p in folder.rglob('*.gz') if p.is_file())
    total_steps = target_step + 1 + 120
    done_steps = max(0, min(total_steps, completed_step + 1))
    remaining_steps = total_steps - done_steps
    growth = max(0, active_raw - initial_active_bytes)
    rate = growth / max(1, done_steps)
    active_future_growth = int(math.ceil(rate * remaining_steps))
    bootstrap_per_run = baseline_bytes / len(runs)
    measured_per_run = max(completed_live) if completed_live else None
    future_per_run = max(bootstrap_per_run, measured_per_run or 0)
    future_count = max(0, len(runs) - len(completed) - 1)
    predicted = max(actual, baseline_bytes,
                    int(math.ceil(actual + active_future_growth + future_count * future_per_run)))
    return {'actual_storage_bytes': actual, 'actual_host_root_bytes': host_root_bytes,
            'actual_docker_root_bytes': remote_root_bytes,
            'active_raw_bytes': active_raw, 'active_host_bytes': host_active_bytes,
            'active_docker_bytes': remote_active_bytes,
            'cleanup_compressed_bytes': compressed_logs,
            'completed_host_run_bytes': completed_host_bytes,
            'completed_physical_run_bytes': completed_physical_bytes,
            'completed_run_count': len(completed),
            'completed_live_count': len(completed_live),
            'remaining_current_control_steps': remaining_steps,
            'remaining_future_run_count': future_count,
            'observed_active_growth_bytes_per_step': rate,
            'projected_active_remaining_bytes': active_future_growth,
            'bootstrap_total_bytes': baseline_bytes,
            'future_run_estimate_bytes': future_per_run,
            'projected_final_storage_bytes': predicted}


def failure_evidence(folder: Path, reason: str, heartbeat: dict):
    evidence = {'reason': reason, 'heartbeat': heartbeat, 'captured_wall_ns': time.time_ns()}
    for key, argv in {
        'host_process_tree': ('ps', '-eo', 'pid,ppid,etimes,%cpu,%mem,rss,nlwp,args'),
        'docker_process_tree': ('docker', 'exec', 'isaac-lab-onix', 'ps', '-eo', 'pid,ppid,etimes,%cpu,%mem,rss,nlwp,args'),
        'gpu_processes': ('nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'),
        'sockets': ('ss', '-tunap'),
    }.items():
        evidence[key] = maybe(lambda argv=argv: call(*argv), 'unavailable')
    for name in ('runner.log', 'simulator.log', 'bridge.log'):
        path = folder / name
        if path.exists():
            evidence[name + '_tail'] = path.read_text(errors='replace').splitlines()[-100:]
    write_failure_marker(folder / 'resource-stop.failure.json', evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--remote-progress', required=True)
    parser.add_argument('--sim-host-pid', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--target-step', type=int, required=True)
    parser.add_argument('--target-episode', type=int, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--v2-schedule', type=Path)
    parser.add_argument('--v2-forecast', type=Path)
    parser.add_argument('--v2-storage-protocol', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    start_unix = time.time()
    start_ns = time.monotonic_ns()
    last_step = -1
    last_advance_ns = start_ns
    state = {'host_low_streak': 0}
    previous_cpu = None
    v2_schedule = json.loads(args.v2_schedule.read_text()) if args.v2_schedule else None
    v2_baseline = (json.loads(args.v2_forecast.read_text())['storage_estimate_bytes']
                   if args.v2_forecast else None)
    v2_protocol = json.loads(args.v2_storage_protocol.read_text()) if args.v2_storage_protocol else None
    if sum(value is not None for value in (v2_schedule, v2_baseline, v2_protocol)) not in (0, 3):
        raise ValueError('v2 schedule, forecast and storage protocol must be supplied together')
    if v2_protocol is not None and (v2_protocol['storage_cap_bytes'] != config['storage_cap_bytes']
            or v2_protocol['bootstrap_total_bytes'] != v2_baseline
            or v2_protocol['run_control_steps'] != 120
            or v2_protocol['schedule_run_count'] != len(v2_schedule['runs'])):
        raise ValueError('v2 storage protocol or frozen threshold differs')
    initial_active_bytes = None
    completed_remote_sizes = {}
    cleanup_started_ns = None
    cleanup_grace_sec = 120
    while True:
        cycle_ns = time.monotonic_ns()
        if v2_protocol is None and ((args.run / 'monitor.stop').exists() or not pid_alive(args.sim_host_pid)):
            atomic(args.run / 'timing-monitor.done.json', {'status': 'stopped', 'last_step': last_step,
                                                            'last_heartbeat_monotonic_ns': cycle_ns})
            return
        if v2_protocol is not None and (args.run / 'monitor.stop').exists():
            stop = json.loads((args.run / 'monitor.stop').read_text())
            if stop.get('cleanup') or stop.get('simulator_exit_code') not in (None, 0):
                atomic(args.run / 'timing-monitor.done.json', {'status': 'aborted', 'last_step': last_step,
                                                                'last_heartbeat_monotonic_ns': cycle_ns})
                return
        progress = maybe(lambda: json.loads(remote('cat', args.remote_progress)))
        if progress and progress.get('last_completed_step', -1) > last_step:
            last_step = progress['last_completed_step']
            last_advance_ns = cycle_ns
        suffix_step = completed_suffix_step(args.run / 'simulator.log', args.target_step)
        if suffix_step is not None and suffix_step > last_step:
            last_step = suffix_step
            last_advance_ns = cycle_ns
        suffix_finished = last_step >= args.target_step + 120
        if v2_protocol is not None and suffix_finished and cleanup_started_ns is None:
            cleanup_started_ns = cycle_ns
        cleanup_active = cleanup_started_ns is not None
        state['cleanup_grace_active'] = cleanup_active
        if v2_protocol is not None and cleanup_active and not pid_alive(args.sim_host_pid):
            try:
                verified = verified_complete_marker(args.run)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                atomic(args.run / 'cleanup.failure.json', {'reason': 'host output SHA verification failed',
                                                            'detail': repr(exc), 'last_step': last_step})
                raise
            lifecycle = ('SHA_VERIFIED' if verified
                         else 'HOST_COPY' if (args.run / 'docker-exec-return.json').exists()
                         else 'SIMULATOR_EXITED')
            if lifecycle == 'SHA_VERIFIED':
                actual_final_bytes = output_bytes(args.run.parent.parent) + int(remote(
                    'du', '-sb', str(args.run.parent.parent)).split()[0])
                if actual_final_bytes > config['storage_cap_bytes']:
                    atomic(args.run / 'cleanup.failure.json', {'reason': '0.8 GB storage cap exceeded during host cleanup',
                                                                'actual_storage_bytes': actual_final_bytes})
                    raise RuntimeError('0.8 GB storage cap exceeded during host cleanup')
                atomic(args.run / 'timing-monitor.done.json', {
                    'status': 'complete', 'lifecycle': 'COMPLETE_MARKER', 'last_step': last_step,
                    'actual_final_storage_bytes': actual_final_bytes,
                    'complete_marker_sha256': hashlib.sha256((args.run / 'run-complete.json').read_bytes()).hexdigest(),
                    'last_heartbeat_monotonic_ns': cycle_ns})
                return
            if (cycle_ns - cleanup_started_ns) / 1e9 > cleanup_grace_sec:
                atomic(args.run / 'cleanup.failure.json', {'reason': 'cleanup grace exceeded before verified complete marker',
                                                            'grace_sec': cleanup_grace_sec, 'last_step': last_step})
                raise RuntimeError('cleanup grace exceeded before verified complete marker')
        else:
            lifecycle = 'SUFFIX_120_REACHED' if cleanup_active else 'RUNNING'
        if v2_protocol is not None and not pid_alive(args.sim_host_pid) and not cleanup_active:
            raise RuntimeError('Docker exec process stopped before 120-step trace')
        pid = progress.get('pid') if progress else None
        stat = parse_pairs(remote('cat', '/sys/fs/cgroup/memory.stat'), wanted=STAT_KEYS)
        events = parse_pairs(remote('cat', '/sys/fs/cgroup/memory.events'), wanted=EVENT_KEYS)
        current = int(remote('cat', '/sys/fs/cgroup/memory.current'))
        peak = int(remote('cat', '/sys/fs/cgroup/memory.peak'))
        max_raw = remote('cat', '/sys/fs/cgroup/memory.max')
        max_bytes = None if max_raw == 'max' else int(max_raw)
        mem = parse_pairs(Path('/proc/meminfo').read_text())
        host_available = mem['MemAvailable'] * 1024
        gpu = gpu_snapshot()
        kernel_xid, kernel_oom = kernel_flags(start_unix)
        event_oom = docker_oom(start_unix)
        cpu_line = Path('/proc/stat').read_text().splitlines()[0].split()[1:]
        cpu = tuple(map(int, cpu_line))
        cpu_use = None
        if previous_cpu:
            total = sum(cpu) - sum(previous_cpu)
            idle = (cpu[3] + cpu[4]) - (previous_cpu[3] + previous_cpu[4])
            cpu_use = 100 * (total - idle) / total if total > 0 else None
        previous_cpu = cpu
        host_detail = {'mem_total_bytes': mem['MemTotal'] * 1024,
                       'mem_available_bytes': host_available,
                       'swap_total_bytes': mem['SwapTotal'] * 1024,
                       'swap_free_bytes': mem['SwapFree'] * 1024,
                       'load_average': os.getloadavg(), 'cpu_utilization_percent': cpu_use,
                       'psi': {key: maybe(lambda key=key: Path('/proc/pressure/' + key).read_text())
                               for key in ('memory', 'cpu', 'io')}}
        remote_folder = args.remote_progress.rsplit('/', 1)[0]
        remote_size = maybe(lambda: int(remote('du', '-sb', remote_folder).split()[0]), 0)
        total_size = output_bytes(args.run) + remote_size
        elapsed = max(1., (cycle_ns - start_ns) / 1e9)
        storage = None
        if v2_schedule is not None:
            remote_root = str(args.run.parent.parent)
            remote_root_size = int(remote('du', '-sb', remote_root).split()[0])
            for entry in v2_schedule['runs']:
                name = entry['run_id'] + '-attempt-1' if entry['order'] == 1 else entry['run_id']
                if name not in completed_remote_sizes and ((args.run.parent / name / 'run-complete.json').exists()
                        or (args.run.parent / name / 'recovered_complete_after_monitor_cleanup_race.json').exists()):
                    completed_remote_sizes[name] = int(remote('du', '-sb',
                        remote_root + '/runs/' + name).split()[0])
            if initial_active_bytes is None:
                initial_active_bytes = total_size
            storage = v2_storage_projection(root=args.run.parent.parent, active=args.run,
                remote_root_bytes=remote_root_size, remote_active_bytes=remote_size,
                completed_remote_bytes=completed_remote_sizes,
                schedule=v2_schedule, completed_step=last_step,
                target_step=args.target_step, initial_active_bytes=initial_active_bytes,
                baseline_bytes=v2_baseline)
            projected_size = storage['projected_final_storage_bytes']
        else:
            projected_size = total_size if elapsed < 60 else int(total_size / elapsed * config['run_timeout_sec'])
        snapshot = {
            'memory_current_bytes': current, 'memory_peak_bytes': peak, 'memory_max': max_raw,
            'memory_max_bytes': max_bytes, 'memory_events': events, 'memory_stat': stat,
            'anonymous_plus_shmem_bytes': anon_shmem(stat),
            'mem_available_bytes': host_available, 'gpu_used_mib': gpu['used_mib'],
            'gpu_xid': bool(kernel_xid), 'kernel_oom': bool(kernel_oom), 'docker_oom': bool(event_oom),
            'simulator_alive': pid_alive(args.sim_host_pid),
            'docker_exec_alive': pid_alive(args.sim_host_pid),
            'router_alive': bool(process_rows('/rmw_zenohd')),
            'seconds_since_progress': (cycle_ns - last_advance_ns) / 1e9,
            'output_bytes': storage['actual_storage_bytes'] if storage else total_size,
            'projected_output_bytes': projected_size,
            'storage': storage,
            'simulator_logical_time_sec': progress.get('simulator_logical_time_sec') if progress else None,
        }
        reason, warnings = evaluate(snapshot, config, state)
        heartbeat = {'run_id': args.run_id, 'monitor_pid': os.getpid(),
                     'lifecycle': lifecycle,
                     'sim_host_pid': args.sim_host_pid, 'simulator_pid': pid,
                     'process_started_monotonic_ns': progress.get('process_started_monotonic_ns') if progress else None,
                     'monotonic_ns': cycle_ns, 'wall_ns': time.time_ns(),
                     'global_step': last_step, 'episode': progress.get('current_episode') if progress else None,
                     'suffix_control_step': (last_step - args.target_step - 1) if suffix_step is not None else None,
                     'episode_local_step': progress.get('episode_local_step') if progress else None,
                     'target_global_step': args.target_step, 'target_episode': args.target_episode,
                     'last_progress_monotonic_ns': last_advance_ns,
                     'last_reset_monotonic_ns': progress.get('last_reset_monotonic_ns') if progress else None,
                     'reset_count': progress.get('current_episode') if progress else None,
                     'simulator_process_cpu_ns': progress.get('process_cpu_ns') if progress else None,
                     'rolling_step_median_ns': progress.get('rolling_step_median_ns') if progress else None,
                     'rolling_step_p95_ns': progress.get('rolling_step_p95_ns') if progress else None,
                     'rolling_step_max_ns': progress.get('rolling_step_max_ns') if progress else None,
                     'elapsed_wall_sec': elapsed, 'process': {
                         'simulator': process_detail(pid),
                         'runner': process_rows('eat_nav_goal_subgoal_resource_probe.py'),
                         'router': process_rows('/rmw_zenohd'),
                         'bridge': process_rows('eat_nav_command_delivery_bridge.py'),
                         'follower': process_rows('simple_trajectory_follower_node')},
                     'host': host_detail, 'gpu': gpu, 'resource': snapshot,
                     'memory_events_delta': event_delta(events, config['memory_events_baseline']),
                     'kernel_log_available': kernel_xid is not None,
                     'docker_events_available': event_oom is not None,
                     'warnings': warnings, 'stop_reason': reason}
        atomic(args.run / 'timing-heartbeat.json', heartbeat)
        atomic(args.run.parent.parent / 'timing-heartbeat.json', heartbeat)
        with (args.run / 'resource-timeseries.jsonl').open('a') as stream:
            stream.write(json.dumps(heartbeat, sort_keys=True, allow_nan=False) + '\n')
        if reason:
            failure_evidence(args.run, reason, heartbeat)
            raise RuntimeError(reason)
        time.sleep(config['heartbeat_interval_sec'])


if __name__ == '__main__':
    main()
