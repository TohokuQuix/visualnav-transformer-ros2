"""Independent heartbeat and stall monitor for a single fresh-process replay."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def atomic(path: Path, value) -> None:
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, timeout=8).strip()


def remote(*args: str) -> str:
    return command('docker', 'exec', 'isaac-lab-onix', *args)


def alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def remote_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        return remote('test', '-d', f'/proc/{pid}') == ''
    except (subprocess.SubprocessError, OSError):
        return False


def remote_json(path: str):
    try:
        return json.loads(remote('cat', path))
    except (subprocess.SubprocessError, OSError, json.JSONDecodeError):
        return None


def process_matches(phrase: str):
    output = command('ps', '-eo', 'pid,ppid,etimes,args')
    return [line for line in output.splitlines() if phrase in line and 'ps -eo' not in line]


def bytes_total(folder: Path) -> int:
    return sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())


def capture_failure(run: Path, reason: str, heartbeat: dict):
    evidence = {'reason': reason, 'last_heartbeat': heartbeat, 'captured_wall_ns': time.time_ns()}
    for key, args in {
        'host_process_tree': ('ps', '-eo', 'pid,ppid,etimes,%cpu,%mem,rss,nlwp,args'),
        'docker_process_tree': ('docker', 'exec', 'isaac-lab-onix', 'ps', '-eo', 'pid,ppid,etimes,%cpu,%mem,rss,nlwp,args'),
        'gpu_processes': ('nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'),
        'network_sockets': ('ss', '-tunap'),
    }.items():
        try:
            evidence[key] = command(*args)
        except (subprocess.SubprocessError, OSError) as exc:
            evidence[key] = repr(exc)
    for name in ('runner.log', 'simulator.log', 'bridge.log'):
        path = run / name
        if path.exists():
            evidence[name + '_tail'] = path.read_text(errors='replace').splitlines()[-100:]
    pid = heartbeat.get('simulator_pid')
    if pid is not None:
        for key, path in {'fd_count': f'/proc/{pid}/fd', 'thread_count': f'/proc/{pid}/task',
                          'socket_summary': f'/proc/{pid}/net/tcp'}.items():
            try:
                evidence[key] = remote('sh', '-c', f'ls {path} | wc -l' if key != 'socket_summary' else f'cat {path}')
            except (subprocess.SubprocessError, OSError) as exc:
                evidence[key] = repr(exc)
    atomic(run / 'timing-stall.failure.json', evidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--remote-progress', required=True)
    parser.add_argument('--sim-host-pid', type=int, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--target-step', type=int, required=True)
    parser.add_argument('--target-episode', type=int, required=True)
    parser.add_argument('--cpu-memory-cap-bytes', type=int, default=8 * 1024**3)
    args = parser.parse_args()
    started_ns = time.monotonic_ns()
    last_step = -1
    last_advance_ns = started_ns
    last_heartbeat_ns = started_ns
    while True:
        now = time.monotonic_ns()
        progress = remote_json(args.remote_progress)
        if progress is not None and progress.get('last_completed_step', -1) > last_step:
            last_step = progress['last_completed_step']
            last_advance_ns = now
        sim_pid = progress.get('pid') if progress else None
        try:
            rss_kib = int(remote('ps', '-o', 'rss=', '-p', str(sim_pid))) if sim_pid else None
        except (ValueError, subprocess.SubprocessError, OSError):
            rss_kib = None
        try:
            cgroup_bytes = int(remote('cat', '/sys/fs/cgroup/memory.current'))
        except (ValueError, subprocess.SubprocessError, OSError):
            cgroup_bytes = None
        try:
            gpu_mib = int(command('nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits').splitlines()[0])
        except (ValueError, IndexError, subprocess.SubprocessError, OSError):
            gpu_mib = None
        try:
            remote_bytes = int(remote('du', '-sb', str(args.remote_progress.rsplit('/', 1)[0])).split()[0])
        except (ValueError, IndexError, subprocess.SubprocessError, OSError):
            remote_bytes = None
        heartbeat = {
            'run_id': args.run_id, 'monitor_pid': os.getpid(), 'sim_host_pid': args.sim_host_pid,
            'simulator_pid': sim_pid, 'process_started_monotonic_ns': progress.get('process_started_monotonic_ns') if progress else None,
            'monotonic_ns': now, 'wall_ns': time.time_ns(), 'current_global_step': last_step,
            'current_episode': progress.get('current_episode') if progress else None,
            'episode_local_step': progress.get('episode_local_step') if progress else None,
            'target_global_step': args.target_step, 'target_episode': args.target_episode,
            'last_completed_simulator_step': last_step, 'last_progress_monotonic_ns': last_advance_ns,
            'last_reset_monotonic_ns': progress.get('last_reset_monotonic_ns') if progress else None,
            'simulator_logical_time_sec': progress.get('simulator_logical_time_sec') if progress else None,
            'elapsed_wall_sec': (now - started_ns) / 1e9,
            'simulator_process_cpu_ns': progress.get('process_cpu_ns') if progress else None,
            'rolling_step_median_ns': progress.get('rolling_step_median_ns') if progress else None,
            'rolling_step_p95_ns': progress.get('rolling_step_p95_ns') if progress else None,
            'rolling_step_max_ns': progress.get('rolling_step_max_ns') if progress else None,
            'docker_process_alive': alive(args.sim_host_pid), 'simulator_alive': remote_alive(sim_pid) if sim_pid else None,
            'bridge_alive': bool(process_matches('eat_nav_command_delivery_bridge.py')),
            'follower_alive': bool(process_matches('simple_trajectory_follower_node')),
            'router_alive': bool(process_matches('/rmw_zenohd')),
            'cpu_rss_bytes': rss_kib * 1024 if rss_kib is not None else None,
            'cgroup_memory_bytes': cgroup_bytes, 'gpu_memory_mib': gpu_mib,
            'output_bytes': bytes_total(args.run) + (remote_bytes or 0),
        }
        atomic(args.run / 'timing-heartbeat.json', heartbeat)
        atomic(args.run.parent.parent / 'timing-heartbeat.json', heartbeat)
        last_heartbeat_ns = now
        reason = None
        if alive(args.sim_host_pid) and (now - last_advance_ns) >= 120 * 1e9:
            reason = 'no completed simulator step for 120 seconds'
        elif gpu_mib is not None and gpu_mib > 6656:
            reason = 'GPU memory exceeded 6.5 GiB'
        elif cgroup_bytes is not None and cgroup_bytes > args.cpu_memory_cap_bytes:
            reason = 'cgroup memory exceeded frozen 8 GiB cap'
        elif not heartbeat['router_alive']:
            reason = 'router process stopped'
        elif progress and progress.get('simulator_logical_time_sec') is not None:
            value = progress['simulator_logical_time_sec']
            if not isinstance(value, (int, float)) or not (-1e12 < value < 1e12):
                reason = 'nonfinite simulator logical time'
        if reason:
            capture_failure(args.run, reason, heartbeat)
            raise RuntimeError(reason)
        if (args.run / 'monitor.stop').exists() or not alive(args.sim_host_pid):
            atomic(args.run / 'timing-monitor.done.json', {'status': 'stopped', 'last_step': last_step,
                                                            'last_heartbeat_monotonic_ns': last_heartbeat_ns})
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
