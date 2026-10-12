"""Read-only external Zenoh router lifecycle observer for P1 G3 v4.

The observer owns only its evidence files. It never signals, starts, or
reconfigures the router. A sample is a point observation, not a guarantee
that no shorter network interruption occurred between samples.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import time


INTERVAL_S = 0.5


def _identity(proc_root: Path, pid: int) -> dict:
    root = proc_root / str(pid)
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    command = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    if fields[0] in ('Z', 'X') or not command:
        raise ValueError('process_not_alive')
    return {'pid': pid, 'parent_pid': int(fields[1]), 'session_id': int(fields[3]),
            'start_tick': int(fields[19]), 'state': fields[0], 'cmdline': command,
            'cgroup': (root / 'cgroup').read_text().strip()}


def _log_bytes(router_dir: Path, proc_root: Path, launcher_pid: int, name: str,
               fd: int) -> tuple[bytes | None, bool, str | None]:
    path = router_dir / name
    fd_path = proc_root / str(launcher_pid) / 'fd' / str(fd)
    try:
        fd_target = os.readlink(fd_path)
    except OSError:
        fd_target = None
    try:
        if stat.S_ISREG(path.stat().st_mode):
            return path.read_bytes(), False, fd_target
    except OSError:
        pass
    try:
        if stat.S_ISREG(fd_path.stat().st_mode):
            return fd_path.read_bytes(), True, fd_target
    except OSError:
        pass
    return None, True, fd_target


def _listener_owned(proc_root: Path, pid: int, port: int) -> bool:
    owned = set()
    try:
        for fd in (proc_root / str(pid) / 'fd').iterdir():
            match = re.fullmatch(r'socket:\[(\d+)\]', os.readlink(fd))
            if match:
                owned.add(match.group(1))
    except OSError:
        return False
    for name in ('tcp', 'tcp6'):
        try:
            lines = (proc_root / 'net' / name).read_text().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) > 9 and parts[1].rsplit(':', 1)[-1].upper() == f'{port:04X}' \
                    and parts[3] == '0A' and parts[9] in owned:
                return True
    return False


def _endpoint_reachable(endpoint: str) -> bool:
    host, port = endpoint.rsplit(':', 1)
    try:
        with socket.create_connection((host, int(port)), timeout=0.2):
            return True
    except OSError:
        return False


def sample_router(router_dir: Path, *, proc_root: Path = Path('/proc'),
                  now_ns=time.time_ns, listener_probe=None,
                  endpoint_probe=None) -> dict:
    """Read one identity/log/listener/endpoint snapshot without router mutation."""
    router_dir, proc_root = Path(router_dir), Path(proc_root)
    row = {'sample_started_wall_ns': now_ns(), 'misses': []}
    try:
        launcher_pid = int((router_dir / 'router-launch.pid').read_text().strip())
        pid = int((router_dir / 'router.pid').read_text().strip())
    except (OSError, ValueError):
        launcher_pid = pid = None
        row['misses'].append('pid_file_missing_or_invalid')
    launcher = router = None
    if launcher_pid is not None:
        try:
            launcher = _identity(proc_root, launcher_pid)
        except (OSError, ValueError, IndexError):
            row['misses'].append('launcher_missing')
    if pid is not None:
        try:
            router = _identity(proc_root, pid)
        except (OSError, ValueError, IndexError):
            row['misses'].append('router_missing')
    row.update({'launcher_pid': launcher_pid if launcher else None,
                'launcher_start_tick': launcher['start_tick'] if launcher else None,
                'launcher_parent_pid': launcher['parent_pid'] if launcher else None,
                'launcher_session_id': launcher['session_id'] if launcher else None,
                'launcher_cgroup': launcher['cgroup'] if launcher else None,
                'launcher_cmdline': launcher['cmdline'] if launcher else None,
                'pid': pid if router else None,
                'start_tick': router['start_tick'] if router else None,
                'router_parent_pid': router['parent_pid'] if router else None,
                'router_session_id': router['session_id'] if router else None,
                'router_cgroup': router['cgroup'] if router else None,
                'router_cmdline': router['cmdline'] if router else None})
    if launcher and router and (router['parent_pid'] != launcher_pid or
                                'rmw_zenohd' not in router['cmdline'] or
                                'ros2 run rmw_zenoh_cpp rmw_zenohd' not in launcher['cmdline']):
        row['misses'].append('process_relationship_mismatch')
    stdout, stdout_deleted, stdout_fd = _log_bytes(
        router_dir, proc_root, launcher_pid or -1, 'router.stdout.log', 1)
    stderr, stderr_deleted, stderr_fd = _log_bytes(
        router_dir, proc_root, launcher_pid or -1, 'router.stderr.log', 2)
    row.update({'stdout_available': stdout is not None,
                'stderr_available': stderr is not None,
                'stdout_path_deleted': stdout_deleted,
                'stderr_path_deleted': stderr_deleted,
                'stdout_fd_target': stdout_fd, 'stderr_fd_target': stderr_fd,
                'stdout_bytes': stdout, 'stderr_bytes': stderr,
                'stdout_sha256': hashlib.sha256(stdout).hexdigest() if stdout is not None else None,
                'stderr_sha256': hashlib.sha256(stderr).hexdigest() if stderr is not None else None,
                'stdout_size': len(stdout) if stdout is not None else None,
                'stderr_size': len(stderr) if stderr is not None else None})
    if stdout is None or stderr is None:
        row['misses'].append('log_unavailable')
    text = (stdout or b'') + b'\n' + (stderr or b'')
    zids = re.findall(rb'Started Zenoh router with id ([0-9a-f]+)', text)
    endpoints = re.findall(rb'Zenoh can be reached at: tcp/([^\s]+:7447)', text)
    row['zid'] = zids[-1].decode() if zids else None
    row['endpoints'] = [item.decode() for item in endpoints]
    if not zids or not endpoints or len(set(zids)) != 1:
        row['misses'].append('zid_or_endpoint_missing')
    probe = endpoint_probe or _endpoint_reachable
    row['reachable_endpoints'] = [endpoint for endpoint in row['endpoints'] if probe(endpoint)]
    if not row['reachable_endpoints']:
        row['misses'].append('endpoint_unreachable')
    listener = listener_probe or (lambda target, port: _listener_owned(proc_root, target, port))
    row['listener_owned'] = bool(router and listener(pid, 7447))
    if not row['listener_owned']:
        row['misses'].append('listener_missing')
    row['sample_finished_wall_ns'] = now_ns()
    return row


def _atomic_bytes(path: Path, data: bytes) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, (json.dumps(value, sort_keys=True) + '\n').encode())


def run_observer(router_dir: Path, output_dir: Path, stop_file: Path, *,
                 proc_root: Path = Path('/proc'), now_ns=time.time_ns,
                 sleeper=time.sleep, listener_probe=None, endpoint_probe=None) -> None:
    """Write fsynced series until the host writes an observer-owned stop marker."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / 'router-series.jsonl').open('x') as stream:
        sequence = 0
        while True:
            row = sample_router(router_dir, proc_root=proc_root, now_ns=now_ns,
                                listener_probe=listener_probe, endpoint_probe=endpoint_probe)
            row['observer_heartbeat_seq'] = sequence
            for label in ('stdout', 'stderr'):
                content = row.pop(label + '_bytes')
                if content is not None:
                    _atomic_bytes(output_dir / f'router-{label}-copy.bin', content)
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush(); os.fsync(stream.fileno())
            if sequence == 0 and not row['misses']:
                _atomic_json(output_dir / 'observer-ready.json',
                             {'first_sample_wall_ns': row['sample_finished_wall_ns'],
                              'pid': row['pid'], 'start_tick': row['start_tick'],
                              'zid': row['zid']})
            sequence += 1
            if Path(stop_file).exists():
                _atomic_json(output_dir / 'observer-complete.json',
                             {'sample_count': sequence, 'final_wall_ns': row['sample_finished_wall_ns']})
                return
            sleeper(INTERVAL_S)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--router-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--stop-file', type=Path, required=True)
    args = parser.parse_args()
    run_observer(args.router_dir, args.output_dir, args.stop_file)


if __name__ == '__main__':
    main()
