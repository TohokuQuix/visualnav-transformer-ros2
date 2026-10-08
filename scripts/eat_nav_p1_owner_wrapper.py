"""Standard-library-only owner handshake before Isaac Sim imports or GPU use."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time


SCHEMA = 'eatnav_p1_owner_v1'
MATCH_FIELDS = ('schema', 'run_token', 'container_pid', 'start_ticks',
                'output_root', 'source_sha256', 'config_sha256')


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic(path: Path, obj: dict) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    if json.loads(path.read_text()) != obj:
        raise RuntimeError('atomic_marker_readback_mismatch')


def self_identity(args: argparse.Namespace) -> dict:
    stat = Path('/proc/self/stat').read_text()
    fields = stat[stat.rfind(')') + 2:].split()
    status = Path('/proc/self/status').read_text().splitlines()
    nspid_line = next(line for line in status if line.startswith('NSpid:'))
    nspid = [int(value) for value in nspid_line.split(':', 1)[1].split()]
    return {'schema': SCHEMA, 'run_token': args.run_token, 'container_id': args.container_id,
            'container_pid': os.getpid(), 'start_ticks': int(fields[19]), 'nspid': nspid,
            'cgroup': Path('/proc/self/cgroup').read_text().strip(), 'argv': sys.argv,
            'source_path': str(args.source), 'source_sha256': args.source_sha,
            'config_path': str(args.config), 'config_sha256': args.config_sha,
            'output_root': str(args.output), 'wall_ns': time.time_ns()}


def validate_authorization(marker: dict, identity: dict) -> dict:
    if any(marker.get(key) != identity.get(key) for key in MATCH_FIELDS):
        raise ValueError('owner_authorization_identity_mismatch')
    if marker.get('monitor_ready') is not True or not isinstance(marker.get('host_pid'), int):
        raise ValueError('monitor_not_ready')
    return marker


def wait_for_authorization(coord: Path, identity: dict, timeout_sec: float,
                           poll_sec: float = 0.05) -> dict:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if (coord / 'owner-rejected.json').exists():
            raise RuntimeError('owner_rejected')
        auth = coord / 'owner-authorized.json'
        if auth.exists():
            return validate_authorization(json.loads(auth.read_text()), identity)
        atomic(coord / 'heartbeat.json', {'run_token': identity['run_token'],
               'phase': 'waiting_for_authorization', 'monotonic_ns': time.monotonic_ns()})
        time.sleep(poll_sec)
    raise TimeoutError('owner_authorization_timeout')


def forbidden_pre_auth_imports(source: str) -> list[str]:
    """Static guard: top-level imports cannot bring in GPU/Isaac libraries."""
    tree = ast.parse(source)
    forbidden = ('torch', 'isaaclab', 'isaacsim', 'omni', 'pxr', 'pynvml', 'cuda')
    names = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names if alias.name.startswith(forbidden)]
        elif isinstance(node, ast.ImportFrom) and (node.module or '').startswith(forbidden):
            names.append(node.module)
    return names


def preflight(coord: Path, identity: dict) -> None:
    # This is deliberately the first Isaac import in this process.
    from isaaclab.app import AppLauncher
    app = None
    try:
        app = AppLauncher(headless=True).app
        if not hasattr(app, 'close'):
            raise RuntimeError('SimulationApp_close_API_missing')
        atomic(coord / 'completion.json', {'schema': 'eatnav_p1_terminal_marker_v1',
               'run_token': identity['run_token'], 'container_pid': identity['container_pid'],
               'host_pid': identity['host_pid'], 'start_ticks': identity['start_ticks'],
               'source_sha256': identity['source_sha256'],
               'config_sha256': identity['config_sha256'],
               'output_root': identity['output_root'], 'completion_state': 'complete',
               'physics_steps': 0, 'control_steps': 0,
               'written_wall_ns': time.time_ns(), 'marker_before_close': True})
    except BaseException as exc:
        atomic(coord / 'failure.json', {'schema': SCHEMA, 'run_token': identity['run_token'],
               'error_type': type(exc).__name__, 'message': str(exc), 'marker_before_close': True})
        raise
    finally:
        if app is not None:
            app.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--coord', type=Path, required=True)
    parser.add_argument('--run-token', required=True)
    parser.add_argument('--container-id', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('preflight', 'smoke'), required=True)
    parser.add_argument('--authorization-timeout', type=float, default=45)
    parser.add_argument('sim_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if len(args.run_token) != 64 or any(c not in '0123456789abcdef' for c in args.run_token):
        raise ValueError('invalid_run_token')
    if sha(args.source) != args.source_sha or sha(args.config) != args.config_sha:
        raise ValueError('frozen_source_or_config_SHA_mismatch')
    if args.output.exists():
        raise ValueError('simulator_output_must_not_exist')
    if not args.coord.is_dir():
        raise ValueError('coordination_dir_missing')
    if any((args.coord / name).exists() for name in ('owner-ready.json', 'owner-authorized.json',
                                                     'owner-rejected.json', 'completion.json', 'failure.json')):
        raise ValueError('stale_coordination_marker')
    intent = json.loads((args.coord / 'launch-intent.json').read_text())
    if intent.get('run_token') != args.run_token or intent.get('output_root') != str(args.output):
        raise ValueError('launch_intent_mismatch')
    identity = self_identity(args)
    atomic(args.coord / 'owner-ready.json', identity)
    try:
        authorization = wait_for_authorization(args.coord, identity, args.authorization_timeout)
    except BaseException as exc:
        atomic(args.coord / 'failure.json', {'schema': SCHEMA, 'run_token': args.run_token,
               'error_type': type(exc).__name__, 'message': str(exc), 'preauthorization': True})
        raise
    if args.mode == 'preflight':
        preflight(args.coord, {**identity, 'host_pid': authorization['host_pid']})
        return 0
    sim_args = args.sim_args[1:] if args.sim_args[:1] == ['--'] else args.sim_args
    if '--output' not in sim_args or sim_args[sim_args.index('--output') + 1] != str(args.output):
        raise ValueError('frozen_sim_output_mismatch')
    atomic(args.coord / 'exec-transition.json', {'schema': SCHEMA, 'run_token': args.run_token,
           'container_pid': identity['container_pid'], 'start_ticks': identity['start_ticks'],
           'source_sha256': args.source_sha, 'argv': [sys.executable, str(args.source), *sim_args]})
    os.execv(sys.executable, [sys.executable, str(args.source), *sim_args])
    raise AssertionError('exec_returned')


if __name__ == '__main__':
    sys.exit(main())
