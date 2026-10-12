"""Fail-closed mapping from one frozen ROS-free attempt to its shared output."""

from __future__ import annotations

from pathlib import Path


SCHEMA = 'eatnav_p1_g3_rosfree_direct_core_v2'
PROCESS_ID = 'batch-first-differential-0-period-0'
HOST_BIND = '/home/kojima/lib/onix_isaac_lab'
CONTAINER_BIND = '/workspace/isaaclab_onix'


def _clean_absolute(path, label):
    path = Path(path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError(label + ' path traversal or relative path')
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(label + ' symlink escape')
    return path


def _identity(manifest, expected_bind):
    if (manifest.get('schema') != SCHEMA or
            manifest.get('runtime_authorized') is not True or
            manifest.get('authorized_process_ids') != [PROCESS_ID] or
            manifest.get('automatic_continuation') is not False):
        raise ValueError('runtime authorization or protocol mismatch')
    attempt = manifest.get('attempt_id')
    if not isinstance(attempt, str) or not attempt.startswith(
            '.eatnav-p1-g3-rosfree-direct-core-v2-attempt-') or '/' in attempt or attempt in ('.','..'):
        raise ValueError('attempt identity mismatch')
    host_bind, container_bind = expected_bind
    if (manifest.get('host_bind_source') != host_bind or
            manifest.get('container_bind_destination') != container_bind or
            manifest.get('attempt_host_root') != str(Path(host_bind)/attempt) or
            manifest.get('attempt_container_root') != str(Path(container_bind)/attempt)):
        raise ValueError('attempt identity or bind mapping mismatch')
    return attempt


def mount_points_from_proc(text: str) -> set[str]:
    """Read the exact mount-point field; no source inference from container paths."""
    points = set()
    for line in text.splitlines():
        prefix = line.split(' - ', 1)[0].split()
        if len(prefix) >= 5:
            points.add(prefix[4].replace('\\040', ' '))
    return points


def validate_simulator_paths(manifest, input_dir, output_dir, source_path,
                             mount_points, *, expected_bind=(HOST_BIND, CONTAINER_BIND)):
    attempt = _identity(manifest, expected_bind)
    input_dir = _clean_absolute(input_dir, 'input')
    output_dir = _clean_absolute(output_dir, 'output')
    source_path = _clean_absolute(source_path, 'source')
    container_root = Path(manifest['attempt_container_root'])
    batch = container_root/PROCESS_ID
    if (input_dir != batch/'input' or output_dir != batch/'output' or
            source_path != batch/'source.py'):
        raise ValueError('output/input/source differs from frozen attempt root')
    if manifest['container_bind_destination'] not in mount_points:
        raise ValueError('container bind mount missing')
    if (not input_dir.is_dir() or not source_path.is_file() or
            (batch/'host-seal.json').exists() or
            (container_root/'host-seal.json').exists()):
        raise ValueError('missing source/input or sealed evidence root')
    if output_dir.exists():
        raise ValueError('output directory already exists; fresh attempt required')
    return {'attempt_id': attempt, 'process_id': PROCESS_ID,
            'container_root': str(container_root),
            'bind_destination': manifest['container_bind_destination'],
            'bind_mapping_verified': True}


def validate_host_bind(inspect, manifest, host_batch, container_batch,
                       *, expected_bind=(HOST_BIND, CONTAINER_BIND)):
    attempt = _identity(manifest, expected_bind)
    host_batch = _clean_absolute(host_batch, 'host batch')
    container_batch = _clean_absolute(container_batch, 'container batch')
    if (host_batch != Path(manifest['attempt_host_root'])/PROCESS_ID or
            container_batch != Path(manifest['attempt_container_root'])/PROCESS_ID):
        raise ValueError('host/container attempt bind root mismatch')
    matches = [mount for mount in inspect.get('Mounts', ()) if
               mount.get('Type') == 'bind' and mount.get('RW') is True and
               mount.get('Source') == manifest['host_bind_source'] and
               mount.get('Destination') == manifest['container_bind_destination']]
    if len(matches) != 1 or not host_batch.is_dir():
        raise ValueError('host/container bind mapping missing or ambiguous')
    return {'attempt_id': attempt, 'host_batch': str(host_batch),
            'container_batch': str(container_batch),
            'bind_mapping_verified': True}
