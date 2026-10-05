"""One ROS/Python environment contract for expanded E1 subprocesses."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess


REPO = Path(__file__).resolve().parents[1]
ROS_SETUP = '/opt/ros/jazzy/setup.bash'
AUDIT_KEYS = ('PYTHONPATH','RMW_IMPLEMENTATION','ROS_DOMAIN_ID','AMENT_PREFIX_PATH',
              'LD_LIBRARY_PATH','ZENOH_CONFIG_OVERRIDE','ROS_DISTRO')


def _parts(value):
    return [part for part in value.split(os.pathsep) if part]


def prepend_unique(front, existing):
    result = []
    for part in [*map(str, front), *_parts(existing)]:
        if part and part not in result:
            result.append(part)
    return os.pathsep.join(result)


def build_env(base, *, repo=REPO, extra_paths=()):
    env = dict(base)
    env['PYTHONPATH'] = prepend_unique(
        [Path(repo)/'src',Path(repo)/'scripts',*extra_paths],
        env.get('PYTHONPATH',''))
    return env


def _parse_null_env(raw):
    result = {}
    for item in raw.split(b'\0'):
        if item:
            key, value = item.split(b'=',1)
            result[key.decode()] = value.decode()
    return result


def build_ros_host_env(*, repo=REPO, extra_paths=()):
    """Source the existing ROS setup, then prepend project paths without losing ROS."""
    result = subprocess.run(['bash','-lc',f'source {ROS_SETUP} && env -0'],
                            env=os.environ.copy(),capture_output=True,check=True)
    return build_env(_parse_null_env(result.stdout),repo=repo,extra_paths=extra_paths)


def build_container_env(*, repo=REPO, extra_paths=(), container='isaac-lab-onix'):
    """Preserve the running container's Python path before Docker exec -e."""
    result = subprocess.run(['docker','exec',container,'/usr/bin/env','-0'],
                            capture_output=True,check=True)
    return build_env(_parse_null_env(result.stdout),repo=repo,extra_paths=extra_paths)


def env_audit(env, executable):
    return {'python_executable':str(executable),
            'variables':{key:env.get(key) for key in AUDIT_KEYS}}


def docker_pythonpath_arg(env):
    return 'PYTHONPATH=' + env['PYTHONPATH']
