"""Run one candidate-free X2 using the shared ROS subprocess environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

from eat_nav_g0e_gpu_probe import atomic, sha
from eat_nav_ros_subprocess_env import build_ros_host_env, env_audit


REPO = Path(__file__).resolve().parents[1]
FOLLOWER = Path('/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower-build/simple_trajectory_follower_node')
PARAMS = Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')
PYTHON = '/usr/bin/python3'


def x2_command(root):
    return [PYTHON,str(REPO/'scripts/eat_nav_e1_router_recovery.py'),
            '--root',str(root),'--follower',str(FOLLOWER),'--params',str(PARAMS)]


def run(root):
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    root.mkdir()
    env=build_ros_host_env(repo=REPO)
    command=x2_command(root)
    atomic(root/'invocation.json',{'command':command,'env':env_audit(env,PYTHON),
        'builder_sha256':sha(REPO/'scripts/eat_nav_ros_subprocess_env.py'),
        'follower_sha256':sha(FOLLOWER),'params_sha256':sha(PARAMS),
        'start_wall_ns':time.time_ns(),'candidate_path_published':False})
    with (root/'x2-runner.log').open('x') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    atomic(root/'x2-return.json',{'exit_code':result.returncode,
        'log_sha256':sha(root/'x2-runner.log'),'readiness_exists':(root/'router-readiness.json').is_file()})
    if result.returncode:raise RuntimeError('X2 candidate-free readiness failed')
    return json.loads((root/'router-readiness.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();print(json.dumps(run(args.root)['gate'],sort_keys=True))
