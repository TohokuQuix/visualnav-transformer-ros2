"""Exact child-process ROS and simulator imports with the shared environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from eat_nav_g0e_gpu_probe import atomic, sha
from eat_nav_ros_subprocess_env import (build_container_env,build_ros_host_env,
    docker_pythonpath_arg,env_audit)


REPO=Path(__file__).resolve().parents[1]
BUILDER=REPO/'scripts/eat_nav_ros_subprocess_env.py'
CHILD_CODE=(
    'import importlib,json,os,sys; '
    'names=sys.argv[1:]; '
    'modules={name:importlib.import_module(name).__file__ for name in names}; '
    'print(json.dumps({"executable":sys.executable,"sys_path":sys.path,'
    '"modules":modules,"env":{k:os.environ.get(k) for k in '
    '["PYTHONPATH","RMW_IMPLEMENTATION","ROS_DOMAIN_ID","AMENT_PREFIX_PATH",'
    '"LD_LIBRARY_PATH","ZENOH_CONFIG_OVERRIDE","ROS_DISTRO"]}},sort_keys=True))')


def run(output,*,root):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    host_env=build_ros_host_env(repo=REPO)
    host_modules=('rclpy','visualnav_transformer','eat_nav_command_delivery_bridge',
                  'eat_nav_exact_stamp_bridge','geometry_msgs.msg','nav_msgs.msg',
                  'rosgraph_msgs.msg','tf2_ros')
    host_command=['/usr/bin/python3','-c',CHILD_CODE,*host_modules]
    host=subprocess.run(host_command,env=host_env,capture_output=True,text=True)
    container_env=build_container_env(repo=REPO,extra_paths=(
        root,'/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src',
        '/tmp/eatnav-train','/tmp/eatnav-train/src',
        '/tmp/eatnav-train/upstream-src','/tmp/eatnav-extended/isaac-source'))
    container_modules=('visualnav_transformer','torch','gymnasium')
    container_command=['docker','exec','-e',docker_pythonpath_arg(container_env),
        'isaac-lab-onix','/isaac-sim/python.sh','-c',CHILD_CODE,*container_modules]
    container=subprocess.run(container_command,capture_output=True,text=True)
    result={'builder_sha256':sha(BUILDER),'host':{'command':host_command,
        'exit_code':host.returncode,'stdout':host.stdout,'stderr':host.stderr,
        'env':env_audit(host_env,host_command[0])},
        'container':{'command':container_command,'exit_code':container.returncode,
        'stdout':container.stdout,'stderr':container.stderr,
        'env':env_audit(container_env,'/isaac-sim/python.sh')}}
    atomic(output,result)
    if host.returncode or container.returncode:
        raise RuntimeError('exact subprocess import preflight failed')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args();run(args.output,root=args.root)
