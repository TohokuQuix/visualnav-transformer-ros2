#!/usr/bin/env python3
"""One Isaac GUI attempt through separated ROS2 Motion-14 nodes.

The external Zenoh router is observed only. Actuator topics remain mock-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
TRAIN = REPO/'checkpoints/eat_nav/motion14'
CHECKPOINT = TRAIN/'training/full-seed123/best.pt'
HEADING_CHECKPOINT = REPO/'checkpoints/eat_nav/motion14_heading/h2a-seed123/best.pt'
RL = REPO/'checkpoints/eat_nav/rl_actor/model_10000.pt'
RL_ENV = REPO/'checkpoints/eat_nav/rl_actor/env.yaml'
ROS_OVERLAY = Path('/tmp/eatnav-ros2-adapter-build/install-v5/setup.bash')
CONTAINER = 'isaac-lab-onix'
CONTAINER_ONIX = Path('/workspace/isaaclab_onix')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*args):
    subprocess.run(args,check=True)


def verify_inner_seal(root):
    ledger_path = root/'sha-ledger.json'
    seal = json.loads((root/'host-seal.json').read_text())
    ledger = json.loads(ledger_path.read_text())
    if seal['ledger_sha256'] != sha(ledger_path) or seal['file_count'] != len(ledger):
        raise ValueError('inner seal header mismatch')
    for name,digest in ledger.items():
        if sha(root/name) != digest:
            raise ValueError(f'inner seal file mismatch: {name}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(ledger)|{'sha-ledger.json','host-seal.json'}:
        raise ValueError('inner seal file set mismatch')
    return len(ledger)


def _ros_environment():
    if not ROS_OVERLAY.is_file():
        raise FileNotFoundError('ROS interface build missing')
    result=subprocess.run(['bash','-lc',
        f'source /opt/ros/jazzy/setup.bash && source {ROS_OVERLAY} && env -0'],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    env=dict(item.decode().split('=',1) for item in result.stdout.split(b'\0')
             if item and b'=' in item)
    env['RMW_IMPLEMENTATION']='rmw_zenoh_cpp'
    # Use the default ROS domain; never inherit a stale private-domain value
    # from the launcher shell or the ROS overlay.
    env.pop('ROS_DOMAIN_ID',None)
    env['PYTHONPATH']=str(REPO/'src')+':'+str(REPO/'scripts')+':'+env.get('PYTHONPATH','')
    return env


def router_identity(pid_file=None, *, router_pid=None, proc_root=Path('/proc')):
    """Observe one external router; never launch or manage it."""
    pid_file = Path(pid_file) if pid_file else None
    if pid_file is not None:
        pid = int(pid_file.read_text().strip())
    elif router_pid is not None:
        pid = int(router_pid)
    else:
        matches = []
        for proc in proc_root.iterdir():
            if not proc.name.isdigit():
                continue
            try:
                cmd = (proc/'cmdline').read_bytes().split(b'\0')[0]
            except (OSError, PermissionError):
                continue
            if Path(os.fsdecode(cmd)).name == 'rmw_zenohd':
                matches.append(int(proc.name))
        if len(matches) != 1:
            raise ValueError(f'external rmw_zenohd router count {len(matches)}; '
                             'start one router or pass --router-pid-file/--router-pid '
                             '(multiple routers are ambiguous)')
        pid = matches[0]
    proc = proc_root/str(pid)
    try:
        parts = (proc/'cmdline').read_bytes().split(b'\0')
        cmd = os.fsdecode(parts[0])
        stat = (proc/'stat').read_text()
        start_tick = int(stat.rsplit(')', 1)[1].split()[19])
    except (OSError, IndexError, ValueError) as exc:
        raise ValueError(f'external router PID {pid} disappeared or is unreadable') from exc
    if Path(cmd).name != 'rmw_zenohd':
        raise ValueError('external router PID/command mismatch')
    logs = pid_file.parent if pid_file else None
    return {'pid': pid, 'start_tick': start_tick, 'cmdline': ' '.join(
                os.fsdecode(part) for part in parts if part),
            'stdout_sha256': sha(logs/'router.stdout.log') if logs and
                (logs/'router.stdout.log').is_file() else None,
            'stderr_sha256': sha(logs/'router.stderr.log') if logs and
                (logs/'router.stderr.log').is_file() else None}


def resolve_onix_mount(mounts):
    for mount in mounts:
        if mount.get('Type') == 'bind' and mount.get('Destination') == str(CONTAINER_ONIX):
            source = Path(mount['Source'])
            if source.is_dir():
                return source, CONTAINER_ONIX
    raise ValueError(f'{CONTAINER} requires a host bind mount at {CONTAINER_ONIX}')


def resolve_worker_python(explicit, *, home=None):
    home = Path(home) if home else Path.home()
    candidates = ([Path(explicit)] if explicit else [
        home/'anaconda3/envs/tinyvla/bin/python',
        home/'miniconda3/envs/tinyvla/bin/python',
        home/'.conda/envs/tinyvla/bin/python'])
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError('Motion Predictor Python not found; pass --worker-python '
                            'with a Python environment containing torch')


def verify_worker_python(python):
    result = subprocess.run([str(python), '-c', 'import torch; print(torch.__version__)'],
                            capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f'Motion Predictor Python cannot import torch: {python}: '
                           f'{result.stderr.strip()}')


def build_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('--max-steps',type=int,default=300000)
    parser.add_argument('--terrain-type',default='boxes',choices=(
        'k_rail_step_field','pyramid_stairs','pyramid_stairs_inv','random_rough',
        'boxes','hf_pyramid_slope','k_rail_field'))
    parser.add_argument('--terrain-level',type=int,default=0,choices=range(5))
    parser.add_argument('--seed',type=int,default=2026101101)
    parser.add_argument('--port',type=int,default=29563)
    parser.add_argument('--auto-goal-dx',type=float,default=2.)
    parser.add_argument('--auto-goal-dy',type=float,default=0.)
    parser.add_argument('--manual-goal',action='store_true',
                        help='Wait for a PoseStamped goal on /goal_pose')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--router-pid-file',type=Path,
                        help='Optional PID file for the externally managed Zenoh router')
    parser.add_argument('--router-pid',type=int,
                        help='Optional PID when no router PID file exists')
    parser.add_argument('--worker-python',type=Path,
                        help='Python with torch for the Motion Predictor worker')
    parser.add_argument('--scoring-mode',choices=('xy_only','xy_heading'),default='xy_heading')
    parser.add_argument('--heading-reward-weight-m',type=float,default=2.0)
    parser.add_argument('--predictor-zero-velocity-command',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Zero Actor indices 6–8 only for Motion Predictor input')
    return parser


def predictor_command_cli_args(enabled):
    return ['--predictor-zero-velocity-command' if enabled else
            '--no-predictor-zero-velocity-command']


def main():
    parser=build_parser()
    args=parser.parse_args()
    print('Motion Predictor command-zero %s; predictor input only, Actor indices 6–8'
          % ('ON' if args.predictor_zero_velocity_command else 'OFF'), flush=True)
    print('Two-second yaw head enabled for RViz arrows; selection mode %s, heading weight %.3f m'
          % (args.scoring_mode,args.heading_reward_weight_m),flush=True)
    if args.max_steps < 120 or not all(map(lambda x: abs(x)<20,
                                            (args.auto_goal_dx,args.auto_goal_dy))):
        raise ValueError('invalid bounded GUI attempt')
    from visualnav_transformer.eat_nav.ros2_phase2.predictor_client import validate_adoption_artifacts
    from eat_nav_verify_checkpoints import verify as verify_checkpoints
    verify_checkpoints()
    validate_adoption_artifacts(TRAIN/'results.json',CHECKPOINT)
    if not RL.is_file():
        raise FileNotFoundError(RL)
    ros_env=_ros_environment()
    if args.router_pid_file and args.router_pid:
        parser.error('choose only one of --router-pid-file and --router-pid')
    router=router_identity(args.router_pid_file, router_pid=args.router_pid)
    worker_python=resolve_worker_python(args.worker_python)
    verify_worker_python(worker_python)
    mounts=json.loads(subprocess.check_output(
        ['docker','inspect','--format','{{json .Mounts}}',CONTAINER],text=True))
    onix,container_onix=resolve_onix_mount(mounts)
    attempt='eatnav-phase3-gui-'+datetime.now().strftime('%Y%m%d-%H%M%S')
    root=onix/'.eatnav-ros2-phase3-evidence'/attempt
    root.mkdir(parents=True,exist_ok=False)
    stage='/tmp/'+attempt
    staged_sim=stage+'/sim.py'
    files={
      'simulator':REPO/'scripts/eat_nav_phase3_gui_isaac.py',
      'frontend':REPO/'scripts/eat_nav_phase3_gui_frontend.py',
      'launcher':Path(__file__),
      'ros_nodes':REPO/'src/visualnav_transformer/eat_nav/ros2_phase2/nodes.py',
      'ros_display':REPO/'src/visualnav_transformer/eat_nav/ros2_phase2/display_node.py',
      'ros_worker':REPO/'src/visualnav_transformer/eat_nav/ros2_phase2/model_worker_server.py',
      'heading_model':REPO/'src/visualnav_transformer/eat_nav/heading_predictor.py',
      'heading_score':REPO/'src/visualnav_transformer/eat_nav/heading_score.py',
      'evaluation_interface':REPO/'ros2/eat_nav_interfaces/msg/CandidateEvaluation.msg',
      'gui_contract':REPO/'src/visualnav_transformer/eat_nav/ros2_phase3_gui_contract.py',
      'phase2_cli':REPO/'scripts/eat_nav_ros2_phase2.py',
      'checkpoint_manifest':REPO/'checkpoints/eat_nav/manifest.json',
      'rl_env':RL_ENV,
      'checkpoint':CHECKPOINT,'results':TRAIN/'results.json',
      'heading_checkpoint':HEADING_CHECKPOINT,
      'normalization':TRAIN/'normalization.json','rl_checkpoint':RL,
      'eat_config':REPO/'config/eat_nav.yaml',
    }
    manifest={'protocol':'eatnav_ros2_phase3_gui_adopted14_heading_score_v4_bidirectional',
              'attempt':attempt,'source_git_head':subprocess.check_output(
                  ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
              'files_sha256':{key:sha(value) for key,value in files.items()},
              'router_start':router,'staging':stage,
              'host_onix_root':str(onix),'container_onix_root':str(container_onix),
              'worker_python':str(worker_python),
              'max_steps':args.max_steps,'terrain_type':args.terrain_type,
              'terrain_level':args.terrain_level,'seed':args.seed,
              'port':args.port,'auto_goal_delta_map_m':[
                  args.auto_goal_dx,args.auto_goal_dy] if not args.manual_goal else None,
              'headless':args.headless,'num_envs':1,
              'checkpoint_mode':'learned14_adopted',
              'predictor_zero_velocity_command':args.predictor_zero_velocity_command,
              'ros_domain_id':None,
              'selection':('predicted_global_goal_progress' if args.scoring_mode=='xy_only'
                           else 'predicted_global_goal_progress_plus_heading'),
              'scoring_mode':args.scoring_mode,
              'heading_reward_weight_m':args.heading_reward_weight_m,
              'yaw_visualization_only':args.scoring_mode=='xy_only',
              'runtime_authorization':'one GUI closed-loop integration attempt'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    run('docker','exec',CONTAINER,'mkdir','-p',stage+'/src',stage+'/config')
    run('docker','cp',str(REPO/'src/visualnav_transformer'),
        CONTAINER+':'+stage+'/src/')
    for source,target in ((files['simulator'],staged_sim),
                          (files['eat_config'],stage+'/config/eat_nav.yaml'),
                          (RL_ENV,stage+'/config/rl_env.yaml'),
                          (CHECKPOINT,stage+'/config/checkpoint.pt'),
                          (RL,stage+'/config/rl_actor.pt')):
        run('docker','cp',str(source),CONTAINER+':'+target)
    for source,target in ((files['simulator'],staged_sim),
                          (CHECKPOINT,stage+'/config/checkpoint.pt'),
                          (RL_ENV,stage+'/config/rl_env.yaml'),
                          (RL,stage+'/config/rl_actor.pt')):
        digest=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',target],text=True).split()[0]
        if digest != sha(source):
            raise ValueError('host/container staging SHA mismatch')
    output=root/'sim-output'
    sim_cmd=['docker','exec','-e',
       'PYTHONPATH='+stage+'/src:'+str(container_onix)+'/source/onix_isaac_lab:'+
       '/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:'+
       '/workspace/isaaclab/source/isaaclab_tasks',
       CONTAINER,'/isaac-sim/python.sh',staged_sim,
       '--output',str(container_onix/'.eatnav-ros2-phase3-evidence'/attempt/'sim-output'),
       '--checkpoint',stage+'/config/checkpoint.pt','--rl-checkpoint',
       stage+'/config/rl_actor.pt',
       '--eat-config',stage+'/config/eat_nav.yaml',
       '--rl-env-config',stage+'/config/rl_env.yaml', '--seed',str(args.seed),
       '--max-steps',str(args.max_steps),'--replan-steps','10','--port',str(args.port),
       '--num-envs','1','--terrain-type',args.terrain_type,
       '--terrain-level',str(args.terrain_level)]
    if args.headless:
        sim_cmd.append('--headless')
    model_args=['--mode','learned14_adopted','--checkpoint',str(CHECKPOINT),
                '--results',str(TRAIN/'results.json'), '--normalization',
                str(TRAIN/'normalization.json'), '--worker-python',
                str(worker_python),'--sim-time',
                '--scoring-mode',args.scoring_mode,
                '--heading-reward-weight-m',str(args.heading_reward_weight_m),
                *predictor_command_cli_args(args.predictor_zero_velocity_command)]
    commands={
        'inference':['/usr/bin/python3',str(files['phase2_cli']),'inference',*model_args,
                     '--evaluation-period-ns','500000000',
                     '--heading-checkpoint',str(HEADING_CHECKPOINT)],
        'supervisor':['/usr/bin/python3',str(files['phase2_cli']),'supervisor_gui',*model_args],
        'display':['/usr/bin/python3',str(files['phase2_cli']),'display'],
        'frontend':['/usr/bin/python3',str(files['frontend']),'--port',str(args.port)] +
                   ([] if args.manual_goal else ['--auto-goal-dx',str(args.auto_goal_dx),
                        '--auto-goal-dy',str(args.auto_goal_dy)]),
    }
    processes={}
    logs={}
    failure=None
    try:
        logs['simulator']=(root/'simulator.log').open('x')
        processes['simulator']=subprocess.Popen(sim_cmd,stdout=logs['simulator'],stderr=subprocess.STDOUT)
        deadline=time.monotonic()+240
        while not (output/'startup-ready.json').exists():
            if (output/'failure.json').exists() or processes['simulator'].poll() is not None:
                raise RuntimeError('Isaac startup failed; inspect simulator.log and sim-output/failure.json')
            if time.monotonic()>deadline:
                raise TimeoutError('Isaac startup deadline')
            time.sleep(.25)
        for role in ('inference','supervisor','display','frontend'):
            logs[role]=(root/(role+'.log')).open('x')
            processes[role]=subprocess.Popen(commands[role],env=ros_env,
                    stdout=logs[role],stderr=subprocess.STDOUT)
            time.sleep(.3)
            if processes[role].poll() is not None:
                raise RuntimeError(f'{role} preflight exited: {processes[role].returncode}')
        with (root/'resource-series.jsonl').open('x') as resource:
            while processes['simulator'].poll() is None:
                snapshot={'wall_ns':time.time_ns(),
                          'pids':{key:p.pid for key,p in processes.items()},
                          'exits':{key:p.poll() for key,p in processes.items()},
                          'router_live':False}
                try:
                    live=router_identity(args.router_pid_file,router_pid=router['pid'])
                    snapshot['router_live']=live['start_tick']==router['start_tick']
                except ValueError:
                    pass
                gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,memory.total',
                    '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                snapshot['gpu']=gpu.stdout.strip() if gpu.returncode==0 else None
                resource.write(json.dumps(snapshot)+'\n');resource.flush();os.fsync(resource.fileno())
                if any(processes[key].poll() is not None for key in ('inference','supervisor')) or \
                        (processes['frontend'].poll() is not None and
                         not (output/'complete.json').exists()):
                    raise RuntimeError('ROS process exited before simulator')
                if not snapshot['router_live']:
                    raise RuntimeError('external router disappeared')
                time.sleep(.5)
        if processes['simulator'].returncode != 0 or not (output/'complete.json').is_file():
            raise RuntimeError('simulator incomplete/failed')
        inner_count=verify_inner_seal(output)
        trace=[json.loads(line) for line in (output/'trace.jsonl').read_text().splitlines()]
        selections=[line for line in trace if line.get('event')=='ros_selection']
        steps=[line for line in trace if line.get('event')=='step']
        if not selections or len(steps)!=args.max_steps or \
                not any(abs(line['command'][0])>1e-3 for line in steps):
            raise RuntimeError('closed loop lacked selection, full trace, or motion command')
        result={'protocol_valid':True,'step_count':len(steps),
                'ros_selection_count':len(selections),'first_selected':selections[0],
                'nonzero_command_steps':sum(abs(line['command'][0])>1e-3 for line in steps),
                'inner_file_count':inner_count,
                'router_end':router_identity(args.router_pid_file,router_pid=router['pid'])}
        (root/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    except Exception as exc:
        failure={'type':type(exc).__name__,'message':str(exc)}
        (root/'failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    finally:
        for role in ('frontend','display','supervisor','inference','simulator'):
            p=processes.get(role)
            if p is not None and p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.terminate();p.wait(timeout=10)
        for out in logs.values():out.close()
        ledger={str(p.relative_to(root)):sha(p) for p in root.rglob('*')
                if p.is_file() and p.name not in ('host-ledger.json','host-seal.json')}
        (root/'host-ledger.json').write_text(json.dumps(ledger,indent=2,sort_keys=True)+'\n')
        (root/'host-seal.json').write_text(json.dumps({'ledger_sha256':sha(root/'host-ledger.json'),
            'file_count':len(ledger)},indent=2)+'\n')
    print(root)
    if failure:
        raise RuntimeError(f'{failure["message"]}; evidence: {root}')


if __name__=='__main__':
    main()
