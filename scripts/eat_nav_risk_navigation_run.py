"""Stage immutable sources and open a real Isaac GUI development rollout."""
import argparse
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import numpy as np
from eat_nav_risk_navigation import REPO,DEFAULT_CHECKPOINT,CHECKPOINT_SHA
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,require,verify_derived,seal_derived

CONTAINER='isaac-lab-onix'
RUNS=REPO/'docs/eat-nav/risk-navigation-gui-20261021'
REGISTRY=REPO/'docs/eat-nav/fall-stall-additional-batch03-freeze-20261021/artifacts/seed-registry.json'


def new_seed():
    verify_derived(REGISTRY.parent);registry=read(REGISTRY)
    excluded={g['terrain_seed'] for g in registry['groups']}
    excluded.update(s for row in registry['known_exclusions'] for s in row['base_seeds'])
    # Historical four-env selected terrain generates 5 rows x 4 columns.
    # L4 terrain uses tile indices16..19, even though robot env IDs are0..3.
    used_sub={int(np.random.SeedSequence([s,stream,i]).generate_state(1)[0])
              for s in excluded for stream in range(3) for i in range(20 if stream==0 else 4)}
    for attempt in range(100):
        token=f'eatnav-risk-gui-development-20261021|{attempt}'
        seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:4],'big')
        sub=[int(np.random.SeedSequence([seed,stream,i]).generate_state(1)[0])
             for stream in range(3) for i in range(5)]
        if seed and seed not in excluded and not set(sub)&used_sub:
            return dict(base_seed=seed,derivation_token=token,sub_seeds=sub,
                        excluded_base_seed_count=len(excluded),excluded_sub_seed_count=len(used_sub),
                        overlap_base_seeds=0,overlap_sub_seeds=0,
                        registry_sha256=sha(REGISTRY),role='development_integration_only')
    raise ValueError('no independent development seed')


def command(*args,**kwargs):return subprocess.run(args,check=True,**kwargs)


def activate_gui():
    """Ask the desktop WM to activate Isaac; XRaiseWindow alone is insufficient."""
    os.environ.setdefault('XAUTHORITY','/run/user/1000/gdm/Xauthority')
    display=os.environ.get('DISPLAY',':1')
    result=subprocess.run(['xwininfo','-display',display,'-root','-tree'],capture_output=True,text=True,timeout=3)
    match=re.search(r'(0x[0-9a-f]+) "Isaac Sim[^\n]+\("IsaacSim"',result.stdout)
    if not match:return None
    c=ctypes
    class Data(c.Union):_fields_=[('l',c.c_long*5)]
    class Message(c.Structure):
        _fields_=[('type',c.c_int),('serial',c.c_ulong),('send_event',c.c_int),('display',c.c_void_p),
                  ('window',c.c_ulong),('message_type',c.c_ulong),('format',c.c_int),('data',Data)]
    class Event(c.Union):_fields_=[('client',Message),('pad',c.c_long*24)]
    x=c.CDLL('libX11.so.6');x.XOpenDisplay.argtypes=[c.c_char_p];x.XOpenDisplay.restype=c.c_void_p
    d=x.XOpenDisplay(display.encode())
    if not d:return None
    x.XDefaultRootWindow.argtypes=[c.c_void_p];x.XDefaultRootWindow.restype=c.c_ulong
    x.XInternAtom.argtypes=[c.c_void_p,c.c_char_p,c.c_int];x.XInternAtom.restype=c.c_ulong
    x.XSendEvent.argtypes=[c.c_void_p,c.c_ulong,c.c_int,c.c_long,c.POINTER(Event)]
    x.XFlush.argtypes=[c.c_void_p];x.XCloseDisplay.argtypes=[c.c_void_p]
    event=Event();event.client.type=33;event.client.display=d;event.client.window=int(match[1],16)
    event.client.message_type=x.XInternAtom(d,b'_NET_ACTIVE_WINDOW',0);event.client.format=32
    event.client.data.l[0]=2
    sent=x.XSendEvent(d,x.XDefaultRootWindow(d),0,(1<<20)|(1<<19),c.byref(event))
    x.XFlush(d);x.XCloseDisplay(d)
    return dict(window=match[1],display=display,wm_activation_requested=bool(sent))


def run(root,steps=600,hold=60.):
    root=Path(root).resolve()
    require(RUNS in root.parents and not root.exists(),'fresh dedicated GUI output required')
    require(40<=steps<=600 and 0<=hold<=300,'bounded GUI duration required')
    verify_derived(DEFAULT_CHECKPOINT.parent.parent)
    require(sha(DEFAULT_CHECKPOINT)==CHECKPOINT_SHA,'risk checkpoint identity mismatch')
    seed=new_seed();actor=REPO/'checkpoints/eat_nav/rl_actor/model_10000.pt'
    actor_env=REPO/'checkpoints/eat_nav/rl_actor/env.yaml'
    require(sha(actor)=='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d','fixed Actor mismatch')
    stage=Path(tempfile.mkdtemp(prefix='eatnav-risk-gui-'));root.mkdir(parents=True)
    shutil.copytree(REPO/'src/visualnav_transformer',stage/'src/visualnav_transformer',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files={'sim.py':REPO/'scripts/eat_nav_risk_navigation_sim.py',
        'eat_nav_risk_navigation.py':REPO/'scripts/eat_nav_risk_navigation.py',
        'checkpoint.pt':DEFAULT_CHECKPOINT,'actor.pt':actor,'env.yaml':actor_env,
        'eat_nav.yaml':REPO/'config/eat_nav.yaml'}
    for name,source in files.items():shutil.copy2(source,stage/name)
    snapshot={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
    write(root/'manifest.json',dict(schema='eatnav_risk_gui_host_v1',seed=seed,
        stage=str(stage),files_sha256=snapshot,host_runner_sha256=sha(__file__),
        checkpoint_sha256=CHECKPOINT_SHA,terrain='k_rail_step_field',level=4,steps=steps,
        hold_seconds=hold,headless=False,display=':1',reserved_test_opened=False,
        selection_weights=[2.,.5,1.,.5],source_package_modified=False,
        started_utc=datetime.now(timezone.utc).isoformat()))
    # Docker copy creates the fresh /tmp stage, never writes the historical mounts.
    command('docker','cp',str(stage),CONTAINER+':/tmp/')
    for relative,digest in snapshot.items():
        result=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',str(stage/relative)],text=True)
        require(result.split()[0]==digest,'staged source SHA mismatch')
    envpath=str(stage) + ':' + str(stage/'src') + ':' + ':'.join([
        '/workspace/isaaclab_onix/source/onix_isaac_lab','/workspace/isaaclab/source/isaaclab',
        '/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
    deadline_seconds=420+steps/20+hold
    # The deadline lives inside the container, supervising the actual process
    # group; killing a local docker client alone would leave Isaac running.
    argv=['docker','exec','-e','PYTHONPATH='+envpath,CONTAINER,
        'timeout','--signal=INT','--kill-after=20s',str(deadline_seconds)+'s',
        '/isaac-sim/python.sh',str(stage/'sim.py'),
        '--output',str(stage/'output'),'--checkpoint',str(stage/'checkpoint.pt'),
        '--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),
        '--eat-config',str(stage/'eat_nav.yaml'),'--seed',str(seed['base_seed']),
        '--control-steps',str(steps),'--hold-seconds',str(hold)]
    write(root/'command.json',argv)
    error=None
    try:
        with (root/'simulator.log').open('x') as log:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            print('STARTED '+json.dumps(dict(root=str(root),pid=process.pid,stage=str(stage),seed=seed['base_seed'])),flush=True)
            try:
                end=time.monotonic()+deadline_seconds+40;activated=False
                while process.poll() is None:
                    if time.monotonic()>end:raise subprocess.TimeoutExpired(argv,deadline_seconds+40)
                    if not activated and '\nSELECTION ' in (root/'simulator.log').read_text(errors='replace'):
                        activation=activate_gui()
                        if activation:
                            write(root/'gui-window.json',activation);activated=True
                            print('GUI_VISIBLE '+json.dumps(activation),flush=True)
                    time.sleep(.5)
                code=process.returncode
            except subprocess.TimeoutExpired:
                process.terminate();process.wait(timeout=30)
                raise TimeoutError('bounded GUI run exceeded deadline')
        command('docker','cp',CONTAINER+':'+str(stage/'output'),str(root/'sim-output'))
        verify_derived(root/'sim-output')
        require(code==0 and not (root/'sim-output/failure.json').exists(),'GUI run failed; inspect simulator.log')
        result=read(root/'sim-output/complete.json')
        require(result['selections']>0 and result['nonzero_command_steps']>0,'no learned selection or motion command')
        require(result['gui'] and not result['headless'] and result['checkpoint_sha256']==CHECKPOINT_SHA,'GUI/model binding mismatch')
        write(root/'result.json',result);print('COMPLETED '+json.dumps(result),flush=True)
        return result
    except BaseException as exc:
        error=exc;write(root/'host-failure.json',dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        # Preserve stage/output and host failure evidence; do not retry automatically.
        seal_derived(root)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--hold-seconds',type=float,default=60.)
    args=parser.parse_args();run(args.output,args.steps,args.hold_seconds)
