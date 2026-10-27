"""Stage immutable sources and open a real Isaac GUI development rollout."""
import argparse
import shlex
import sys
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
from eat_nav_risk_navigation_v2 import REPO,DEFAULT_CHECKPOINT,training_binding
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,require,verify_derived,seal_derived

CONTAINER='isaac-lab-onix'
RUNS=REPO/'docs/eat-nav/risk-guard-gui-20261022'
REFERENCE=REPO/'docs/eat-nav/command-free-gui-20261021'
REFERENCE_LEDGER='36be18f3c0b797f7989c94f31adb08293f6ba6a5bc3789fe5ef5af5093c016b7'
from eat_nav_risk_guard import CONFIG

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


def validate_guard_result(result,trace):
    require(result['selections']>0 and result['control_steps']>0,'no decisions or control trace')
    require(len(trace)==result['control_steps'],'control trace length mismatch')
    stopped=0
    for row in trace:
        values=np.asarray(row['follower_command'],dtype=float)
        require(values.shape==(3,) and np.isfinite(values).all(),'invalid command trace')
        if row['guard_stopped']:
            stopped+=1
            require(row['selected_candidate']=='STOP' and np.all(values==0),'nonzero command during STOP')
        else:require(row['selected_candidate']!='STOP','running state selected STOP')
    require(stopped==result['guard_stop_steps'],'STOP summary mismatch')


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


def reference_conditions():
    verify_derived(REFERENCE)
    require(sha(REFERENCE/'sha-ledger.json')==REFERENCE_LEDGER,'sealed comparison reference mismatch')
    manifest=read(REFERENCE/'attempt01/manifest.json');result=read(REFERENCE/'attempt01/result.json')
    require(result['control_steps']==600 and result['state']=='completed','reference was not the full30sec run')
    return dict(seed=manifest['seed']['base_seed'],steps=manifest['steps'],terrain=manifest['terrain'],
                level=manifest['level'],selection_weights=manifest['selection_weights'],
                initial_pose=read(REFERENCE/'attempt01/sim-output/startup-ready.json')['initial_pose'],
                files_sha256=manifest['files_sha256'])


def run(root,steps=600,hold=60.):
    root=Path(root).resolve()
    require(RUNS in root.parents and not root.exists(),'fresh dedicated GUI output required')
    require(40<=steps<=600 and 0<=hold<=300,'bounded GUI duration required')
    binding=training_binding();CHECKPOINT_SHA=binding['checkpoint_sha256']
    reference=reference_conditions()
    require(steps==reference['steps'],'same-condition comparison requires600 control steps')
    seed=new_seed();actor=REPO/'checkpoints/eat_nav/rl_actor/model_10000.pt'
    require(seed['base_seed']==reference['seed'] and reference['terrain']=='k_rail_step_field' and reference['level']==4
            and reference['selection_weights']==[2.,.5,1.,.5],'comparison conditions mismatch')
    actor_env=REPO/'checkpoints/eat_nav/rl_actor/env.yaml'
    require(sha(actor)=='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d','fixed Actor mismatch')
    stage=Path(tempfile.mkdtemp(prefix='eatnav-risk-gui-'));root.mkdir(parents=True)
    shutil.copytree(REPO/'src/visualnav_transformer',stage/'src/visualnav_transformer',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files={'sim.py':REPO/'scripts/eat_nav_risk_guard_sim.py',
        'eat_nav_risk_guard.py':REPO/'scripts/eat_nav_risk_guard.py',
        'eat_nav_risk_navigation_v2.py':REPO/'scripts/eat_nav_risk_navigation_v2.py',
        'eat_nav_command_free_learning.py':REPO/'scripts/eat_nav_command_free_learning.py',
        'eat_nav_current_learning.py':REPO/'scripts/eat_nav_current_learning.py',
        'eat_nav_risk_navigation.py':REPO/'scripts/eat_nav_risk_navigation.py',
        'checkpoint.pt':DEFAULT_CHECKPOINT,'actor.pt':actor,'env.yaml':actor_env,
        'eat_nav.yaml':REPO/'config/eat_nav.yaml'}
    for name,source in files.items():shutil.copy2(source,stage/name)
    write(stage/'guard-config.json',CONFIG)
    snapshot={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
    for relative,digest in reference['files_sha256'].items():
        if relative!='sim.py':
            require(snapshot.get(relative)==digest,'reference Actor/config/package source changed: '+relative)
    write(root/'manifest.json',dict(schema='eatnav_risk_gui_host_guard_v1',reference_ledger_sha256=REFERENCE_LEDGER,guard_config=CONFIG,
        guard_spec_sha256=sha(REPO/'docs/eat-nav/risk-guard-spec-v0.1.md'),
        training_binding=binding,input_state_dim=28,excluded_actor_indices=[6,7,8],seed=seed,
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
        '--output',str(stage/'output'),'--guard-config',str(stage/'guard-config.json'),
        '--checkpoint',str(stage/'checkpoint.pt'),
        '--checkpoint-sha256',CHECKPOINT_SHA,
        '--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),
        '--eat-config',str(stage/'eat_nav.yaml'),'--seed',str(seed['base_seed']),
        '--control-steps',str(steps),'--hold-seconds',str(hold)]
    write(root/'command.json',argv)
    (root/'simulator-command.sh').write_text(shlex.join(argv)+'\n')
    (root/'host-command.sh').write_text('PYTHONPATH=src:scripts:. '+shlex.join([sys.executable,*sys.argv])+'\n')
    print('SIMULATOR_COMMAND\n'+shlex.join(argv),flush=True)
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
        trace=[json.loads(line) for line in (root/'sim-output/trace.jsonl').read_text().splitlines()]
        validate_guard_result(result,trace)
        require(result['gui'] and not result['headless'] and result['checkpoint_sha256']==CHECKPOINT_SHA,'GUI/model binding mismatch')
        initial=read(root/'sim-output/startup-ready.json')['initial_pose']
        np.testing.assert_allclose(initial,reference['initial_pose'],atol=1e-6,rtol=0)
        result['initial_pose_matches_reference']=True
        with np.load(root/'sim-output/input-0000.npz',allow_pickle=False) as actual, \
                np.load(REFERENCE/'attempt01/sim-output/input-0000.npz',allow_pickle=False) as expected:
            require(set(actual.files)==set(expected.files),'initial input membership mismatch')
            for field in actual.files:np.testing.assert_allclose(actual[field],expected[field],atol=1e-6,rtol=0)
        result['initial_input_matches_reference']=True
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
