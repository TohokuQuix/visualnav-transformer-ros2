"""Fixed-manifest all-field experiment coordinator; all real rollouts have GUI."""
from __future__ import annotations
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys,tempfile,time,traceback
from pathlib import Path
import numpy as np
from eat_nav_all_fields import FIELDS,make_manifest,compare_anchor
from eat_nav_krail_ablation import SEEDS
from eat_nav_krail_ablation_run import REGISTRY,activate_gui
from eat_nav_risk_navigation_v2 import REPO,DEFAULT_CHECKPOINT,training_binding
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,require,verify_derived,seal_derived
CONTAINER='isaac-lab-onix'
ROOT=REPO/'docs/eat-nav/all-fields-learning-20261023'
DEPENDENCIES=['eat_nav_all_fields','eat_nav_krail_ablation','eat_nav_risk_guard','eat_nav_risk_navigation_v2',
              'eat_nav_command_free_learning','eat_nav_current_learning','eat_nav_risk_navigation']

def atomic(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def prepare(root):
    root=Path(root).resolve()
    require(not root.exists(),'fresh experiment root required');verify_derived(REGISTRY.parent);training_binding()
    manifest=make_manifest(read(REGISTRY),SEEDS)
    manifest.update(spec_sha256=sha(REPO/'docs/eat-nav/all-fields-learning-and-evaluator-spec-v0.1.md'),
        registry_sha256=sha(REGISTRY),source_sha256={str(p.relative_to(REPO)):sha(p) for p in
        [REPO/'scripts/eat_nav_all_fields_sim.py',REPO/'scripts/eat_nav_all_fields_run.py']+[REPO/'scripts'/(n+'.py') for n in DEPENDENCIES]})
    root.mkdir(parents=True);write(root/'manifest.json',manifest)
    stage=Path(tempfile.mkdtemp(prefix='eatnav-all-fields-'))
    shutil.copytree(REPO/'src/visualnav_transformer',stage/'src/visualnav_transformer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in DEPENDENCIES:shutil.copy2(REPO/'scripts'/(name+'.py'),stage/(name+'.py'))
    files={'sim.py':REPO/'scripts/eat_nav_all_fields_sim.py','checkpoint.pt':DEFAULT_CHECKPOINT,
        'actor.pt':REPO/'checkpoints/eat_nav/rl_actor/model_10000.pt','env.yaml':REPO/'checkpoints/eat_nav/rl_actor/env.yaml',
        'eat_nav.yaml':REPO/'config/eat_nav.yaml'}
    for name,p in files.items():shutil.copy2(p,stage/name)
    require(sha(stage/'actor.pt')=='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d','actor identity')
    require(sha(stage/'env.yaml')=='c2b7e9b7959a231c5391ed714a4de4be958fe5812cf03c10ab60f5636598233b','actor env identity')
    hashes={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
    subprocess.run(['docker','cp',str(stage),CONTAINER+':/tmp/'],check=True)
    out=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',*[str(stage/p) for p in hashes]],text=True)
    actual={line.split(maxsplit=1)[1].strip():line.split()[0] for line in out.splitlines()}
    require(all(actual[str(stage/p)]==v for p,v in hashes.items()),'staged SHA mismatch')
    # Read-only environment identity, no secret/environment-variable dump.
    image=subprocess.check_output(['docker','inspect','--format','{{.Image}}',CONTAINER],text=True).strip()
    onix='/workspace/isaaclab_onix/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/'
    runtime=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',onix+'mdp/terrains.py',onix+'config/onix/rough_env_cfg.py',
        '/workspace/isaaclab/source/isaaclab/isaaclab/terrains/terrain_generator.py'],text=True)
    write(root/'runtime-binding.json',dict(stage=str(stage),container=CONTAINER,image_id=image,files_sha256=hashes,onix_and_generator_source=runtime,
        checkpoint_sha256=sha(DEFAULT_CHECKPOINT),reserved_test_opened=False))
    (root/'host-command.sh').write_text('PYTHONPATH=src:scripts:. '+shlex.join([sys.executable,*sys.argv])+'\n')
    return manifest

def run_one(root,item,tag,*,model=None):
    root=Path(root);binding=read(root/'runtime-binding.json');stage=Path(binding['stage']);dest=root/tag
    require(not dest.exists(),'fresh attempt required');dest.mkdir(parents=True)
    manifest=read(root/'manifest.json')
    for p,digest in manifest['source_sha256'].items():require(sha(REPO/p)==digest,'execution source changed: '+p)
    model=Path(model) if model is not None else DEFAULT_CHECKPOINT
    model_sha=sha(model);container_model=str(stage/'checkpoint.pt')
    if model.resolve()!=DEFAULT_CHECKPOINT.resolve():
        container_model=str(stage/('checkpoint-'+item['field']+'.pt'))
        subprocess.run(['docker','cp',str(model),CONTAINER+':'+container_model],check=True)
        require(subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',container_model],text=True).split()[0]==model_sha,'new model SHA')
    output=str(stage/'outputs'/tag)
    envpath=':'.join([str(stage),str(stage/'src'),'/workspace/isaaclab_onix/source/onix_isaac_lab',
        '/workspace/isaaclab/source/isaaclab','/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
    argv=['docker','exec','-e','PYTHONPATH='+envpath,CONTAINER,'timeout','--signal=INT','--kill-after=20s','600s',
        '/isaac-sim/python.sh',str(stage/'sim.py'),'--output',output,'--phase',item['phase'],
        '--seed',str(item['seed']),'--terrain-type',item['field'],'--checkpoint',container_model,'--checkpoint-sha256',model_sha,
        '--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),'--eat-config',str(stage/'eat_nav.yaml'),
        '--hold-seconds','1']
    if item['phase']=='collect':argv+=['--anchor-seconds',str(item['anchor']),'--candidate',str(item['candidate'])]
    else:argv+=['--condition',item['condition']]
    write(dest/'command.json',argv);write(dest/'assignment.json',dict(item,checkpoint_sha256=model_sha))
    (dest/'simulator-command.sh').write_text(shlex.join(argv)+'\n');print('SIMULATOR_COMMAND\n'+shlex.join(argv),flush=True)
    code=None
    try:
        with (dest/'simulator.log').open('x') as log:
            proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic();activated=False
            while proc.poll() is None:
                if not activated and '\nSELECTION ' in (dest/'simulator.log').read_text(errors='replace'):
                    status=activate_gui()
                    if status:
                        write(dest/'gui-window.json',status);activated=True;print('GUI_VISIBLE '+tag,flush=True)
                        capture=Path('/tmp/capture-krail-running.py')
                        if capture.exists():
                            snap=subprocess.run([sys.executable,str(capture),str(dest)],capture_output=True,text=True,timeout=10)
                            if snap.returncode:write(dest/'capture-failure.json',dict(stderr=snap.stderr))
                if time.monotonic()-start>640:proc.terminate();proc.wait(timeout=30);raise TimeoutError('container timeout')
                time.sleep(.5)
            code=proc.returncode
        subprocess.run(['docker','cp',CONTAINER+':'+output,str(dest/'sim-output')],check=True)
        verify_derived(dest/'sim-output')
        if code!=0 or (dest/'sim-output/failure.json').exists():raise RuntimeError('simulator failed: '+tag)
        result=read(dest/'sim-output/complete.json');write(dest/'result.json',result)
        print('COMPLETED '+json.dumps(dict(tag=tag,**result)),flush=True)
        return result
    except BaseException as exc:
        write(dest/'host-failure.json',dict(type=type(exc).__name__,message=str(exc),code=code))
        raise
    finally:seal_derived(dest)

def collect(root,limit=None):
    root=Path(root);manifest=read(root/'manifest.json');results=[]
    for i,item in enumerate(manifest['collect'][:limit]):
        tag=f'collect-{i:03d}'
        if (root/tag/'result.json').exists():
            verify_derived(root/tag);result=read(root/tag/'result.json')
        elif (root/tag/'host-failure.json').exists():
            verify_derived(root/tag);result=dict(state='failed',error=read(root/tag/'host-failure.json'))
        elif (root/tag).exists():raise RuntimeError('unfinished attempt requires explicit preserved recovery: '+tag)
        else:
            try:result=run_one(root,item,tag)
            except Exception as exc:result=dict(state='failed',error=str(exc))
        results.append(dict(index=i,folder=tag,assignment=item,result=result))
        atomic(root/'collection-progress.json',dict(done=len(results),planned=len(manifest['collect']),results=results))
    return results

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--collect',action='store_true');p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.prepare:prepare(args.root)
    if args.collect:collect(args.root,args.limit)
if __name__=='__main__':main()
