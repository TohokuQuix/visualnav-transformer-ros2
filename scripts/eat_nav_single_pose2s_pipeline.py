"""Resumable host-side headless collection; never launches path selection implicitly."""
from __future__ import annotations
import argparse,json,subprocess,time,uuid,os,shutil,fcntl,functools
from pathlib import Path
from visualnav_transformer.eat_nav.single_pose2s_contract import make_manifest
from visualnav_transformer.eat_nav.single_pose2s_collection import pending_parent_batches

REPO=Path(__file__).resolve().parents[1]
CONTAINER='isaac-lab-onix'


def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)



def active_collection_workers(proc_root=Path('/proc')):
    result=[]
    for proc in Path(proc_root).iterdir():
        if not proc.name.isdigit():continue
        try:
            args=(proc/'cmdline').read_bytes().split(b'\0')
            if args and args[0]==b'/isaac-sim/kit/python/bin/python3' and any(any(name in x for name in [b'eat_nav_single_pose2s_collect_sim.py',b'eat_nav_single_pose2s_select_sim.py']) for x in args[1:]):result.append(int(proc.name))
        except (OSError,ProcessLookupError):pass
    return sorted(result)


def serialized_gpu(function):
    @functools.wraps(function)
    def wrapped(*args,**kwargs):
        with Path('/tmp/eatnav-single-pose2s-gpu-collection.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            start=time.monotonic()
            while active_collection_workers():
                if time.monotonic()-start>2400:raise RuntimeError('another collector still active; not starting overlapping simulator')
                time.sleep(2)
            return function(*args,**kwargs)
    return wrapped

def stage_source(root):
    path='/tmp/eatnav-single-dem-pose2s-'+uuid.uuid4().hex[:10]
    subprocess.run(['docker','exec',CONTAINER,'mkdir','-p',path],check=True)
    for source,name in [(REPO/'src','src'),(REPO/'scripts','scripts'),(REPO/'checkpoints/eat_nav/rl_actor/model_10000.pt','actor.pt'),(REPO/'checkpoints/eat_nav/rl_actor/env.yaml','env.yaml'),(REPO/'config/eat_nav.yaml','eat_nav.yaml')]:
        subprocess.run(['docker','cp',str(source),CONTAINER+':'+path+'/'+name],check=True,stdout=subprocess.DEVNULL)
    import hashlib
    sources={}
    for source in list((REPO/'src/visualnav_transformer/eat_nav').glob('*.py'))+list((REPO/'scripts').glob('eat_nav_single_pose2s*.py')):
        sources[str(source.relative_to(REPO))]=hashlib.sha256(source.read_bytes()).hexdigest()
    record=dict(container=CONTAINER,path=path,created_unix=time.time(),source_sha256=sources)
    stage_id=path.rsplit('/',1)[-1];immutable=root/'runtime-stages'/f'{stage_id}.json'
    if immutable.exists():raise FileExistsError(immutable)
    write(immutable,record)
    write(root/'runtime-stage.json',record)  # latest pointer; immutable records retained above

    return path


@serialized_gpu
def run_batch(root,stage,parents,tag,record_lidar=False):
    target=root/'raw'/tag
    if (target/'complete.json').exists():return json.loads((target/'complete.json').read_text())
    if target.exists():raise ValueError('existing incomplete output requires audit: '+str(target))
    group=root/'runtime'/f'{tag}.json';write(group,dict(parents=parents))
    subprocess.run(['docker','cp',str(group),CONTAINER+':'+stage+'/'+tag+'.json'],check=True,stdout=subprocess.DEVNULL)
    out=stage+'/'+tag
    command=['docker','exec','-e','PYTHONPATH='+stage+'/src:'+stage+'/scripts:/workspace/isaaclab_onix/source/onix_isaac_lab:/workspace/isaaclab/source/isaaclab:/workspace/isaaclab/source/isaaclab_rl:/workspace/isaaclab/source/isaaclab_tasks',CONTAINER,'timeout','--signal=INT','--kill-after=30s','1800s','/isaac-sim/python.sh',stage+'/scripts/eat_nav_single_pose2s_collect_sim.py','--headless','--parents',stage+'/'+tag+'.json','--output',out,'--rl-checkpoint',stage+'/actor.pt','--actor-env-yaml',stage+'/env.yaml','--eat-config',stage+'/eat_nav.yaml']
    if record_lidar:command.append('--record-lidar')
    write(root/'runtime'/f'{tag}-command.json',dict(command=command,parents=[p['parent_id'] for p in parents],stage_manifest=str(root/'runtime-stages'/(stage.rsplit('/',1)[-1]+'.json'))))
    start=time.time();print('START '+tag,flush=True)
    log=root/'logs'/f'{tag}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('x') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    target.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['docker','cp',CONTAINER+':'+out,str(target)],check=True,stdout=subprocess.DEVNULL)
    if result.returncode or not (target/'complete.json').exists():raise RuntimeError(f'collector failed {tag}, returncode={result.returncode}; see {log}')
    identity=json.loads((target/'runtime-imports.json').read_text())['canonical_sha256']
    identity_path=root/'collection-runtime-identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())['canonical_sha256']!=identity:raise ValueError('runtime source identity changed; preserve batch and audit before continuing')
    else:write(identity_path,dict(canonical_sha256=identity,first_batch=tag))
    complete=json.loads((target/'complete.json').read_text());complete['host_wall_seconds']=time.time()-start;complete['host_valid_windows_per_second']=complete['valid_windows']/complete['host_wall_seconds'];write(target/'complete.json',complete)
    print('DONE '+tag+' '+json.dumps({k:complete[k] for k in ['valid_windows','host_wall_seconds','host_valid_windows_per_second']}),flush=True)
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--mode',choices=['prepare','benchmark','collect'],required=True);parser.add_argument('--num-envs',type=int,default=None);parser.add_argument('--record-lidar',action='store_true');parser.add_argument('--benchmark-repeats',type=int,nargs=3,default=[2,2,2]);a=parser.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json'
    if not manifest_path.exists():write(manifest_path,make_manifest(123))
    manifest=json.loads(manifest_path.read_text())
    if a.mode=='prepare':print(manifest_path);return
    stage=stage_source(root)
    if a.mode=='benchmark':
        selected=[p for p in manifest['parents'] if p['role']=='train' and p['field']=='k_rail_field' and p['level']==4]
        results=[]
        for n,repeats in zip([2,5,10],a.benchmark_repeats):
            for repeat in range(repeats):
                parents=[dict(p,parent_id='benchmark-'+p['parent_id'],role='benchmark') for p in selected[:n]]
                r=run_batch(root,stage,parents,f'benchmark-n{n}-r{repeat}');results.append(dict(num_envs=n,repeat=repeat,**r))
        rates={n:sum(r['host_valid_windows_per_second'] for r in results if r['num_envs']==n)/sum(r['num_envs']==n for r in results) for n in [2,5,10]}
        choice=max(rates,key=rates.get)
        write(root/'benchmark.json',dict(state='completed',results=results,mean_rate=rates,selected_num_envs=choice,physics_dt=.005,control_dt=.05,horizon_s=2.))
        print('BENCHMARK '+str(choice),flush=True);return
    if manifest.get('evaluation_terrain_policy')!='known_terrain_independent_runs':raise ValueError('explicit terrain policy approval required before production')
    benchmark=root/'benchmark.json'
    if not benchmark.exists():raise ValueError('benchmark must complete before production')
    n=a.num_envs if a.num_envs is not None else json.loads(benchmark.read_text())['selected_num_envs']
    if n not in [2,5,10,20,30]:raise ValueError('unsupported measured environment count')
    parents=manifest['parents'];completed=set()
    import hashlib
    expected={p['parent_id']:p for p in parents}
    for complete in (root/'raw').glob('*/complete.json'):
        for path in complete.parent.glob('*/metadata.json'):
            m=json.loads(path.read_text());pid=m['parent']['parent_id']
            if pid not in expected:continue
            if pid in completed:raise ValueError('duplicate completed parent '+pid)
            if m['parent']!=expected[pid] or m['profile_sha256']!=manifest['profile_sha256']:raise ValueError('completed parent/collection profile mismatch')
            raw=path.with_name('raw.npz')
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=m['raw_sha256']:raise ValueError('completed raw hash mismatch')
            completed.add(pid)
    write(root/'batch-policy.json',dict(max_environments=n,combine_roles_within_cell=True,roles_preserved=True,completed_parents=len(completed),pending_parents=len(parents)-len(completed)))
    for rows in pending_parent_batches(manifest,completed,n):
        digest=hashlib.sha256(json.dumps([p['parent_id'] for p in rows]).encode()).hexdigest()[:12]
        tag=f"mixed-{rows[0]['field']}-l{rows[0]['level']}-n{len(rows)}-{digest}"
        run_batch(root,stage,rows,tag,record_lidar=a.record_lidar)
    write(root/'collection-complete.json',dict(state='completed',planned_parents=len(parents),planned_windows=len(parents)*5,finished_unix=time.time(),selected_num_envs=n))

if __name__=='__main__':main()
