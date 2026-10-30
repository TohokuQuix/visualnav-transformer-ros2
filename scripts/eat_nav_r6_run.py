"""Frozen R6 cohorts and resumable headless production collector host.

Preparing a manifest never starts Isaac. Running requires successful benchmark
JSON and a separate internal production approval record. Failed attempts remain
on disk and are skipped on restart for explicit investigation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import numpy as np

FIELDS=('k_rail_field','k_rail_step_field','pyramid_stairs','pyramid_stairs_inv','random_rough','boxes','hf_pyramid_slope')
SCENES=('wall','narrow','one_sided','diagonal','approach')
REPO=Path(__file__).resolve().parents[1]
PROTOCOL=REPO/'config/eat_nav/r6/protocol.json'
DEFAULT_ROOT=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/collection')


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def public_exclusions(repo=REPO):
    """Membership metadata only; never traverse reserved-test outcome folders."""
    excluded=set();sources=[];docs=Path(repo)/'docs/eat-nav'
    registries=sorted(docs.glob('fall-stall-*freeze-*/artifacts/seed-registry.json'))
    for path in registries:
        registry=read(path)
        for group in registry['groups']:
            excluded.add(int(group['terrain_seed']))
            for values in group.get('sub_seeds',{}).values():excluded.update(map(int,values))
        for row in registry.get('known_exclusions',[]):excluded.update(map(int,row['base_seeds']))
        sources.append(dict(path=str(path),sha256=sha(path)))
    for path in [docs/'all-fields-learning-20261023/manifest.json',docs/'motion-predictor-vnext/motion_14_collection_manifest.json',
                 docs/'motion-predictor-vnext/mi_a_v3_assignment_manifest.json',docs/'motion-predictor-vnext/mi_a_v31_assignment_manifest.json']:
        if not path.exists():continue
        data=read(path)
        for group in data.get('groups',[]):
            excluded.add(int(group['seed']))
            excluded.update(map(int,group.get('sub_seeds',[])))
        for row in data.get('assignment',[]):
            if row.get('terrain_seed') is not None:excluded.add(int(row['terrain_seed']))
        sources.append(dict(path=str(path),sha256=sha(path)))
    for parent in list(excluded):
        excluded.update(int(np.random.SeedSequence([parent,stream,i]).generate_state(1)[0])
                        for stream in range(3) for i in range(50))
    return excluded,sources


def make_plan(excluded_seeds=(),num_envs=10):
    # num_envs is execution capacity only. Cohort parent width remains ten.
    if num_envs not in (1,2,5,10):raise ValueError('supported collection capacity is 1/2/5/10')
    excluded=set(map(int,excluded_seeds));used=set(excluded);parents=set();collection=[]
    def seed(token):
        nonce=0
        while True:
            value=int.from_bytes(hashlib.sha256(f'eatnav-r6-20261030|{token}|{nonce}'.encode()).digest()[:4],'big')
            children=[int(np.random.SeedSequence([value,s,i]).generate_state(1)[0]) for s in range(3) for i in range(50)]
            # Reserve child streams even for unused eval tiles to avoid collisions.
            if value and value not in used and not set(children)&used and len(set(children))==len(children):
                used.add(value);used.update(children);parents.add(value);return value
            nonce+=1
    def group(parent,role,field,level,tile,cohort,scene='field'):
        return dict(group_id=f'{cohort}/{field}/level{level}/{role}/seed{parent}/tile{tile:02d}',
            role=role,field=field,level=level,seed=parent,terrain_seed=parent,tile_index=tile,
            parent_batch_id=f'{cohort}/{field}/level{level}/{role}/seed{parent}',scene=scene,
            derived_seed_ids=dict(terrain=int(np.random.SeedSequence([parent,0,tile*5+level]).generate_state(1)[0]),
                initial=int(np.random.SeedSequence([parent,2,tile]).generate_state(1)[0]),
                command=int(np.random.SeedSequence([parent,1,tile]).generate_state(1)[0])),
            physical_identity_scope='terrain and root RNG fixed; vector dynamics randomization batch sensitivity must be verified',
            tile_generation_index=tile*5+level)
    for field in FIELDS:
        for level in range(5):
            for role in ('train','validation'):
                parent=seed(f'base/{field}/{level}/{role}')
                collection.append(dict(job_id=f'base-{field}-l{level}-{role}',field=field,level=level,scene='field',
                    seed=parent,terrain_cols=10,groups=[group(parent,role,field,level,i,'base') for i in range(10)]))
    for scene in SCENES:
        for role in ('train','validation'):
            parent=seed(f'supplement/{scene}/{role}')
            collection.append(dict(job_id=f'supplement-{scene}-{role}',field='boxes',level=0,scene=scene,
                seed=parent,terrain_cols=10,groups=[group(parent,role,'boxes',0,i,'supplement-'+scene,scene) for i in range(10)]))
    evaluation={}
    for role in ('validation','test'):
        members=[]
        for field in FIELDS:
            for level in range(5):
                for trial in range(30):
                    parent=seed(f'evaluation/{role}/{field}/{level}/{trial}')
                    members.append(group(parent,role,field,level,0,'closed_loop'))
        evaluation[role]=members
    return dict(schema='eatnav_r6_fixed_cohorts_v1',namespace='eatnav-r6-20261030',collection=collection,
        base_group_count=700,base_branch_count=9800,supplemental_group_count=100,supplemental_branch_count=1400,
        terrain_cols=10,execution_num_envs=num_envs,seed_exclusion_count=len(excluded),
        validation_rollouts=evaluation['validation'],test_rollouts=evaluation['test'],
        validation_conditions=['old_reference','event_2s']+[f'pose_profile_{i:02d}' for i in range(10)]+['combined'],
        validation_condition_runs=35*30*13,test_conditions=['old_reference','event_2s','frozen_top1','frozen_top2','frozen_top3','combined'],
        test_condition_runs=35*30*6,evaluation_status='pending root model/policy integration',
        collection_headless=True,reserved_test_outcomes_opened=False)


def pending_jobs(jobs,status):return [j for j in jobs if status.get(j['job_id']) not in ('completed','failed','running')]


def validate_benchmark(benchmark,approval):
    n=benchmark.get('selected_num_envs')
    if not benchmark.get('passed') or n not in (1,2,5,10):raise ValueError('successful recorded benchmark required')
    if approval.get('production_collection_approved') is not True:raise ValueError('internal production approval required')
    return n


def chunk_jobs(plan,n,blocked_groups=()):
    for parent in plan['collection']:
        available=[g for g in parent['groups'] if g['group_id'] not in blocked_groups]
        for offset in range(0,len(available),n):
            selected=available[offset:offset+n]
            yield dict(parent,job_id=parent['job_id']+'-tiles'+'-'.join(f"{g['tile_index']:02d}" for g in selected),
                groups=[dict(g,env_index=i) for i,g in enumerate(selected)])


def stage_runtime(container,root):
    stage=Path(tempfile.mkdtemp(prefix='eatnav-r6-production-'))
    shutil.copytree(REPO/'src/visualnav_transformer',stage/'src/visualnav_transformer',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for source in (REPO/'scripts').glob('eat_nav*.py'):shutil.copy2(source,stage/source.name)
    for name,source in [('actor.pt',REPO/'checkpoints/eat_nav/rl_actor/model_10000.pt'),('env.yaml',REPO/'checkpoints/eat_nav/rl_actor/env.yaml'),('eat_nav.yaml',REPO/'config/eat_nav.yaml')]:
        shutil.copy2(source,stage/name)
    hashes={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()}
    subprocess.run(['docker','cp',str(stage),container+':/tmp/'],check=True)
    # Verify every staged file, including transitive Python imports, before launch.
    actual=subprocess.check_output(['docker','exec',container,'sha256sum',*[str(stage/name) for name in hashes]],text=True)
    observed={line.split(maxsplit=1)[1].strip():line.split()[0] for line in actual.splitlines()}
    if any(observed.get(str(stage/name))!=digest for name,digest in hashes.items()):raise ValueError('staging hash mismatch')
    binding=dict(stage=str(stage),container=container,source_sha256=hashes,
        image_id=subprocess.check_output(['docker','inspect','--format','{{.Image}}',container],text=True).strip())
    write(Path(root)/'runtime-binding.json',binding);return binding



def validate_collected_group(folder,expected):
    """A missing physical anchor masks this group, not its batch peers."""
    folder=Path(folder);meta=read(folder/'group.json')
    if any(meta.get(k)!=expected.get(k) for k in ('group_id','seed','role')):raise ValueError('collected group identity mismatch')
    available=0
    for slot in range(14):
        branch=folder/f'candidate-{slot:02d}';record=read(branch/'complete.json');trace=branch/'trace.jsonl'
        count=sum(1 for _ in trace.open())
        if record.get('smoke') or record.get('candidate')!=slot or record.get('control_steps')!=count:raise ValueError('candidate completion/trace mismatch')
        if count:available+=1
        elif meta.get('anchor_available',True) or record.get('stop_reason') not in ('termination_during_settling','termination_during_prefix'):
            raise ValueError('empty candidate without explicit unavailable anchor')
    return dict(structurally_complete=True,group_id=meta['group_id'],available_candidates=available,unavailable_candidates=14-available,anchor_available=meta.get('anchor_available',True))

def run_collection(plan,output,benchmark,approval,*,container='isaac-lab-onix',timeout_seconds=1200,max_jobs=None):
    n=validate_benchmark(benchmark,approval);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    planfile=output/'plan.json'
    if planfile.exists() and read(planfile)!=plan:raise ValueError('frozen collection plan changed')
    write(planfile,plan)
    capacity=output/'capacity.json'
    # Tile/RNG identities are fixed in the plan, so smaller later chunks are safe.
    write(capacity,dict(num_envs=n,benchmark=benchmark,approval=approval))
    ledger=read(output/'progress.json') if (output/'progress.json').exists() else {}
    # Interrupted running attempts become preserved failed attempts; no silent overwrite.
    for key,record in ledger.items():
        if record['state']=='running':record.update(state='failed',reason='interrupted_host_attempt_preserved')
    write(output/'progress.json',ledger)
    blocked={g for record in ledger.values() for g in record.get('group_ids',[])}
    jobs=pending_jobs(list(chunk_jobs(plan,n,blocked)),{k:v['state'] for k,v in ledger.items()})
    if max_jobs is not None:jobs=jobs[:max_jobs]
    if not jobs:return ledger
    binding=stage_runtime(container,output);stage=Path(binding['stage'])
    for job in jobs:
        directory=output/job['job_id']
        if directory.exists():raise ValueError('unindexed attempt preserved: '+str(directory))
        directory.mkdir();write(directory/'assignment.json',job)
        groupfile=stage/(job['job_id']+'-groups.json');write(groupfile,dict(groups=job['groups']))
        subprocess.run(['docker','cp',str(groupfile),container+':'+str(groupfile)],check=True)
        simout=stage/'outputs'/job['job_id'];envpath=':'.join([str(stage),str(stage/'src'),'/workspace/isaaclab_onix/source/onix_isaac_lab','/workspace/isaaclab/source/isaaclab','/workspace/isaaclab/source/isaaclab_rl','/workspace/isaaclab/source/isaaclab_tasks'])
        command=['docker','exec','-e','PYTHONPATH='+envpath,container,'timeout','--signal=INT','--kill-after=20s',str(timeout_seconds)+'s','/isaac-sim/python.sh',str(stage/'eat_nav_r6_collect_sim.py'),
            '--headless','--output',str(simout),'--terrain-type',job['field'],'--level',str(job['level']),
            '--scene',job['scene'],'--anchor-seconds',str(job.get('anchor_seconds',0)),'--seed',str(job['seed']),'--num-envs',str(len(job['groups'])),
            '--terrain-cols','10','--tile-indices',','.join(str(g['tile_index']) for g in job['groups']),
            '--groups',str(groupfile),'--rl-checkpoint',str(stage/'actor.pt'),'--actor-env-yaml',str(stage/'env.yaml'),'--eat-config',str(stage/'eat_nav.yaml')]
        write(directory/'command.json',command);ledger[job['job_id']]=dict(state='running',started_unix=time.time(),group_ids=[g['group_id'] for g in job['groups']]);write(output/'progress.json',ledger)
        try:
            with (directory/'simulator.log').open('x') as log:
                result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout_seconds+30)
            code=result.returncode
        except subprocess.TimeoutExpired:code=None
        copied=subprocess.run(['docker','cp',container+':'+str(simout),str(directory/'sim-output')],capture_output=True,text=True)
        complete=directory/'sim-output/complete.json';failure=directory/'sim-output/failure.json'
        success=code==0 and copied.returncode==0 and complete.exists() and not failure.exists() and not read(complete).get('smoke',True)
        group_audits=[]
        if success:
            try:
                group_audits=[validate_collected_group(directory/'sim-output'/f'group-{i:03d}',group) for i,group in enumerate(job['groups'])]
            except (ValueError,FileNotFoundError,KeyError):success=False
        ledger[job['job_id']].update(state='completed' if success else 'failed',returncode=code,ended_unix=time.time(),
            group_audits=group_audits,output_copy_returncode=copied.returncode,complete=read(complete) if complete.exists() else None,
            reason=None if success else 'failed_timeout_or_incomplete_preserved')
        write(output/'progress.json',ledger)
        if code is None:break # docker exec may still be alive; never launch overlapping jobs.
    return ledger


def pack_collection(plan,collection_root,output,reuse_dataset=None):
    """Pack recorded actual candidates; preserve failed/missing groups in audit.

    Missing branches receive no teacher, and missing initial observations cannot
    form model inputs. Their frozen IDs remain in coverage as unavailable.
    """
    from eat_nav_r6_dataset import pack_group,validate_group_roles,validate_input_times
    from visualnav_transformer.eat_nav.r6_targets import TIMES_SEC,TRAJECTORY_LAYOUT
    root=Path(collection_root);output=Path(output)
    if output.exists():raise ValueError('fresh packed output required')
    expected={g['group_id']:g for j in plan['collection'] for g in j['groups']}
    observed={};audit=[];samples=[];index=[]
    for path in sorted(root.glob('*/sim-output/group-*/group.json')):
        meta=read(path);gid=meta['group_id']
        if gid not in expected:raise ValueError('unexpected collected group')
        if gid in observed:raise ValueError('duplicate group attempt; explicit selection required')
        frozen=expected[gid]
        if meta['role']!=frozen['role'] or meta['seed']!=frozen['seed']:raise ValueError('collected/frozen identity mismatch')
        observed[gid]=path.parent
    for gid,frozen in expected.items():
        folder=observed.get(gid)
        if folder is None or not (folder/'initial-input.npz').exists():
            audit.append(dict(group_id=gid,state='unavailable_initial_input',unknown_candidates=14));continue
        with np.load(folder/'initial-input.npz',allow_pickle=False) as data:initial={k:data[k] for k in data.files}
        initial={k:(v[0] if k in ('dem','actor_state','body_state','paths','directions','candidate_mask') else v) for k,v in initial.items()}
        candidates=[]
        for slot in range(14):
            branch=folder/f'candidate-{slot:02d}';trace=branch/'trace.jsonl'
            if not trace.exists():continue
            rows=[json.loads(line) for line in trace.read_text().splitlines()]
            if not rows:continue
            validate_input_times(read(folder/'group.json'),rows[0]['logical_sim_time_ns'])
            candidates.append(dict(slot=slot,path=initial['paths'][slot],direction=float(initial['directions'][slot,0]),rows=rows))
        if not candidates:
            audit.append(dict(group_id=gid,state='no_actual_branch_observation',unknown_candidates=14));continue
        packed,contacts=pack_group(initial,candidates);samples.append(packed)
        index.append(dict(frozen,sample_id=gid,episode_group=frozen['parent_batch_id'],terrain_seed=frozen['seed'],contact_metadata=contacts,source=str(folder)))
        audit.append(dict(group_id=gid,state='included',unknown_candidates=14-len(candidates)))
    if reuse_dataset:
        reuse=Path(reuse_dataset);oldindex=read(reuse/'sample-index.json')
        # Reused seeds were excluded from fresh collection at freeze time.
        if {g['seed'] for g in expected.values()}&{g['seed'] for g in oldindex}:raise ValueError('fresh/reused seed collision')
        with np.load(reuse/'samples.npz',allow_pickle=False) as data:
            for i,member in enumerate(oldindex):
                samples.append({k:data[k][i].copy() for k in data.files});index.append(member)
    if not samples:raise ValueError('no actual data available to pack')
    keys=set(samples[0]);shapes={k:v.shape for k,v in samples[0].items()}
    for sample in samples:
        if set(sample)!=keys or any(sample[k].shape!=shapes[k] or sample[k].dtype!=samples[0][k].dtype for k in keys):raise ValueError('strict merged schema mismatch')
    validate_group_roles(index);output.mkdir(parents=True)
    arrays={k:np.stack([s[k] for s in samples]) for k in sorted(keys)}
    np.savez_compressed(output/'samples.npz',**arrays)
    for i,member in enumerate(index):member['sample_index']=i
    write(output/'sample-index.json',index)
    write(output/'split.json',dict(schema='eatnav_r6_seed_group_split_v1',roles={r:[i for i,m in enumerate(index) if m['role']==r] for r in ('train','validation')},reserved_test_included=False))
    write(output/'collection-pack-audit.json',audit)
    result=dict(schema='eatnav_r6_merged_dataset_v1',samples=len(index),planned_fresh_groups=len(expected),
        included_fresh_groups=sum(a['state']=='included' for a in audit),unavailable_fresh_groups=sum(a['state']!='included' for a in audit),
        event_horizon_sec=2,times_sec=TIMES_SEC.tolist(),trajectory_layout=list(TRAJECTORY_LAYOUT),
        shapes={k:list(v.shape) for k,v in arrays.items()},known_event_targets=int(arrays['event_mask'].sum()),
        valid_trajectory_samples=int(arrays['trajectory_mask'].sum()),files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    write(output/'manifest.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-plan',type=Path);parser.add_argument('--plan',type=Path)
    parser.add_argument('--seed-reference-repo',type=Path,default=Path('/home/ubuntu/visualnav-transformer-ros2'))
    parser.add_argument('--output',type=Path,default=DEFAULT_ROOT);parser.add_argument('--benchmark',type=Path);parser.add_argument('--approval-record',type=Path)
    parser.add_argument('--num-envs',type=int,default=10);parser.add_argument('--max-jobs',type=int)
    parser.add_argument('--pack-output',type=Path);parser.add_argument('--reuse-dataset',type=Path);args=parser.parse_args()
    if args.pack_output:
        if not args.plan:parser.error('packing requires --plan')
        print(json.dumps(pack_collection(read(args.plan),args.output,args.pack_output,args.reuse_dataset),indent=2));return
    if args.prepare_plan:
        excluded,sources=public_exclusions(args.seed_reference_repo);plan=make_plan(excluded,args.num_envs);plan['exclusion_sources']=sources
        plan['seed_reference_repo']=str(args.seed_reference_repo.resolve())
        plan['protocol_sha256']=sha(PROTOCOL);write(args.prepare_plan,plan);print(args.prepare_plan);return
    if not all((args.plan,args.benchmark,args.approval_record)):parser.error('run requires --plan --benchmark --approval-record')
    run_collection(read(args.plan),args.output,read(args.benchmark),read(args.approval_record),max_jobs=args.max_jobs)

if __name__=='__main__':main()
