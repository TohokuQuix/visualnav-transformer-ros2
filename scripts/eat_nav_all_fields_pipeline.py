"""Execute the approved fixed all-field learning loop, retaining every outcome."""
from __future__ import annotations
import argparse,json,sys,traceback,shutil,subprocess
from pathlib import Path
import torch
from eat_nav_all_fields import FIELDS,compare_anchor
from eat_nav_all_fields_run import ROOT,REPO,CONTAINER,prepare,collect,run_one,atomic
from eat_nav_all_fields_learning import KRAIL_SEEDS,field_dataset,fit_field,npz
from eat_nav_all_fields_audit import audit_evaluation,report,comparison_eligible
from eat_nav_risk_navigation_v2 import DEFAULT_CHECKPOINT
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,verify_derived,seal_derived

def plan(root):
    root=Path(root);manifest=read(root/'manifest.json');jobs=[]
    for field in FIELDS:
        cases=[r for r in manifest['groups'] if r['field']==field and r['role']=='development_evaluation']
        for case in cases:
            order=['R0','R1','R2','R3'];n=case['slot']-8;order=order[n:]+order[:n]
            for condition in order:
                model_seeds=[None] if condition=='R0' else (KRAIL_SEEDS if field=='k_rail_field' else [123])
                for model_seed in model_seeds:
                    jobs.append(dict(field=field,seed=case['seed'],phase='evaluate',condition=condition,model_seed=model_seed))
    assert len(jobs)==129
    paths=[REPO/'scripts'/(n+'.py') for n in ('eat_nav_all_fields_learning','eat_nav_all_fields_audit','eat_nav_all_fields_pipeline','eat_nav_all_fields_eval_sim')]
    p=dict(schema='eatnav_all_fields_learning_and_evaluation_plan_v1',learning_seeds={f:KRAIL_SEEDS if f=='k_rail_field' else [123] for f in FIELDS},
        evaluation=jobs,source_sha256={str(p.relative_to(REPO)):sha(p) for p in paths},collection_manifest_sha256=sha(root/'manifest.json'),
        spec_sha256=sha(REPO/'docs/eat-nav/all-fields-learning-and-evaluator-spec-v0.1.md'),reserved_test_opened=False)
    if (root/'learning-evaluation-plan.json').exists():
        old=read(root/'learning-evaluation-plan.json')
        if old!=p:raise ValueError('frozen learning/evaluation plan changed')
        return old
    write(root/'learning-evaluation-plan.json',p)
    snapshot=root/'executed-learning-source';snapshot.mkdir()
    for path in paths:shutil.copy2(path,snapshot/path.name)
    shutil.copy2(REPO/'tests/eat_nav/test_all_fields.py',snapshot/'test_all_fields.py')
    return p

def evaluation_runtime(root):
    # A separate immutable GUI adapter adds risk HUD without changing collection.
    root=Path(root);dest=root/'evaluation-runtime';source=REPO/'scripts/eat_nav_all_fields_eval_sim.py'
    if (dest/'evaluation-overlay.json').exists():
        if read(dest/'evaluation-overlay.json')['sim_sha256']!=sha(source):raise ValueError('evaluation GUI source changed')
        return dest
    if dest.exists():raise ValueError('incomplete evaluation staging')
    prepare(dest)
    manifest=read(dest/'manifest.json');binding=read(dest/'runtime-binding.json');stage=Path(binding['stage'])
    write(dest/'manifest-before-evaluation-overlay.json',manifest);write(dest/'runtime-before-evaluation-overlay.json',binding)
    shutil.copy2(source,stage/'sim.py')
    subprocess.run(['docker','cp',str(source),CONTAINER+':'+str(stage/'sim.py')],check=True)
    actual=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',str(stage/'sim.py')],text=True).split()[0]
    if actual!=sha(source):raise ValueError('evaluation sim stage SHA mismatch')
    manifest['source_sha256'][str(source.relative_to(REPO))]=actual
    manifest['phase']='evaluation_only';binding['files_sha256']['sim.py']=actual
    atomic(dest/'manifest.json',manifest);atomic(dest/'runtime-binding.json',binding)
    write(dest/'evaluation-overlay.json',dict(sim_sha256=actual,change='HUD risk values only; collection adapter preserved'))
    return dest

def run(root):
    root=Path(root)
    if not root.exists():prepare(root)
    p=plan(root);results=[]
    if (root/'evaluation-progress.json').exists():results=read(root/'evaluation-progress.json')['results']
    for field_index,field in enumerate(FIELDS):
        for source,digest in p['source_sha256'].items():
            if sha(REPO/source)!=digest:raise ValueError('frozen learning source changed: '+source)
        collect(root,limit=(field_index+1)*64)
        new,index,roles=field_dataset(root,field);models={}
        if new is not None:
            for seed in p['learning_seeds'][field]:
                try:
                    model=fit_field(new,index,roles,seed,root/'models'/field/f'seed{seed}')
                    models[seed]=model
                except Exception as exc:
                    write(root/f'training-failure-{field}-{seed}.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
                    models[seed]=None
        evalroot=evaluation_runtime(root)
        for i,item in enumerate(p['evaluation']):
            if item['field']!=field:continue
            if any(r['index']==i for r in results):continue
            model=DEFAULT_CHECKPOINT if item['condition']=='R0' else models.get(item['model_seed'])
            reason=None
            if model is None:reason='N/A_no_trained_field_model'
            else:
                payload=torch.load(model,map_location='cpu',weights_only=True)
                needed=[] if item['condition']=='R1' else [2,3] if item['condition']=='R2' else list(range(4))
                if not all(payload['support']['active'][h] for h in needed):reason='N/A_unsupported_heads'
            entry=dict(index=i,assignment=item,folder=f'evaluation-runtime/evaluate-{i:03d}')
            if reason:entry['state']=reason
            else:
                folder=root/entry['folder']
                try:
                    if folder.exists():verify_derived(folder);result=read(folder/'result.json')
                    else:result=run_one(evalroot,item,f'evaluate-{i:03d}',model=model)
                    audit=audit_evaluation(folder,model)
                    prior=next((r for r in results if r['assignment']['field']==field and r['assignment']['seed']==item['seed'] and r.get('state')=='completed'),None)
                    diffs=[]
                    if prior:
                        ref=root/prior['folder']/'sim-output'
                        for name in ('anchor-state.npz','initial-input.npz'):
                            diffs+=compare_anchor(npz(folder/'sim-output'/name),npz(ref/name))
                    entry.update(state='completed',result=result,audit=audit,initial_pairing=dict(matched=not diffs,differences=diffs,reference=prior['folder'] if prior else None))
                except Exception as exc:entry.update(state='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            if entry.get('audit'):
                entry['comparison_eligible']=comparison_eligible(entry['audit'],entry['initial_pairing'])
                if not entry['initial_pairing']['matched']:
                    for prior_entry in results:
                        if prior_entry['assignment']['field']==field and prior_entry['assignment']['seed']==item['seed']:
                            prior_entry['comparison_eligible']=False
                            prior_entry['pair_group_invalidated_by']=entry['folder']
                if any(r.get('pair_group_invalidated_by') or not r.get('initial_pairing',{'matched':True})['matched'] for r in results if r['assignment']['field']==field and r['assignment']['seed']==item['seed']):
                    entry['comparison_eligible']=False
                    entry['pair_group_invalidated_by']='previous_case_mismatch'
            results.append(entry);atomic(root/'evaluation-progress.json',dict(done=len(results),planned=129,results=results));report(root,results)
        atomic(root/'pipeline-progress.json',dict(completed_fields=list(FIELDS[:field_index+1]),total_fields=7,collection_planned=448,evaluation_planned=129))
    write(root/'suite-results.json',results);report(root,results)
    seal_derived(root)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);args=parser.parse_args()
    run(args.root)
