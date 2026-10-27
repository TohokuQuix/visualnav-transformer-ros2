"""After R4:224 reverse-curve branches,12 fresh models,36 R4* GUI evaluations."""
from __future__ import annotations
import argparse,difflib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import torch
from eat_nav_all_fields_run import REPO,CONTAINER,prepare as legacy_prepare,run_one,atomic
from eat_nav_all_fields import compare_anchor
from eat_nav_all_fields_learning import npz
from eat_nav_r4_run import BASE,ROOT as R4_ROOT,fixed_jobs as original_jobs
from eat_nav_r4_star_run import REVIEW,ready,verify_predecessor
from eat_nav_r4_star import CONFIG,compare_initial
from eat_nav_r4_star_learning import field_dataset,fit_field,KRAIL_SEEDS
from eat_nav_r4_star_audit import audit_evaluation,comparison_eligible,report
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,require,verify_derived,seal_derived
ROOT=REPO/'docs/eat-nav/r4-star-learned-six-candidate-20261026'
QUEUE=REPO/'docs/eat-nav/r4-star-learned-six-candidate-queue-20261026'
SPEC=REPO/'docs/eat-nav/r4-star-six-candidate-learning-spec-v0.2.md'
NEW_SOURCES=[REPO/'scripts'/(n+'.py') for n in ('eat_nav_r4_star_tensors','eat_nav_r4_star_learning','eat_nav_r4_star_collect_sim','eat_nav_r4_star_learning_run')]

def jobs():
    manifest=read(BASE/'manifest.json');collection=[]
    for r in manifest['collect']:
        if r['candidate']!=0:continue
        for k in (4,5):collection.append(dict(r,candidate=k))
    evaluation=[dict(r,condition='R4*') for r in original_jobs()]
    require(len(collection)==224 and len(evaluation)==36,'fixed job counts')
    return collection,evaluation

def prepare_runtime(root,phase,plan):
    root=Path(root)
    if (root/'overlay.json').exists():return root
    require(not root.exists(),'incomplete staging preserved; inspect before recovery')
    legacy_prepare(root);before=read(root/'manifest.json');binding=read(root/'runtime-binding.json')
    reference=read((BASE if phase=='collect' else R4_ROOT)/'runtime-binding.json')
    require(binding['image_id']==reference['image_id'],'image changed')
    require(binding['onix_and_generator_source']==reference['onix_and_generator_source'],'Onix/generator changed')
    write(root/'manifest-before-overlay.json',before);write(root/'runtime-before-overlay.json',binding)
    stage=Path(binding['stage']);sim=REPO/'scripts'/('eat_nav_r4_star_collect_sim.py' if phase=='collect' else 'eat_nav_r4_star_sim.py')
    overlays={'sim.py':sim,'eat_nav_r4.py':REPO/'scripts/eat_nav_r4.py','eat_nav_r4_star.py':REPO/'scripts/eat_nav_r4_star.py'}
    for name,source in overlays.items():
        shutil.copy2(source,stage/name);subprocess.run(['docker','cp',str(source),CONTAINER+':'+str(stage/name)],check=True)
        actual=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',str(stage/name)],text=True).split()[0]
        require(actual==sha(source),'overlay SHA mismatch');binding['files_sha256'][name]=actual
    for name,digest in reference['files_sha256'].items():
        if name!='sim.py':require(binding['files_sha256'].get(name)==digest,'shared runtime changed: '+name)
    atomic(root/'manifest.json',dict(schema='eatnav_r4_star_learned_runtime_v2',phase=phase,source_sha256=plan['source_sha256'],reserved_test_opened=False))
    atomic(root/'runtime-binding.json',binding);write(root/'overlay.json',dict(sim_sha256=sha(sim),phase=phase))
    old=REPO/'scripts'/('eat_nav_all_fields_sim.py' if phase=='collect' else 'eat_nav_r4_sim.py')
    (root/'sim-adapter.diff').write_text(''.join(difflib.unified_diff(old.read_text().splitlines(True),sim.read_text().splitlines(True),fromfile=old.name,tofile=sim.name)))
    return root

def attempt(runtime,item,tag,model=None):
    folder=runtime/tag
    if folder.exists():
        verify_derived(folder)
        if not (folder/'result.json').exists():raise RuntimeError('preserved failed attempt')
        return read(folder/'result.json')
    return run_one(runtime,item,tag,model=model)

def error_record(exc,folder):
    evidence=read(folder/'sim-output/failure.json') if (folder/'sim-output/failure.json').exists() else None
    return dict(type=type(exc).__name__,message=str(exc),simulator=evidence)

def run(root,plan):
    root=Path(root);verify_predecessor();verify_derived(BASE)
    if not root.exists():
        root.mkdir();write(root/'plan.json',plan)
        snapshot=root/'executed-source';snapshot.mkdir()
        for source in plan['source_sha256']:
            if source.startswith('scripts/eat_nav_r4_star') or source.startswith('tests/eat_nav/test_r4_star'):shutil.copy2(REPO/source,snapshot/Path(source).name)
        for name in ('focused-tests.txt','independent-review.md'):
            if (QUEUE/name).exists():shutil.copy2(QUEUE/name,root/name)
    require(read(root/'plan.json')==plan,'frozen plan changed');require(not (root/'derived-seal.json').exists(),'completed root sealed')
    collectroot=prepare_runtime(root/'collection-runtime','collect',plan)
    collected=read(root/'collection-progress.json')['results'] if (root/'collection-progress.json').exists() else []
    for i,item in enumerate(plan['collect']):
        if any(e['index']==i for e in collected):continue
        verify_sources(plan);tag=f'collect-{i:03d}';e=dict(index=i,folder=str(Path('collection-runtime')/tag),assignment=item)
        try:e.update(state='completed',result=attempt(collectroot,item,tag))
        except Exception as exc:e.update(state='failed',error=error_record(exc,collectroot/tag))
        collected.append(e);atomic(root/'collection-progress.json',dict(done=len(collected),planned=224,results=collected))
        write(QUEUE/'status.json',dict(state='collecting',done=len(collected),planned=224,root=str(root)))
        print('R4_STAR_COLLECTION '+json.dumps(dict(done=len(collected),state=e['state'])),flush=True)
    models={};training=[]
    for field in plan['fields']:
        verify_sources(plan);arrays,index,roles=field_dataset(root,field,BASE)
        for seed in (KRAIL_SEEDS if field=='k_rail_field' else [123]):
            write(QUEUE/'status.json',dict(state='training',field=field,model_seed=seed,root=str(root)))
            item=dict(field=field,model_seed=seed)
            try:
                if arrays is None:item.update(state='N/A_no_added_candidate_training_targets')
                else:
                    path=fit_field(arrays,index,roles,seed,root/'models'/field/f'seed{seed}')
                    require(path is not None,'no field training groups')
                    model,payload=__import__('eat_nav_command_free_learning').load_checkpoint(path)
                    require(payload['support']['active'][2],'linear head unsupported')
                    models[field,seed]=path;item.update(state='completed',checkpoint=str(path),sha256=sha(path))
            except Exception as exc:item.update(state='failed',error=str(exc));write(root/f'training-failure-{field}-{seed}.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            training.append(item);atomic(root/'training-progress.json',dict(done=len(training),planned=12,results=training))
    evalroot=prepare_runtime(root/'evaluation-runtime','evaluate',plan);results=[]
    if (root/'progress.json').exists():results=read(root/'progress.json')['results']
    references=read(R4_ROOT/'suite-results.json');runs=[]
    for i,old in enumerate(plan['evaluation']):
        item=dict(old);model=models.get((item['field'],item['model_seed']))
        item['model']=str(model) if model else None;item['model_sha256']=sha(model) if model else None
        runs.append(item)
    write(root/'manifest.json',dict(schema='eatnav_r4_star_learned_suite_v2',fields=plan['fields'],runs=runs,config=CONFIG,models_retrained=True,
        source_sha256=plan['source_sha256'],baseline_root=str(R4_ROOT),baseline_ledger_sha256=sha(R4_ROOT/'sha-ledger.json'),reserved_test_opened=False))
    for i,item in enumerate(runs):
        if any(e['index']==i for e in results):continue
        verify_sources(plan);tag=f'evaluate-{i:03d}';e=dict(index=i,folder=str(Path('evaluation-runtime')/tag),assignment=item,comparison_eligible=False)
        model=item['model']
        if model is None:e.update(state='N/A_no_trained_model')
        else:
            require(sha(Path(model))==item['model_sha256'],'trained checkpoint changed')
            try:
                result=attempt(evalroot,item,tag,model=model);audit=audit_evaluation(evalroot/tag,model)
                reference=references[i]
                require(all(reference['assignment'][k]==item[k] for k in ('field','seed','model_seed')),'reference assignment mismatch')
                unavailable=not reference.get('comparison_eligible',False);diffs=[]
                if not unavailable:
                    before=R4_ROOT/reference['folder']/'sim-output';after=evalroot/tag/'sim-output'
                    diffs=['anchor:'+s for s in compare_anchor(npz(after/'anchor-state.npz'),npz(before/'anchor-state.npz'))]
                    diffs+=['input:'+s for s in compare_initial(npz(after/'initial-input.npz'),npz(before/'initial-input.npz'))]
                pair=dict(matched=not unavailable and not diffs,reference_unavailable=unavailable,differences=diffs)
                e.update(state='completed',result=result,audit=audit,initial_pairing=pair,comparison_eligible=comparison_eligible(audit,pair))
            except Exception as exc:e.update(state='failed',error=error_record(exc,evalroot/tag))
        results.append(e);atomic(root/'progress.json',dict(done=len(results),planned=36,results=results));report(root,results,R4_ROOT)
        correct_report(root)
        write(QUEUE/'status.json',dict(state='evaluating',done=len(results),planned=36,root=str(root)))
        print('R4_STAR_EVALUATION '+json.dumps(dict(done=len(results),state=e['state'],comparison_eligible=e['comparison_eligible'])),flush=True)
    checks=[]
    for e in results:
        if e['state']!='completed':continue
        require(sha(Path(e['assignment']['model']))==e['assignment']['model_sha256'],'trained model changed')
        fresh=audit_evaluation(root/e['folder'],e['assignment']['model']);require(fresh==e['audit'],'final replay mismatch')
        checks.append(dict(folder=e['folder'],replay_errors=fresh['replay_errors']))
    write(root/'suite-results.json',results);write(root/'verification.json',dict(completed_evaluations=len(checks),eligible_evaluations=sum(e['comparison_eligible'] for e in results),
        trained_models=sum(t['state']=='completed' for t in training),collection_completed=sum(t['state']=='completed' for t in collected),models_retrained=True,reserved_test_opened=False,audits=checks))
    verify_predecessor();report(root,results,R4_ROOT);correct_report(root);seal_derived(collectroot);seal_derived(evalroot);seal_derived(root);verify_derived(root)

def correct_report(root):
    p=Path(root)/'comparison-report.md';s=p.read_text()
    s=s.replace('再学習なし。R4*は先頭4候補を保持し、後退−30度/＋30度だけを追加。','後退−30度/＋30度の実走行データを追加して同じ学習設定で新規学習。R4*は6候補。')
    s=s.replace('追加後退曲線は現行4候補datasetで直接学習していない。同じweightsによる候補拡張であり追加学習の効果ではない。','追加後退曲線は照合済み教師を含めて学習済み。R4との差は候補数と追加教師による学習の組合せであり、候補数だけの効果を分離しない。')
    p.write_text(s)

def verify_sources(plan):
    require(sha(SPEC)==plan['spec_sha256'],'spec changed')
    for path,digest in plan['source_sha256'].items():require(sha(REPO/path)==digest,'frozen source changed: '+path)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait',action='store_true');p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args()
    torch.set_num_threads(1);plan=read(QUEUE/'plan.json');verify_sources(plan)
    write(QUEUE/'status.json',dict(state='waiting_for_R4_and_review',pid=__import__('os').getpid()))
    try:
        if args.wait:
            print('WAITING_FOR_R4_AND_REVIEW',flush=True);deadline=time.monotonic()+24*3600
            while not ready():
                if (REVIEW/'failure.json').exists():raise RuntimeError('R4 review failed')
                if time.monotonic()>deadline:raise TimeoutError('R4 completion wait exceeded24h')
                time.sleep(2)
        verify_sources(plan);verify_predecessor();run(args.root,plan)
        write(QUEUE/'status.json',dict(state='completed',root=str(args.root)))
        print('R4_STAR_LEARNED_COMPLETE '+str(args.root/'comparison-report.md'),flush=True)
    except Exception as exc:
        write(QUEUE/'status.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
