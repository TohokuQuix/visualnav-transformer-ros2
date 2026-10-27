"""Wait for trained14/evaluated14 R4*, then run paired R5 without retraining."""
from __future__ import annotations
import argparse,difflib,json,shutil,subprocess,time,traceback,os
from pathlib import Path
import torch
from eat_nav_all_fields_run import REPO,CONTAINER,prepare as legacy_prepare,run_one,atomic
from eat_nav_all_fields_learning import npz
from eat_nav_all_fields import compare_anchor
from eat_nav_r4_star14_run import ROOT as PREVIOUS,QUEUE as PREVIOUS_QUEUE
from eat_nav_r5_14 import CONFIG
from eat_nav_r5_14_audit import audit_evaluation,comparison_eligible,report
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,require,verify_derived,seal_derived
ROOT=REPO/'docs/eat-nav/r5-stall-only-no-pose-stop-20261026'
QUEUE=REPO/'docs/eat-nav/r5-stall-only-no-pose-stop-queue-20261026'
SPEC=REPO/'docs/eat-nav/r5-stall-only-no-pose-stop-spec-v0.2.md'
SOURCES=[REPO/'scripts'/('eat_nav_r5'+suffix+'.py') for suffix in ('','_sim','_audit','_run','_14','_14_sim','_14_audit')]

def template_jobs():
    jobs=[dict(x,condition='R5') for x in read(PREVIOUS_QUEUE/'plan.json')['evaluation']]
    require(len(jobs)==36,'36 fixed assignments required');return jobs

def ready(root=PREVIOUS):return (Path(root)/'derived-seal.json').exists()

def supported(payload):
    active=payload.get('support',{}).get('active',[])
    return len(active)==4 and bool(active[2]) and bool(active[3])

def verify_previous():
    require(ready(),'R4* must be fully sealed first');verify_derived(PREVIOUS)
    results=read(PREVIOUS/'suite-results.json');check=read(PREVIOUS/'verification.json');manifest=read(PREVIOUS/'manifest.json')
    require(len(results)==len(manifest['runs'])==36,'R4* suite incomplete')
    require(manifest.get('evaluation_candidate_count')==14,'R4* must evaluate fourteen candidates')
    require(check['models_retrained'] is True and check['completed_evaluations']==sum(e['state']=='completed' for e in results),'R4* final replay missing')
    return results,manifest

def bind_jobs(template):
    results,manifest=verify_previous();jobs=[]
    for i,item in enumerate(template):
        reference=results[i];old=reference['assignment']
        require(all(item[k]==old[k] for k in ('field','seed','model_seed')),'predecessor assignment mismatch')
        r=dict(item,model=old.get('model'),model_sha256=old.get('model_sha256'),previous_folder=reference['folder'],previous_state=reference['state'])
        reason=None
        if not r['model']:reason='N/A_no_predecessor_model'
        else:
            path=Path(r['model']);verify_derived(path.parent);require(sha(path)==r['model_sha256'],'predecessor model SHA mismatch')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if not supported(payload):reason='N/A_unsupported_linear_or_turn_head'
            require(payload.get('config',{}).get('training_candidate_ids')==read(PREVIOUS/'plan.json')['training_candidate_ids'],'fourteen-training provenance mismatch')
        r['unavailable_reason']=reason;jobs.append(r)
    return jobs

def prepare(root,plan):
    previous,previous_manifest=verify_previous();runs=bind_jobs(plan['runs']);root=Path(root);require(not root.exists(),'fresh R5 root required')
    legacy_prepare(root);before=read(root/'manifest.json');binding=read(root/'runtime-binding.json');ref=read(PREVIOUS/'evaluation-runtime/runtime-binding.json')
    require(binding['image_id']==ref['image_id'],'image changed');require(binding['onix_and_generator_source']==ref['onix_and_generator_source'],'Onix source changed')
    write(root/'manifest-before-overlay.json',before);write(root/'runtime-before-overlay.json',binding);stage=Path(binding['stage'])
    overlays={'sim.py':REPO/'scripts/eat_nav_r5_14_sim.py','eat_nav_r5.py':REPO/'scripts/eat_nav_r5.py',
        **{n+'.py':REPO/'scripts'/(n+'.py') for n in ('eat_nav_r4','eat_nav_r4_star','eat_nav_r4_star14','eat_nav_r4_star14_eval','eat_nav_r5_14')}}
    for name,source in overlays.items():
        shutil.copy2(source,stage/name);subprocess.run(['docker','cp',str(source),CONTAINER+':'+str(stage/name)],check=True)
        actual=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',str(stage/name)],text=True).split()[0]
        require(actual==sha(source),'staged source mismatch');binding['files_sha256'][name]=actual
    for name,digest in ref['files_sha256'].items():
        if name!='sim.py':require(binding['files_sha256'].get(name)==digest,'shared runtime changed: '+name)
    manifest=dict(schema='eatnav_r5_same_trained_models_fourteen_v2',evaluation_candidate_count=14,evaluation_candidate_ids=CONFIG['candidate_ids'],condition='R5',config=CONFIG,fields=previous_manifest['fields'],runs=runs,
        source_sha256=plan['source_sha256'],spec_sha256=plan['spec_sha256'],previous_root=str(PREVIOUS),previous_ledger_sha256=sha(PREVIOUS/'sha-ledger.json'),
        models_retrained=False,reserved_test_opened=False)
    atomic(root/'manifest.json',manifest);atomic(root/'runtime-binding.json',binding)
    dest=root/'executed-source';dest.mkdir()
    for p in SOURCES+[REPO/'tests/eat_nav/test_r5.py']:shutil.copy2(p,dest/p.name)
    for name in ('focused-tests.txt','independent-review.md'):
        if (QUEUE/name).exists():shutil.copy2(QUEUE/name,root/name)
    a=(REPO/'scripts/eat_nav_r4_star14_eval_sim.py').read_text().splitlines(True);b=(REPO/'scripts/eat_nav_r5_14_sim.py').read_text().splitlines(True)
    (root/'sim-adapter.diff').write_text(''.join(difflib.unified_diff(a,b,fromfile='R4-star-sim',tofile='R5-sim')))
    return manifest

def verify_sources(plan):
    require(sha(SPEC)==plan['spec_sha256'],'R5 spec changed')
    for name,digest in plan['source_sha256'].items():require(sha(REPO/name)==digest,'frozen source changed: '+name)

def run(root,plan):
    root=Path(root);previous,old_manifest=verify_previous()
    manifest=read(root/'manifest.json') if root.exists() else prepare(root,plan)
    require(not (root/'derived-seal.json').exists(),'R5 already complete')
    require(manifest['previous_ledger_sha256']==sha(PREVIOUS/'sha-ledger.json'),'predecessor changed')
    results=read(root/'progress.json')['results'] if (root/'progress.json').exists() else []
    for i,item in enumerate(manifest['runs']):
        if any(e['index']==i for e in results):continue
        verify_sources(plan);tag=f'run-{i:03d}';folder=root/tag;e=dict(index=i,folder=tag,assignment=item,comparison_eligible=False)
        if item['unavailable_reason']:e.update(state=item['unavailable_reason'])
        else:
            require(sha(Path(item['model']))==item['model_sha256'],'same-model SHA changed')
            try:
                if folder.exists():
                    verify_derived(folder)
                    if not (folder/'result.json').exists():raise RuntimeError('preserved failed attempt')
                    result=read(folder/'result.json')
                else:result=run_one(root,item,tag,model=item['model'])
                audit=audit_evaluation(folder,item['model']);reference=previous[i]
                unavailable=reference['state']!='completed' or not reference.get('audit',{}).get('eligible',False);diffs=[]
                if not unavailable:
                    before=PREVIOUS/reference['folder']/'sim-output';after=folder/'sim-output'
                    for name in ('anchor-state.npz','initial-input.npz'):diffs.extend(name+':'+k for k in compare_anchor(npz(after/name),npz(before/name)))
                pair=dict(matched=not unavailable and not diffs,differences=diffs,reference_unavailable=unavailable)
                e.update(state='completed',result=result,audit=audit,initial_pairing=pair,comparison_eligible=comparison_eligible(audit,pair))
            except Exception as exc:
                evidence=read(folder/'sim-output/failure.json') if (folder/'sim-output/failure.json').exists() else None
                e.update(state='failed',error=dict(type=type(exc).__name__,message=str(exc),simulator=evidence))
        results.append(e);atomic(root/'progress.json',dict(done=len(results),planned=36,results=results));report(root,results,PREVIOUS)
        atomic(QUEUE/'status.json',dict(state='evaluating',done=len(results),planned=36,root=str(root)))
        print('R5_PROGRESS '+json.dumps(dict(done=len(results),state=e['state'],comparison_eligible=e['comparison_eligible'])),flush=True)
    require(len(results)==36,'incomplete R5 suite');checks=[]
    for e in results:
        if e['state']!='completed':continue
        require(sha(Path(e['assignment']['model']))==e['assignment']['model_sha256'],'model changed')
        fresh=audit_evaluation(root/e['folder'],e['assignment']['model']);require(fresh==e['audit'],'R5 final replay mismatch')
        checks.append(dict(folder=e['folder'],replay_errors=fresh['replay_errors']))
    verify_previous();write(root/'suite-results.json',results);report(root,results,PREVIOUS)
    write(root/'verification.json',dict(completed_runs=len(checks),eligible_runs=sum(e['comparison_eligible'] for e in results),
        failed_runs=sum(e['state']=='failed' for e in results),models_retrained=False,reserved_test_opened=False,audits=checks))
    seal_derived(root);verify_derived(root)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait',action='store_true');p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args()
    torch.set_num_threads(1);plan=read(QUEUE/'plan.json');verify_sources(plan)
    atomic(QUEUE/'status.json',dict(state='waiting_for_R4_star_completion',pid=os.getpid()))
    try:
        if args.wait:
            print('WAITING_FOR_R4_STAR_COMPLETION',flush=True)
            while not ready():
                status=read(PREVIOUS_QUEUE/'status.json') if (PREVIOUS_QUEUE/'status.json').exists() else {}
                if status.get('state') in ('failed','cancelled_before_start'):raise RuntimeError('R4* stopped: '+str(status.get('error',status.get('reason',''))))
                time.sleep(2)
        verify_sources(plan);verify_previous();atomic(QUEUE/'status.json',dict(state='starting',root=str(args.root)))
        run(args.root,plan);atomic(QUEUE/'status.json',dict(state='completed',root=str(args.root)))
        print('R5_COMPLETE '+str(args.root/'comparison-report.md'),flush=True)
    except Exception as exc:
        atomic(QUEUE/'status.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
