"""Sequential R4* GUI experiment; never starts before the R4 suite/review seal."""
from __future__ import annotations
import argparse,difflib,json,shlex,shutil,subprocess,sys,time,traceback
from pathlib import Path
import torch
from eat_nav_all_fields_run import REPO,CONTAINER,prepare as prepare_stage,run_one,atomic
from eat_nav_all_fields_learning import npz
from eat_nav_all_fields import compare_anchor
from eat_nav_r4_run import ROOT as R4_ROOT,fixed_jobs as r4_jobs
from eat_nav_r4_star import CONFIG,compare_initial
from eat_nav_r4_star_audit import audit_evaluation,comparison_eligible,report
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,verify_derived,require,seal_derived
ROOT=REPO/'docs/eat-nav/r4-star-six-candidate-evaluation-20261026'
REVIEW=REPO/'docs/eat-nav/r4-linear-only-evaluation-review-20261026'
QUEUE=REPO/'docs/eat-nav/r4-star-six-candidate-queue-20261026'
SPEC=REPO/'docs/eat-nav/r4-star-six-candidate-evaluation-spec-v0.1.md'
SOURCES=[REPO/'scripts'/f'eat_nav_r4_star{suffix}.py' for suffix in ('','_sim','_audit','_run')]

def fixed_jobs():
    return [dict(item,condition='R4*') for item in r4_jobs()]

def ready(root=R4_ROOT,review=REVIEW):
    return (Path(root)/'derived-seal.json').exists() and (Path(review)/'derived-seal.json').exists()

def verify_predecessor():
    require(ready(),'R4 and independent review must both complete first')
    verify_derived(R4_ROOT);verify_derived(REVIEW)
    old=read(R4_ROOT/'suite-results.json');check=read(REVIEW/'verification.json')
    require(len(old)==36 and read(R4_ROOT/'progress.json')['done']==36,'incomplete R4 suite')
    require(check['suite_ledger_sha256']==sha(R4_ROOT/'sha-ledger.json'),'review does not bind R4 final output')
    require(check['completed_runs']==sum(e['state']=='completed' for e in old),'R4 review count mismatch')
    return old

def prepare(root,queued):
    reference_results=verify_predecessor();root=Path(root).resolve();require(not root.exists(),'fresh R4* root required')
    prepare_stage(root)
    previous=read(root/'manifest.json');binding=read(root/'runtime-binding.json');reference=read(R4_ROOT/'runtime-binding.json')
    require(binding['image_id']==reference['image_id'],'runtime image changed')
    require(binding['onix_and_generator_source']==reference['onix_and_generator_source'],'Onix/generator source changed')
    write(root/'staging-reference-manifest.json',previous);write(root/'staging-reference-binding.json',binding)
    stage=Path(binding['stage'])
    overlays={'sim.py':REPO/'scripts/eat_nav_r4_star_sim.py','eat_nav_r4.py':REPO/'scripts/eat_nav_r4.py','eat_nav_r4_star.py':REPO/'scripts/eat_nav_r4_star.py'}
    for name,source in overlays.items():
        shutil.copy2(source,stage/name);subprocess.run(['docker','cp',str(source),CONTAINER+':'+str(stage/name)],check=True)
        actual=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',str(stage/name)],text=True).split()[0]
        require(actual==sha(source),'staged source mismatch: '+name);binding['files_sha256'][name]=actual
    for path,digest in reference['files_sha256'].items():
        if path!='sim.py':require(binding['files_sha256'].get(path)==digest,'R4 runtime source changed: '+path)
    jobs=queued['runs']
    for i,item in enumerate(jobs):
        old=reference_results[i]
        for key in ('field','seed','model_seed','model_sha256'):require(item[key]==old['assignment'][key],'R4 assignment mismatch')
        item['r4_reference']=dict(folder=old['folder'],state=old['state'],comparison_eligible=old.get('comparison_eligible',False))
    source=dict(read(R4_ROOT/'manifest.json')['source_sha256']);source.update(queued['source_sha256'])
    manifest=dict(schema='eatnav_r4_star_six_candidate_evaluation_v1',condition='R4*',config=CONFIG,fields=read(R4_ROOT/'manifest.json')['fields'],
        source_sha256=source,spec_sha256=queued['spec_sha256'],runs=jobs,planned_runs=36,models_retrained=False,reserved_test_opened=False,
        baseline_root=str(R4_ROOT),baseline_ledger_sha256=sha(R4_ROOT/'sha-ledger.json'),queue_manifest_sha256=sha(QUEUE/'queue-manifest.json'))
    atomic(root/'manifest.json',manifest);atomic(root/'runtime-binding.json',binding)
    dest=root/'executed-source';dest.mkdir()
    for p in SOURCES:shutil.copy2(p,dest/p.name)
    shutil.copy2(REPO/'tests/eat_nav/test_r4_star.py',dest/'test_r4_star.py')
    for name in ('focused-tests.txt','independent-review.md'):
        if (QUEUE/name).exists():shutil.copy2(QUEUE/name,root/name)
    old=(REPO/'scripts/eat_nav_r4_sim.py').read_text().splitlines(True);new=(REPO/'scripts/eat_nav_r4_star_sim.py').read_text().splitlines(True)
    (root/'sim-adapter.diff').write_text(''.join(difflib.unified_diff(old,new,fromfile='R4-sim',tofile='R4-star-sim')))
    return manifest

def run(root,queued):
    root=Path(root).resolve();verify_predecessor()
    manifest=read(root/'manifest.json') if root.exists() else prepare(root,queued)
    require(not (root/'derived-seal.json').exists(),'R4* output already sealed')
    results=read(root/'progress.json')['results'] if (root/'progress.json').exists() else []
    for i,item in enumerate(manifest['runs']):
        if any(e['index']==i for e in results):continue
        tag=f'run-{i:03d}';folder=root/tag;entry=dict(index=i,folder=tag,assignment=item)
        for source,digest in manifest['source_sha256'].items():require(sha(REPO/source)==digest,'frozen source changed: '+source)
        require(sha(Path(item['model']))==item['model_sha256'],'checkpoint changed')
        try:
            if folder.exists():
                verify_derived(folder)
                if not (folder/'result.json').exists():raise RuntimeError('preserved failed attempt')
                result=read(folder/'result.json')
            else:result=run_one(root,item,tag,model=item['model'])
            audit=audit_evaluation(folder,item['model']);reference=item['r4_reference'];diffs=[]
            unavailable=not reference['comparison_eligible']
            if not unavailable:
                base=R4_ROOT/reference['folder']/'sim-output'
                diffs.extend('anchor:'+k for k in compare_anchor(npz(folder/'sim-output/anchor-state.npz'),npz(base/'anchor-state.npz')))
                diffs.extend('input:'+k for k in compare_initial(npz(folder/'sim-output/initial-input.npz'),npz(base/'initial-input.npz')))
            pair=dict(matched=not diffs and not unavailable,differences=diffs,reference_unavailable=unavailable)
            entry.update(state='completed',result=result,audit=audit,initial_pairing=pair,comparison_eligible=comparison_eligible(audit,pair))
        except Exception as exc:
            evidence=read(folder/'sim-output/failure.json') if (folder/'sim-output/failure.json').exists() else None
            entry.update(state='failed',error=dict(type=type(exc).__name__,message=str(exc),simulator=evidence),comparison_eligible=False)
        results.append(entry);atomic(root/'progress.json',dict(done=len(results),planned=36,results=results));report(root,results,R4_ROOT)
        print('R4_STAR_PROGRESS '+json.dumps(dict(done=len(results),planned=36,state=entry['state'],comparison_eligible=entry['comparison_eligible'])),flush=True)
    require(len(results)==36,'incomplete R4* suite')
    # A second saved-input pass after the complete suite, before sealing.
    checks=[]
    for e in results:
        if e['state']!='completed':continue
        require(sha(Path(e['assignment']['model']))==e['assignment']['model_sha256'],'checkpoint changed before final review')
        fresh=audit_evaluation(root/e['folder'],e['assignment']['model']);require(fresh==e['audit'],'final audit mismatch')
        checks.append(dict(folder=e['folder'],replay_errors=fresh['replay_errors'],decisions=fresh['inference_decisions']))
    verify_predecessor();write(root/'suite-results.json',results);report(root,results,R4_ROOT)
    write(root/'verification.json',dict(completed_runs=len(checks),failed_runs=sum(e['state']=='failed' for e in results),
        eligible_runs=sum(e.get('comparison_eligible',False) for e in results),audits=checks,models_retrained=False,reserved_test_opened=False))
    seal_derived(root);verify_derived(root)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait',action='store_true');p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args()
    torch.set_num_threads(1)
    require(QUEUE.exists(),'freeze queue-manifest before launching')
    queued=read(QUEUE/'queue-manifest.json')
    def verify_queue():
        require(sha(SPEC)==queued['spec_sha256'],'queued spec changed')
        for path,digest in queued['source_sha256'].items():require(sha(REPO/path)==digest,'queued source changed: '+path)
        for item in queued['runs']:require(sha(Path(item['model']))==item['model_sha256'],'queued model changed')
    verify_queue();write(QUEUE/'status.json',dict(state='waiting_for_R4_and_review' if not ready() else 'starting',pid=__import__('os').getpid()))
    try:
        if args.wait:
            print('WAITING_FOR_R4_AND_REVIEW '+str(R4_ROOT),flush=True);deadline=time.monotonic()+24*3600
            while not ready():
                if (REVIEW/'failure.json').exists():raise RuntimeError('R4 independent review failed; inspect failure.json')
                if time.monotonic()>deadline:raise TimeoutError('R4 did not finish within24h')
                time.sleep(2)
        verify_queue();verify_predecessor();write(QUEUE/'status.json',dict(state='running',root=str(args.root)))
        run(args.root,queued);write(QUEUE/'status.json',dict(state='completed',root=str(args.root)))
        print('R4_STAR_COMPLETE '+str(args.root/'comparison-report.md'),flush=True)
    except Exception as exc:
        write(QUEUE/'status.json',dict(state='failed',error=str(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
