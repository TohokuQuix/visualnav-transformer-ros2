"""Fixed36-case R4 GUI evaluation with existing checkpoints and paired baselines."""
from __future__ import annotations
import argparse,copy,difflib,json,shlex,shutil,subprocess,sys
from pathlib import Path
import torch
from eat_nav_all_fields_run import REPO,CONTAINER,prepare as prepare_stage,run_one,atomic
from eat_nav_all_fields_learning import npz
from eat_nav_all_fields import compare_anchor
from eat_nav_r4 import CONFIG
from eat_nav_r4_audit import audit_evaluation,comparison_eligible,report
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,verify_derived,require,seal_derived
BASE=REPO/'docs/eat-nav/all-fields-learning-20261023'
ROOT=REPO/'docs/eat-nav/r4-linear-only-evaluation-20261026'


def fixed_jobs():
    p=read(BASE/'learning-evaluation-plan.json');old=read(BASE/'suite-results.json');jobs=[]
    for item in p['evaluation']:
        if item['condition']!='R3':continue
        r=dict(item,condition='R4');r['references']={}
        for mode in ('R1','R2','R3'):
            match=next(e for e in old if all(e['assignment'].get(k)==item.get(k) for k in ('field','seed','model_seed')) and e['assignment']['condition']==mode)
            r['references'][mode]=dict(folder=match['folder'],state=match['state'],comparison_eligible=match.get('comparison_eligible',False))
        model=BASE/'models'/r['field']/f"seed{r['model_seed']}"/'checkpoint.pt'
        verify_derived(model.parent)
        require(sha(model)==read(model.parent/'training.json')['checkpoint_sha256'],'model binding')
        r.update(model=str(model),model_sha256=sha(model));jobs.append(r)
    require(len(jobs)==36,'expected36 R4 jobs')
    return jobs


def prepare(root):
    root=Path(root).resolve();require(not root.exists(),'fresh R4 root required');verify_derived(BASE)
    jobs=fixed_jobs();prepare_stage(root)
    previous_manifest=read(root/'manifest.json');binding=read(root/'runtime-binding.json');reference=read(BASE/'evaluation-runtime/runtime-binding.json')
    require(binding['image_id']==reference['image_id'],'runtime image changed')
    require(binding['onix_and_generator_source']==reference['onix_and_generator_source'],'Onix/generator source changed')
    for path,digest in reference['files_sha256'].items():
        if path!='sim.py':require(binding['files_sha256'].get(path)==digest,'baseline runtime source changed: '+path)
    write(root/'staging-reference-manifest.json',previous_manifest);write(root/'staging-reference-binding.json',binding)
    stage=Path(binding['stage']);newfiles={'sim.py':REPO/'scripts/eat_nav_r4_sim.py','eat_nav_r4.py':REPO/'scripts/eat_nav_r4.py'}
    for name,source in newfiles.items():
        shutil.copy2(source,stage/name);subprocess.run(['docker','cp',str(source),CONTAINER+':'+str(stage/name)],check=True)
        actual=subprocess.check_output(['docker','exec',CONTAINER,'sha256sum',str(stage/name)],text=True).split()[0]
        require(actual==sha(source),'R4 staged source mismatch');binding['files_sha256'][name]=actual
    source_paths=[REPO/'scripts'/('eat_nav_r4'+suffix+'.py') for suffix in ('','_sim','_run','_audit')]
    source_sha=dict(previous_manifest['source_sha256']);source_sha.update({str(p.relative_to(REPO)):sha(p) for p in source_paths})
    manifest=dict(schema='eatnav_r4_fixed_evaluation_v1',fields=read(BASE/'manifest.json')['fields'],condition='R4',config=CONFIG,
        baseline_root=str(BASE),baseline_ledger_sha256=sha(BASE/'sha-ledger.json'),runs=jobs,planned_runs=36,
        source_sha256=source_sha,spec_sha256=sha(REPO/'docs/eat-nav/r4-linear-only-evaluation-spec-v0.1.md'),
        models_retrained=False,reserved_test_opened=False)
    atomic(root/'manifest.json',manifest);atomic(root/'runtime-binding.json',binding)
    snapshots=root/'executed-source';snapshots.mkdir()
    for p in source_paths:shutil.copy2(p,snapshots/p.name)
    shutil.copy2(REPO/'tests/eat_nav/test_r4.py',snapshots/'test_r4.py')
    old=(REPO/'scripts/eat_nav_all_fields_eval_sim.py').read_text().splitlines(True)
    new=(REPO/'scripts/eat_nav_r4_sim.py').read_text().splitlines(True)
    (root/'sim-adapter.diff').write_text(''.join(difflib.unified_diff(old,new,fromfile='existing-sim',tofile='R4-sim')))
    return manifest


def run(root,limit=None):
    root=Path(root).resolve()
    manifest=read(root/'manifest.json') if root.exists() else prepare(root)
    require(not (root/'derived-seal.json').exists(),'completed R4 root is sealed')
    results=read(root/'progress.json')['results'] if (root/'progress.json').exists() else []
    for i,item in enumerate(manifest['runs'][:limit]):
        if any(e['index']==i for e in results):continue
        tag=f'run-{i:03d}';folder=root/tag;entry=dict(index=i,folder=tag,assignment=item)
        for source,digest in manifest['source_sha256'].items():require(sha(REPO/source)==digest,'frozen source changed: '+source)
        require(sha(Path(item['model']))==item['model_sha256'],'frozen checkpoint changed')
        try:
            if folder.exists():
                verify_derived(folder)
                if not (folder/'result.json').exists():raise RuntimeError('preserved failed attempt')
                result=read(folder/'result.json')
            else:result=run_one(root,item,tag,model=item['model'])
            audit=audit_evaluation(folder,item['model']);diffs=[];unavailable=[]
            for mode,ref in item['references'].items():
                if not ref['comparison_eligible']:unavailable.append(mode);continue
                base=BASE/ref['folder']/'sim-output'
                for name in ('anchor-state.npz','initial-input.npz'):
                    diffs.extend(mode+':'+name+':'+k for k in compare_anchor(npz(folder/'sim-output'/name),npz(base/name)))
            pair=dict(matched=not diffs and not unavailable,differences=diffs,reference_unavailable=unavailable)
            entry.update(state='completed',result=result,audit=audit,initial_pairing=pair,comparison_eligible=comparison_eligible(audit,pair))
        except Exception as exc:
            evidence=read(folder/'sim-output/failure.json') if (folder/'sim-output/failure.json').exists() else None
            entry.update(state='failed',error=dict(type=type(exc).__name__,message=str(exc),simulator=evidence),comparison_eligible=False)
        results.append(entry);atomic(root/'progress.json',dict(done=len(results),planned=36,results=results));report(root,results,BASE)
        print('R4_PROGRESS '+json.dumps(dict(done=len(results),planned=36,state=entry['state'],comparison_eligible=entry['comparison_eligible'])),flush=True)
    if len(results)==36:
        write(root/'suite-results.json',results);report(root,results,BASE)
        seal_derived(root)
    return results

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--limit',type=int)
    args=p.parse_args();run(args.root,args.limit)
