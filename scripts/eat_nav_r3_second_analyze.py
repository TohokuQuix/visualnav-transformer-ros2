"""Read-only eligibility, robustness comparison, and second-seed supply review."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/tmp/eatnav-r3-second-seed-20261008')
WORKTREE=Path('/tmp/eatnav-vnt-r3-second-seed')
FIRST=Path('/tmp/eatnav-r3-settling-pilot-20261008')
RESERVE=Path('/tmp/eatnav-r3-transition-reserve-20261008')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.anchor_placement_rules import CLASS_ORDER
from visualnav_transformer.eat_nav.radial_interaction import classify_dem
from visualnav_transformer.eat_nav.radial_shard_inventory import verify_shard_artifacts


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_old_analyzer():
    path=WORKTREE/'scripts/eat_nav_r3_settling_analyze.py'
    spec=importlib.util.spec_from_file_location('r3_first_analyzer',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.ROOT=ROOT
    return mod


def read_jsonl(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def build_resource():
    rows=[]
    for shard in ('shard-A','shard-B'):
        base=ROOT/shard;seal=json.loads((base/'seal.json').read_text())
        series=read_jsonl(base/'resource-series.jsonl')
        rows.append({'shard':shard,'samples':len(series),
            'gpu_peak_mib':max(s['gpu_device_used_mib'] for s in series),
            'anon_shmem_peak_bytes':max(s['container_anon_shmem_bytes'] for s in series),
            'host_available_min_bytes':min(s['host_available_bytes'] for s in series),
            'storage_peak_bytes':max(s['storage_bytes'] for s in series),
            'elapsed_sec':max(s['elapsed_sec'] for s in series),
            'native_exit_code':seal['native_exit_code'],
            'owned_process':seal['owned_process'],'owned_residual':seal['owned_residual'],
            'seal_sha256':sha(base/'seal.json')})
    return rows


def analyze_poses():
    analyzer=load_old_analyzer()
    virtual={x['pose_id']:x for radius in (.30,.40,.50,.60)
        for x in json.loads((ROOT/f'scan-v1/radius-{radius:.2f}.json').read_text())['candidates']}
    contract=json.loads((ROOT/'supply-contract.json').read_text())
    if contract['selection_uses_robustness'] is not False:
        raise RuntimeError('supply contract drift')
    rows=json.loads((ROOT/'primary-manifest.json').read_text())['poses']
    results=[]
    for row in rows:
        shard='shard-A' if row['pilot_slot']<=4 else 'shard-B'
        item=analyzer.analyze_pose(shard,row,virtual[row['pose_id']])
        item['pilot_slot']=row['pilot_slot'];item['r3_rank']=row['rank']
        item['virtual_interaction_onset_m']=row['virtual_interaction_onset_m']
        item['robustness']=row['robustness']
        item['virtual_unknown_only_difference']=row['virtual_unknown_only_difference']
        item['virtual_corridor_known_fraction']=row['virtual_corridor_known_fraction']
        frames=[]
        for fp in sorted((ROOT/shard/'run/dem'/row['pose_id']).glob('frame-*.json')):
            f=json.loads(fp.read_text());a=np.asarray(f['transformed'],dtype=np.float32)
            if a.shape!=(2,20,20) or hashlib.sha256(a.tobytes()).hexdigest()!=f['transformed_sha256']:
                raise RuntimeError('saved settled DEM invalid: '+row['pose_id'])
            detail=classify_dem(a[0],a[1]>.5,.1)
            frames.append({'frame_index':f['frame_index'],'step':f['physics_step'],
                'class':detail['class'],'onset_m':detail['interaction_start_m'],
                'known_fraction':detail['corridor_valid_fraction'],
                'dem_sha256':f['transformed_sha256']})
        item['secondary_frame_classes']=frames
        eligible=item['classification']=='settled_anchor_eligible'
        match=eligible and item['settled_class']==row['virtual_class']
        if row['virtual_class']=='flat-only':
            known=bool(match and item.get('settled_interaction_onset_m') is None and
                item['dem_valid_fraction']>=.7 and not row['virtual_unknown_only_difference'])
        else:
            known=bool(match and item.get('known_onset') is True and
                item.get('settled_interaction_onset_m') is not None and
                item['settled_interaction_onset_m']<=.6 and
                not row['virtual_unknown_only_difference'])
        item['physical_supply_second_seed']=known
        item['primary_virtual_settled_class_match']=bool(match)
        item['primary_onset_match']=bool(match and
            ((item.get('settled_interaction_onset_m') is None and row['virtual_interaction_onset_m'] is None) or
             (item.get('settled_interaction_onset_m') is not None and row['virtual_interaction_onset_m'] is not None and
              abs(item['settled_interaction_onset_m']-row['virtual_interaction_onset_m'])<=1e-6)))
        results.append(item)
    return results


def decide(results):
    if len(results)!=8 or any(x['classification']=='measurement_invalid' for x in results):
        return 'r3_second_seed_measurement_invalid'
    supply={c:sum(x['physical_supply_second_seed'] for x in results if x['virtual_class']==c)
            for c in CLASS_ORDER}
    if all(v>=1 for v in supply.values()):
        return 'r3_second_seed_physical_supply_pass_ready_for_goal_candidate_bank_design'
    if any(x['classification']=='settled_anchor_eligible' and
           x['settled_class']!=x['virtual_class'] for x in results):
        return 'r3_second_seed_class_mismatch_review_required'
    if sum(x['classification'] in ('anchor_ineligible_numeric','anchor_ineligible_physical_termination')
           for x in results)>=5:
        return 'r3_second_seed_physical_feasibility_fail'
    return 'r3_second_seed_partial_class_support_review_required'


def figures(items,selected,results,resource,decision,out):
    out.mkdir(parents=True,exist_ok=True)
    colors={'transition':'#c14256','differential':'#ef9234','graded':'#4682b4','flat-only':'#b9bcc2','unknown':'#777777'}
    def save(fig,name):fig.tight_layout();fig.savefig(out/name,dpi=140);plt.close(fig)
    terrains=sorted({x['terrain'] for x in items});radii=(.30,.40,.50,.60)
    # 1. Nominal class map.
    fig,axes=plt.subplots(7,1,figsize=(12,12),sharex=True)
    classes=list(colors)
    for ax,t in zip(axes,terrains):
        a=np.full((4,16),4)
        for x in items:
            if x['terrain']==t:a[radii.index(x['radius_m']),x['direction_index']]=classes.index(x['classification'])
        ax.imshow(a,aspect='auto',vmin=0,vmax=4,cmap='tab10');ax.set_yticks(range(4),[str(r) for r in radii]);ax.set_ylabel(t)
    axes[-1].set_xticks(range(16));axes[-1].set_xlabel('direction index')
    fig.suptitle('Second-seed radius × direction nominal class');save(fig,'01-radius-direction-class-map.png')
    # 2. R3 class supply.
    fig,ax=plt.subplots(figsize=(7,4));xx=np.arange(4)
    ax.bar(xx-.18,[sum(x['classification']==c for x in items) for c in CLASS_ORDER],.36,label='448 scan')
    ax.bar(xx+.18,[sum(x['classification']==c for x in selected) for c in CLASS_ORDER],.36,label='R3 112')
    ax.set_xticks(xx,CLASS_ORDER,rotation=20);ax.set_ylabel('related pose count');ax.legend();save(fig,'02-r3-class-supply.png')
    # 3. Perturbation retention by terrain and radius.
    fig,ax=plt.subplots(figsize=(8,5));matrix=np.asarray([[np.mean([x['robustness']['class_retention_fraction']
        for x in items if x['terrain']==t and x['radius_m']==r]) for r in radii] for t in terrains])
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(4),[str(r) for r in radii]);ax.set_yticks(range(7),terrains);fig.colorbar(im,ax=ax,label='15-condition class retention')
    save(fig,'03-perturbation-class-retention.png')
    # 4. Onset sensitivity.
    fig,ax=plt.subplots(figsize=(8,4));onset=[x['robustness']['onset_range_m'] for x in results]
    ax.bar(range(8),[0 if x is None else x for x in onset]);ax.set_xticks(range(8),[str(x['pilot_slot']) for x in results]);ax.set(xlabel='pilot slot',ylabel='onset range (m)',title='15-condition onset sensitivity')
    save(fig,'04-onset-sensitivity.png')
    # 5. Pilot placements.
    fig,ax=plt.subplots(figsize=(6,6),subplot_kw={'projection':'polar'})
    for x in results:
        angle=x['direction_index']*math.pi/8;ax.scatter(angle,x['radius_m'],color=colors[x['virtual_class']],s=100)
        ax.text(angle,x['radius_m']+.025,str(x['pilot_slot']))
    ax.set_ylim(0,.7);ax.set_title('Frozen eight second-seed pilot poses');save(fig,'05-eight-pilot-placements.png')
    # 6. Settling kinematics.
    fig,ax=plt.subplots(figsize=(10,5))
    for x in results:
        series=read_jsonl(ROOT/x['shard_id']/'run/trace'/f"{x['pose_id']}.jsonl")
        ax.plot([v['physics_step'] for v in series],[v['state']['angular_speed_radps'] for v in series],
                label=str(x['pilot_slot']))
    ax.axhline(.2,ls='--',color='red');ax.axvspan(161,200,color='gray',alpha=.1)
    ax.set(xlabel='physics step',ylabel='angular speed (rad/s)',title='Frozen 200-step settling');ax.legend(ncol=8)
    save(fig,'06-settling-kinematics.png')
    # 7. Virtual and primary settled class, NA for ineligible.
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(range(8),[1]*8,color=[colors[x['virtual_class']] for x in results])
    for i,x in enumerate(results):ax.text(i,.5,f"{x['virtual_class']}\n→ {x['settled_class'] or 'NA'}",ha='center',va='center',rotation=90,fontsize=8)
    ax.set(xticks=range(8),xticklabels=[str(x['pilot_slot']) for x in results],yticks=[],xlabel='pilot slot',title='Virtual → primary settled DEM class')
    save(fig,'07-virtual-settled-class.png')
    # 8. Descriptive robustness vs class-match.
    fig,ax=plt.subplots(figsize=(7,4))
    for x in results:
        category=(1 if x['primary_virtual_settled_class_match'] else
                  0 if x['classification']=='settled_anchor_eligible' else -1)
        ax.scatter(x['robustness']['class_retention_fraction'],category,
            color=colors[x['virtual_class']]);ax.text(x['robustness']['class_retention_fraction'],
            category+.025,str(x['pilot_slot']))
    ax.set(xlim=(0,1.05),ylim=(-1.2,1.2),yticks=(-1,0,1),
        yticklabels=('NA: ineligible','eligible mismatch','eligible match'),
        xlabel='15-condition class retention',title='Descriptive, n=8; no fitted threshold')
    save(fig,'08-robustness-vs-class-match.png')
    # 9. First/second seed class support.
    first=json.loads((FIRST/'analysis/per-pose-results.json').read_text())
    if isinstance(first,dict):first=first.get('poses',first.get('rows',[]))
    reserve=json.loads((RESERVE/'analysis/reserve-per-pose.json').read_text())
    first_count=[sum(x.get('physical_supply',False) and x['virtual_class']==c for x in first)+
        sum(x.get('physical_supply',False) and x['virtual_class']==c for x in reserve) for c in CLASS_ORDER]
    second_count=[sum(x['physical_supply_second_seed'] and x['virtual_class']==c for x in results) for c in CLASS_ORDER]
    fig,ax=plt.subplots(figsize=(8,4));xx=np.arange(4)
    ax.bar(xx-.18,first_count,.36,label='first seed');ax.bar(xx+.18,second_count,.36,label='second seed')
    ax.axhline(1,color='black',ls='--');ax.set_xticks(xx,CLASS_ORDER,rotation=20);ax.set_ylabel('physical supply anchors');ax.legend()
    save(fig,'09-first-second-support.png')
    # 10. Decision flow.
    fig,ax=plt.subplots(figsize=(9,2.5));ax.axis('off')
    ax.text(.5,.78,'seed freeze → 448 zero-step DEMs → unchanged R3 → eight frozen poses',ha='center')
    ax.text(.5,.48,'two sealed candidate-free shards → eligibility + primary DEM class',ha='center')
    ax.text(.5,.16,decision,ha='center',color='#24527a',fontsize=9)
    save(fig,'10-decision-flow.png')


def main():
    status=json.loads((ROOT/'status.json').read_text())
    pre=json.loads((ROOT/'pilot-preflight.json').read_text())
    if status['stage']!='measurement_complete' or status['completed_shards']!=2:
        raise RuntimeError('two sealed shards required before analysis')
    if sha(ROOT/'selection/pilot-manifest.json')!=pre['pilot_manifest_sha256']:
        raise RuntimeError('pilot manifest SHA drift')
    for shard in ('shard-A','shard-B'):
        base=ROOT/shard;seal=json.loads((base/'seal.json').read_text())
        manifest=json.loads((base/'pose-manifest.json').read_text())['poses']
        if seal['native_exit_code']!=0 or seal['owned_residual'] is not False or seal['pose_ids']!=[x['pose_id'] for x in manifest]:
            raise RuntimeError('shard protocol invalid: '+shard)
        verify_shard_artifacts(base/'run',seal['pose_ids'],termination_aware=True)
    results=analyze_poses();decision=decide(results)
    resources=build_resource()
    items=[x for radius in (.30,.40,.50,.60) for x in json.loads((ROOT/f'scan-v1/radius-{radius:.2f}.json').read_text())['candidates']]
    selected_ids={x['pose_id'] for x in json.loads((ROOT/'selection/reserve-rank-freeze.json').read_text())['poses']}|{x['pose_id'] for x in results}
    selected=[x for x in items if x['pose_id'] in selected_ids]
    output=ROOT/'analysis-v2';output.mkdir(exist_ok=False)
    supply={c:sum(x['physical_supply_second_seed'] for x in results if x['virtual_class']==c) for c in CLASS_ORDER}
    for name,obj in [('per-pose-results.json',results),('resource-lifecycle.json',resources),
        ('robustness-vs-settled.json',{'class_match_groups':{
            'eligible_match':[x['robustness']['class_retention_fraction'] for x in results if x['primary_virtual_settled_class_match']],
            'eligible_mismatch':[x['robustness']['class_retention_fraction'] for x in results if x['classification']=='settled_anchor_eligible' and not x['primary_virtual_settled_class_match']],
            'ineligible_NA':[x['robustness']['class_retention_fraction'] for x in results if x['classification']!='settled_anchor_eligible']},'supply_groups':{
            str(supply_flag):[x['robustness']['class_retention_fraction'] for x in results if x['physical_supply_second_seed']==supply_flag]
            for supply_flag in (True,False)},'per_pose':[{'pose_id':x['pose_id'],
            'class_retention_fraction':x['robustness']['class_retention_fraction'],
            'onset_range_m':x['robustness']['onset_range_m'],
            'threshold_margin':x['robustness']['nominal_threshold_margin'],
            'class_match':x['primary_virtual_settled_class_match'],
            'physical_supply':x['physical_supply_second_seed']} for x in results]}),
        ('final-decision.json',{'schema':'eatnav_r3_second_decision_v1','decision':decision,
            'protocol_valid_count':len(results),'class_supply_count':supply,
            'pilot_manifest_sha256':pre['pilot_manifest_sha256'],
            'interpretation':'second-seed goal/candidate-bank design readiness only; no terrain-generalization claim',
            'additional_seed_started':False,'reserve_executed':False,'candidate_path_generated':False})]:
        (output/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
    figures(items,selected,results,resources,decision,output/'figures')
    files=[p for p in output.rglob('*') if p.is_file()]
    (output/'sha-ledger.json').write_text(json.dumps({'files':[{'path':str(p.relative_to(ROOT)),
        'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]},indent=2,sort_keys=True)+'\n')
    print(json.dumps(json.loads((output/'final-decision.json').read_text()),indent=2))

if __name__=='__main__':main()
