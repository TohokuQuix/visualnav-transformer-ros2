"""Offline R3 reserve eligibility, class supply, and review figures."""
from __future__ import annotations

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

ROOT=Path('/tmp/eatnav-r3-transition-reserve-20261008')
OLD=Path('/tmp/eatnav-r3-settling-pilot-20261008')
WORKTREE=Path('/tmp/eatnav-vnt-r3-transition-reserve')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_interaction import classify_dem
from visualnav_transformer.eat_nav.radial_shard_inventory import verify_shard_artifacts
from visualnav_transformer.eat_nav.r3_transition_decision import decide_transition


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_old_analyzer():
    path=WORKTREE/'scripts/eat_nav_r3_settling_analyze.py'
    spec=importlib.util.spec_from_file_location('old_analyzer',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mod.ROOT=ROOT
    return mod


def read_series(p):return [json.loads(s) for s in p.read_text().splitlines()]


def savefig(fig,path):fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def figures(cause,results,resource,decision,out):
    out.mkdir(parents=True,exist_ok=True)
    # 1. Rank 1/2 angular settling.
    fig,ax=plt.subplots(figsize=(10,4))
    for pose in cause[:2]:
        oldrow=next(r for r in json.loads((OLD/'primary-manifest.json').read_text())['poses'] if r['rank']==pose['rank'])
        shard='shard-A' if pose['rank']%2 else 'shard-B'
        series=read_series(OLD/shard/'run/trace'/f"{oldrow['pose_id']}.jsonl")
        ax.plot([s['physics_step'] for s in series], [s['state']['angular_speed_radps'] for s in series],label=f"Rank {pose['rank']}")
    ax.axhline(.2,color='r',ls='--',label='frozen 0.20 gate');ax.axvspan(161,200,color='gray',alpha=.12)
    ax.set(xlabel='physics step',ylabel='angular speed (rad/s)',title='Original transition poses: 200-step settling')
    ax.legend();savefig(fig,out/'01-rank1-2-angular-speed.png')
    # 2. Contact force aligned with angular peak.
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,pose in zip(axes,cause[:2]):
        shard='shard-A' if pose['rank']%2 else 'shard-B'
        series=read_series(OLD/shard/'run/trace'/f"{pose['pose_id']}.jsonl")
        x=[s['physics_step'] for s in series]
        ax.plot(x,[s['contact']['force_sum_n'] for s in series],label='force sum (N)')
        ax.axvline(pose['angular_speed']['peak_step'],color='r',ls='--',label='angular peak')
        ax.set(ylabel='contact force (N)',title=f"Rank {pose['rank']}")
        ax.legend()
    axes[-1].set_xlabel('physics step');savefig(fig,out/'02-contact-angular-timing.png')
    # 3. Four-frame DEM classes for original eight.
    classes=['flat-only','graded','transition','differential','unknown']
    image=np.asarray([[classes.index(f['class']) for f in p['dem_frames']] for p in cause])
    fig,ax=plt.subplots(figsize=(7,5));ax.imshow(image,vmin=0,vmax=4,cmap='tab10',aspect='auto')
    ax.set(xticks=range(4),xticklabels=[str(i) for i in range(4)],yticks=range(8),
        yticklabels=[str(i) for i in range(1,9)],xlabel='DEM frame',ylabel='original rank',title='Original eight: settled DEM class by frame')
    for i in range(8):
        for j in range(4):ax.text(j,i,cause[i]['dem_frames'][j]['class'][:4],ha='center',va='center',fontsize=7,color='white')
    savefig(fig,out/'03-eight-dem-class-sequence.png')
    # 4. Virtual/settled slope margin to graded threshold.
    fig,ax=plt.subplots(figsize=(9,4))
    xs=np.arange(8);old=[p['virtual_metrics']['slope']-.12 for p in cause]
    new=[p['dem_frames'][-1]['threshold_margin']['graded_slope'] for p in cause]
    ax.bar(xs-.18,old,width=.36,label='virtual');ax.bar(xs+.18,new,width=.36,label='settled frame 3')
    ax.axhline(0,color='black');ax.set(xticks=xs,xticklabels=[str(i) for i in range(1,9)],
        xlabel='original rank',ylabel='slope minus 0.12',title='Virtual vs settled DEM threshold margin')
    ax.legend();savefig(fig,out/'04-virtual-settled-threshold-margin.png')
    # 5. Frozen reserve placement.
    fig,ax=plt.subplots(figsize=(5,5),subplot_kw={'projection':'polar'})
    for r in results:
        a=r['direction_index']*math.pi/8
        ax.scatter(a,r['radius_m'],s=100);ax.text(a,r['radius_m']+.025,f"rank {r['rank']}")
    ax.set_ylim(0,.7);ax.set_title('Frozen transition reserve placement')
    savefig(fig,out/'05-reserve-placement.png')
    # 6. Reserve eligibility and class outcome.
    fig,ax=plt.subplots(figsize=(8,3))
    for i,r in enumerate(results):
        ax.barh(i,1,color='#4c9f70' if r['physical_supply'] else '#c65d5d')
        ax.text(.02,i,f"{r['classification']} / {r['settled_class']} / supply={r['physical_supply']}",va='center',fontsize=9)
    ax.set(xlim=(0,1),yticks=range(len(results)),yticklabels=[f"R3 {r['rank']}" for r in results],
        xticks=[],title='Reserve eligibility and primary settled class')
    savefig(fig,out/'06-reserve-eligibility-class.png')
    # 7. First-seed supply count, original plus reserve.
    old_summary=json.loads((OLD/'analysis/class-support.json').read_text())
    old_poses=json.loads((OLD/'analysis/per-pose-results.json').read_text())
    if isinstance(old_poses,dict):old_poses=old_poses.get('poses',old_poses.get('rows',[]))
    classes=['transition','differential','graded','flat-only']
    counts=[sum(p.get('physical_supply',False) and p['virtual_class']==c for p in old_poses)+
        sum(p.get('physical_supply',False) and p['virtual_class']==c for p in results) for c in classes]
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(classes,counts,color=['#c14256','#ef9234','#4682b4','#b9bcc2'])
    ax.axhline(1,color='black',ls='--');ax.set(ylabel='physical supply count',title='First-seed class supply after frozen reserve')
    savefig(fig,out/'07-first-seed-four-class-supply.png')
    # 8. Decision flow.
    fig,ax=plt.subplots(figsize=(8,2.5));ax.axis('off')
    ax.text(.5,.75,'104 frozen reserve → rank 3/4 transition → one sealed shard',ha='center',fontsize=11)
    ax.text(.5,.42,'eligibility + primary DEM class + known onset',ha='center',fontsize=11)
    ax.text(.5,.12,decision,ha='center',fontsize=9,color='#24527a')
    savefig(fig,out/'08-decision-flow.png')


def main():
    ledger=json.loads((ROOT/'freeze-sha.json').read_text())
    status=json.loads((ROOT/'status.json').read_text())
    if status['stage'] not in ('measurement_complete','complete'):
        raise RuntimeError('reserve shard not sealed')
    if sha(ROOT/'reserve-manifest.json')!=ledger['reserve_manifest_sha256']:
        raise RuntimeError('reserve manifest SHA drift')
    sr=ROOT/'shard-reserve'
    seal=json.loads((sr/'seal.json').read_text())
    rows=json.loads((sr/'pose-manifest.json').read_text())['poses']
    if (seal['native_exit_code']!=0 or seal['owned_residual'] is not False or
        seal['pose_ids']!=[x['pose_id'] for x in rows] or
        sha(sr/'seal.json')!=status['seal_sha256']):raise RuntimeError('reserve protocol seal drift')
    verify_shard_artifacts(sr/'run',seal['pose_ids'],termination_aware=True)
    analyzer=load_old_analyzer();virtual=analyzer.load_virtual()
    frozen_items={x['pose_id']:x for x in json.loads((ROOT/'reserve-manifest.json').read_text())['selected']}
    results=[]
    for row in rows:
        result=analyzer.analyze_pose('shard-reserve',row,virtual[row['pose_id']])
        frames=[]
        for fp in sorted((sr/'run/dem'/row['pose_id']).glob('frame-*.json')):
            f=json.loads(fp.read_text());a=np.asarray(f['transformed'],dtype=np.float32)
            frames.append({'index':f['frame_index'],'class':classify_dem(a[0],a[1]>.5,.1)['class'],
                'valid_fraction':f['valid_fraction'],'sha256':f['transformed_sha256']})
        result['secondary_all_frame_classes']=frames
        result['virtual_unknown_only_difference']=frozen_items[row['pose_id']]['virtual_unknown_only_difference']
        result['physical_supply']=bool(result['classification']=='settled_anchor_eligible' and
            result.get('settled_class')=='transition' and result.get('known_onset') and
            result.get('virtual_class')=='transition' and
            frozen_items[row['pose_id']]['virtual_unknown_only_difference'] is False and
            result.get('dem_valid_fraction',0)>=.7)
        results.append(result)
    decision=decide_transition(results,len(json.loads((ROOT/'reserve-manifest.json').read_text())['selected']))
    cause=json.loads((ROOT/'analysis/eight-pose-cause-audit.json').read_text())['poses']
    values=read_series(sr/'resource-series.jsonl')
    resource={'gpu_device_peak_mib':max(x['gpu_device_used_mib'] for x in values),
        'container_anon_shmem_peak_bytes':max(x['container_anon_shmem_bytes'] for x in values),
        'host_available_min_bytes':min(x['host_available_bytes'] for x in values),
        'storage_peak_bytes':max(x['storage_bytes'] for x in values),
        'samples':len(values),'elapsed_sec':max(x['elapsed_sec'] for x in values),
        'native_exit_code':seal['native_exit_code'],'owned_residual':seal['owned_residual']}
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    (out/'reserve-per-pose.json').write_text(json.dumps(results,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (out/'resource-lifecycle.json').write_text(json.dumps(resource,indent=2,sort_keys=True)+'\n')
    result={'schema':'eatnav_r3_transition_decision_v1','decision':decision,
        'reserve_manifest_sha256':ledger['reserve_manifest_sha256'],
        'pose_ids':[x['pose_id'] for x in rows],
        'protocol_valid_count':len(results),'transition_physical_supply_count':sum(x['physical_supply'] for x in results),
        'second_seed_started':False,'additional_reserve_started':False}
    (out/'final-decision.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    figures(cause,results,resource,decision,out/'figures')
    files=[p for p in out.rglob('*') if p.is_file()]
    (out/'sha-ledger.json').write_text(json.dumps({'files':[{'path':str(p.relative_to(ROOT)),
        'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]},indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
