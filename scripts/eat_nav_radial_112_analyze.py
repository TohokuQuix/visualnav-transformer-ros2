"""Outcome-blind analysis of sealed first-seed radial anchor shards."""
from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKTREE=Path('/tmp/eatnav-vnt-radial-112-sharded')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_shard_inventory import (
    validate_inventory, verify_shard_artifacts, settled_exceedance_reasons,
)
from visualnav_transformer.eat_nav.radial_interaction import classify_dem
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def metric(rows,key):
    values=[float(row['state'][key]) for row in rows[160:]]
    return {'final':values[-1],'mean':float(statistics.mean(values)),
            'max':max(map(abs,values)),'rms':float(math.sqrt(statistics.mean(v*v for v in values)))}


def analyze(root):
    root=Path(root);inventory=read(root/'inventory-manifest.json');validate_inventory(inventory)
    results=[];resources=[];sealed=[]
    for shard in inventory['shards']:
        sr=root/'cycle-0'/shard['shard_id'];seal=read(sr/'seal.json');status=read(sr/'status.json')
        ids=[p['pose_id'] for p in shard['poses']]
        if (seal['pose_ids']!=ids or sha(sr/'seal.json')!=status['seal_sha256'] or
            status['stage']!='full_complete' or seal['owned_residual'] is not False):
            raise ValueError('shard seal invalid '+shard['shard_id'])
        gate=verify_shard_artifacts(sr/'run',ids)
        if gate['pose_count']!=len(ids):raise ValueError('pose count mismatch')
        sealed.append({'shard_id':shard['shard_id'],'seal_sha256':sha(sr/'seal.json'),
                       'pose_count':len(ids)})
        samples=[json.loads(line) for line in (sr/'resource-series.jsonl').read_text().splitlines()]
        resources.append({'shard_id':shard['shard_id'],'gpu_peak_mib':max(x['gpu_device_used_mib'] for x in samples),
                          'anon_shmem_peak_bytes':max(x['container_anon_shmem_bytes'] for x in samples),
                          'host_available_min_bytes':min(x['host_available_bytes'] for x in samples),
                          'storage_peak_bytes':max(x['storage_bytes'] for x in samples),
                          'runtime_sec':max(x['elapsed_sec'] for x in samples)})
        for pose in shard['poses']:
            name=pose['pose_id'];run=sr/'run'
            trace=[json.loads(line) for line in (run/'trace'/f'{name}.jsonl').read_text().splitlines()]
            frames=[read(run/'dem'/name/f'frame-{j:02d}.json') for j in range(4)]
            reasons=settled_exceedance_reasons(trace,[f['valid_fraction'] for f in frames])
            last=frames[-1];dem=np.asarray(last['transformed'],dtype=float)
            interaction=classify_dem(dem[0],dem[1],float(last['resolution_m'])) if not reasons else None
            contacts=[r['contact'] for r in trace]
            body={}
            for c in contacts:
                for category,v in c['body_categories'].items():
                    b=body.setdefault(category,{'count':0,'force_sum_n':0.,'force_max_n':0.,'min_separation_m':None})
                    b['count']+=v['contact_count'];b['force_sum_n']+=v['force_sum_n'];b['force_max_n']=max(b['force_max_n'],v['force_max_n'])
                    sep=v['min_separation_m'];b['min_separation_m']=sep if b['min_separation_m'] is None else min(sep,b['min_separation_m'])
            item={'pose_id':name,'shard_id':shard['shard_id'],'terrain':pose['terrain'],
                  'terrain_class':pose['terrain_class'],'angle_index':pose['angle_index'],
                  'theta_rad':pose['theta_rad'],'env_index':pose['env_index'],
                  'source_family':pose['source_family'],'terrain_seed':pose['terrain_seed'],
                  'measurement_valid':True,'settled_anchor_eligible':not reasons,
                  'classification':'settled_anchor_eligible' if not reasons else 'measurement_valid_but_anchor_ineligible',
                  'ineligible_reasons':reasons,
                  'xy_drift_m':metric(trace,'xy_drift_m'),
                  'h1_heading_rad':metric(trace,'h1_signed_rad'),
                  'linear_speed_mps':metric(trace,'linear_speed_mps'),
                  'angular_speed_radps':metric(trace,'angular_speed_radps'),
                  'roll_final_rad':trace[-1]['state']['roll_rad'],
                  'pitch_final_rad':trace[-1]['state']['pitch_rad'],
                  'tilt_final_rad':trace[-1]['state']['tilt_rad'],
                  'dem_valid_fractions':[f['valid_fraction'] for f in frames],
                  'dem_complete':True,'actor_complete':True,
                  'contact_count_sum_across_steps':sum(c['contact_count'] for c in contacts),
                  'contact_occupancy_max':max(c['buffer_occupancy'] for c in contacts),
                  'contact_occupancy_ratio_max':max(c['capacity_ratio'] for c in contacts),
                  'min_separation_m':min((c['min_separation_m'] for c in contacts if c['min_separation_m'] is not None),default=None),
                  'max_decoded_penetration_m':max(c['max_decoded_penetration_m'] for c in contacts),
                  'final40_penetration_distribution_m':[c['max_decoded_penetration_m'] for c in contacts[160:]],
                  'all_force_sum_n':sum(c['force_sum_n'] for c in contacts),
                  'all_force_max_n':max(c['force_max_n'] for c in contacts),
                  'active_force_count':sum(c['active_force_distribution_n']['count'] for c in contacts),
                  'body_categories':body,'interaction':interaction}
            item['candidate_bank_supply']=bool(interaction and interaction['supply_candidate'])
            results.append(item)
    if len(results)!=112 or len({x['pose_id'] for x in results})!=112:
        raise ValueError('missing/duplicate 112-pose result')
    byterrain={t:[x for x in results if x['terrain']==t] for t in TERRAIN_ORDER}
    summaries={}
    for terrain,items in byterrain.items():
        classes=collections.Counter(x['interaction']['class'] for x in items if x['interaction'])
        summaries[terrain]={'measurement_valid':sum(x['measurement_valid'] for x in items),
                           'settled_anchor_eligible':sum(x['settled_anchor_eligible'] for x in items),
                           'numeric_exceedance':sum(not x['settled_anchor_eligible'] for x in items),
                           'candidate_bank_supply':sum(x['candidate_bank_supply'] for x in items),
                           'direction_indices_eligible':[x['angle_index'] for x in items if x['settled_anchor_eligible']],
                           'interaction_classes':dict(classes),'source_family_count':len({x['source_family'] for x in items}),
                           'terrain_seed_count':len({x['terrain_seed'] for x in items})}
    shortlist=[{'pose_id':x['pose_id'],'terrain':x['terrain'],'angle_index':x['angle_index'],
                'shard_id':x['shard_id'],'source_family':x['source_family'],'terrain_seed':x['terrain_seed'],
                'interaction_class':x['interaction']['class'],'start_distance_m':x['interaction']['interaction_start_m']}
               for x in results if x['candidate_bank_supply']]
    class_supply=collections.Counter(x['interaction_class'] for x in shortlist)
    flat_controls=sum(x['settled_anchor_eligible'] and x['interaction'] and x['interaction']['class']=='flat-only' for x in results)
    missing=[c for c in ('transition','graded','differential') if class_supply[c]==0]
    if flat_controls==0:missing.append('flat control')
    families=len({x['source_family'] for x in shortlist})
    eligible=sum(x['settled_anchor_eligible'] for x in results)
    if eligible<8:
        decision='radial_inventory_complete_insufficient_eligible_anchors'
    elif missing or families<4:
        decision='radial_inventory_complete_second_seed_review_required'
    else:
        decision='radial_inventory_complete_candidate_bank_review_ready'
    write(root/'per-pose-results.json',{'poses':results})
    write(root/'terrain-summary.json',summaries)
    write(root/'candidate-bank-shortlist.json',{'anchors':shortlist,'candidate_rollout_authorized':False})
    write(root/'missing-classes.json',{'missing':missing,'flat_controls':flat_controls,'interaction_source_families':families})
    write(root/'shard-resource-summary.json',{'shards':resources})
    write(root/'sealed-shards.json',{'shards':sealed})
    write(root/'final-decision.json',{'decision':decision,'measurement_valid':112,
          'settled_anchor_eligible':eligible,'candidate_bank_supply':len(shortlist),
          'source_families_total':len({x['source_family'] for x in results}),
          'source_families_in_shortlist':families,'missing_classes':missing,
          'independence_note':'16 directions per terrain share one source terrain seed/environment family; they are related measurements, not 16 independent StateGroups',
          'second_seed_executed':False,'candidate_rollout_executed':False})
    figures(root,results,resources,decision)
    report=['# Radial 112-pose sharded inventory','',f'**Decision:** `{decision}`','',
            f'Measurement-valid: 112/112. Settled-anchor eligible: {eligible}/112. Candidate-bank supply: {len(shortlist)}.','',
            '| Terrain | Eligible / 16 | Candidate supply | Numeric exceedance | Classes |','|---|---:|---:|---:|---|']
    for t in TERRAIN_ORDER:
        s=summaries[t]
        report.append(f"| {t} | {s['settled_anchor_eligible']} | {s['candidate_bank_supply']} | {s['numeric_exceedance']} | {s['interaction_classes']} |")
    report+=['','The classification uses only the settled anchor DEM. The 0–0.6 m forward corridor is a **proxy** for 2 s reach at an assumed 0.30 m/s; no candidate Path or rollout was used. The height/step/slope/validity thresholds were frozen before any shard result. Penetration is descriptive and has no inferred Onix safety threshold. Contact counts sum persistent decoded entries across 200 steps.','',
             f'Missing classes: {missing}. Source families in shortlist: {families}. All 16 directions of one terrain share its seed/family.','',
             'Resource and process evidence is in `shard-resource-summary.json` and each immutable shard seal. Figures 01–11 are in `figures/`. Second seed, Path, follower, bridge, Outcome and training were not run.','']
    (root/'report.md').write_text('\n'.join(report))
    ledger=[]
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name!='final-output-sha-ledger.json':
            ledger.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha(path)})
    write(root/'final-output-sha-ledger.json',{'count':len(ledger),'files':ledger})


def figures(root,items,resources,decision):
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    grid=np.empty((7,16),dtype=object)
    for x in items:grid[TERRAIN_ORDER.index(x['terrain']),x['angle_index']]=x
    theta=np.arange(16)*math.pi/8
    fig,ax=plt.subplots(subplot_kw={'projection':'polar'},figsize=(8,8))
    for ti in range(7):
        for ai in range(16):
            x=grid[ti,ai];ax.scatter(theta[ai],ti+1,c='#168045' if x['settled_anchor_eligible'] else '#c43b32',s=55)
    ax.set_yticks(range(1,8),TERRAIN_ORDER);ax.set_title('Settled-anchor eligibility: green eligible, red exceedance')
    fig.tight_layout();fig.savefig(figdir/'01-anchor-polar.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,5))
    for ax,key,title in zip(axs,('xy_drift_m','linear_speed_mps','angular_speed_radps'),('XY drift','Linear speed','Angular speed')):
        ax.boxplot([[x[key]['max'] for x in items if x['terrain']==t] for t in TERRAIN_ORDER],labels=TERRAIN_ORDER)
        ax.tick_params(axis='x',rotation=70);ax.set_title(title+' final40 max')
    fig.tight_layout();fig.savefig(figdir/'02-settled-distributions.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,key,title in zip(axs,('contact_occupancy_max','max_decoded_penetration_m'),('Peak contact occupancy','Max decoded penetration (m)')):
        ax.boxplot([[x[key] for x in items if x['terrain']==t] for t in TERRAIN_ORDER],labels=TERRAIN_ORDER)
        ax.tick_params(axis='x',rotation=70);ax.set_title(title)
    fig.tight_layout();fig.savefig(figdir/'03-contact-distributions.png',dpi=140);plt.close(fig)
    categories=sorted({k for x in items for k in x['body_categories']});fig,ax=plt.subplots(figsize=(11,5));bottom=np.zeros(7)
    for category in categories:
        values=np.array([sum(x['body_categories'].get(category,{}).get('count',0) for x in items if x['terrain']==t) for t in TERRAIN_ORDER])
        ax.bar(TERRAIN_ORDER,values,bottom=bottom,label=category);bottom+=values
    ax.tick_params(axis='x',rotation=35);ax.legend();ax.set_ylabel('Decoded entries across steps')
    fig.tight_layout();fig.savefig(figdir/'04-body-category.png',dpi=140);plt.close(fig)
    def heat(filename,title,value,cmap='viridis',vmin=None,vmax=None):
        a=np.array([[value(grid[ti,ai]) for ai in range(16)] for ti in range(7)],dtype=float)
        fig,ax=plt.subplots(figsize=(12,5));im=ax.imshow(a,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
        ax.set_yticks(range(7),TERRAIN_ORDER);ax.set_xticks(range(16),range(16));ax.set_xlabel('direction index');ax.set_title(title)
        fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/filename,dpi=140);plt.close(fig)
    heat('05-dem-mask.png','Minimum of four DEM valid fractions',lambda x:min(x['dem_valid_fractions']),vmin=0,vmax=1)
    heat('06-interaction-start.png','Interaction start distance (m); NaN = none/unknown',lambda x:x['interaction']['interaction_start_m'] if x['interaction'] and x['interaction']['interaction_start_m'] is not None else np.nan,vmin=0,vmax=.6)
    codes={'flat-only':0,'transition':1,'graded':2,'differential':3,'unknown':4}
    heat('07-interaction-class.png','0 flat, 1 transition, 2 graded, 3 differential, 4 unknown; -1 ineligible',lambda x:codes[x['interaction']['class']] if x['interaction'] else -1,cmap='tab10',vmin=-1,vmax=4)
    reasons=collections.Counter(reason for x in items for reason in x['ineligible_reasons']);fig,ax=plt.subplots(figsize=(9,5))
    ax.barh(list(reasons),list(reasons.values()));ax.set_title('Ineligible reasons (pose may have multiple)')
    fig.tight_layout();fig.savefig(figdir/'08-ineligible-reasons.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(TERRAIN_ORDER,[sum(x['candidate_bank_supply'] for x in items if x['terrain']==t) for t in TERRAIN_ORDER])
    ax.tick_params(axis='x',rotation=35);ax.set_title('Candidate-bank anchor supply by terrain')
    fig.tight_layout();fig.savefig(figdir/'09-shortlist-supply.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3));ax.axis('off')
    labels=['112 sealed','eligible anchors','interaction classes','independent families','review second seed?']
    for i,label in enumerate(labels):
        x=.1+.2*i;ax.text(x,.5,label,ha='center',bbox={'boxstyle':'round','fc':'#e7f2fb'})
        if i<4:ax.annotate('',xy=(x+.14,.5),xytext=(x+.07,.5),arrowprops={'arrowstyle':'->'})
    ax.set_title(decision);fig.tight_layout();fig.savefig(figdir/'10-second-seed-flow.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));ax.plot(range(23),[x['runtime_sec'] for x in resources],label='runtime s')
    ax2=ax.twinx();ax2.plot(range(23),[x['gpu_peak_mib'] for x in resources],color='tab:red',label='GPU MiB')
    ax.set_xlabel('shard order');ax.set_ylabel('runtime s');ax2.set_ylabel('GPU peak MiB');ax.set_title('Sealed shard lifecycle/resource')
    fig.tight_layout();fig.savefig(figdir/'11-shard-resource.png',dpi=140);plt.close(fig)


if __name__=='__main__':analyze(Path(sys.argv[1]))
