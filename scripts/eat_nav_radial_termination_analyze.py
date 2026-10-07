"""Read-only combined 112-pose analysis for sealed old and continuation shards."""
from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKTREE=Path('/tmp/eatnav-vnt-radial-112-sharded')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_shard_inventory import (
    validate_inventory, verify_shard_artifacts, settled_exceedance_reasons)
from visualnav_transformer.eat_nav.radial_interaction import classify_dem
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER
from visualnav_transformer.eat_nav.radial_termination_capture import verify_pose_coverage


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')


def analyze(root):
    root=Path(root);inventory=read(root/'inventory-manifest.json');validate_inventory(inventory)
    reuse=read(root/'reuse-ledger.json')
    if len(reuse['reused_shards'])!=15 or reuse['reused_pose_count']!=75:
        raise ValueError('reuse ledger count drift')
    results=[];resources=[];seals=[]
    for index,shard in enumerate(inventory['shards']):
        name=shard['shard_id']
        sr=Path(reuse['reused_shards'][index]['old_root']) if index<15 else root/'cycle-1'/name
        seal=read(sr/'seal.json');status=read(sr/'status.json')
        ids=[p['pose_id'] for p in shard['poses']]
        if (status.get('stage')!='full_complete' or status.get('owned_residual') is not False or
            status.get('seal_sha256')!=sha(sr/'seal.json') or seal['pose_ids']!=ids or
            seal['native_exit_code']!=0 or seal['run_sha_ledger_sha256']!=sha(sr/'sha-ledger.json')):
            raise ValueError('invalid shard seal: '+name)
        if index<15 and sha(sr/'seal.json')!=reuse['reused_shards'][index]['seal_sha256']:
            raise ValueError('reused shard changed: '+name)
        artifact=verify_shard_artifacts(sr/'run',ids,termination_aware=index>=15)
        if artifact['pose_count']!=len(ids):raise ValueError('shard completeness mismatch')
        seals.append({'shard_id':name,'root':str(sr),'seal_sha256':sha(sr/'seal.json'),
                      'terminated_pose_count':artifact['terminated_pose_count']})
        samples=[json.loads(line) for line in (sr/'resource-series.jsonl').read_text().splitlines()]
        resources.append({'shard_id':name,'gpu_peak_mib':max(x['gpu_device_used_mib'] for x in samples),
                          'anon_shmem_peak_bytes':max(x['container_anon_shmem_bytes'] for x in samples),
                          'host_available_min_bytes':min(x['host_available_bytes'] for x in samples),
                          'storage_peak_bytes':max(x['storage_bytes'] for x in samples),
                          'runtime_sec':max(x['elapsed_sec'] for x in samples)})
        for pose in shard['poses']:
            pid=pose['pose_id'];run=sr/'run'
            trace=[json.loads(line) for line in (run/'trace'/f'{pid}.jsonl').read_text().splitlines()]
            marker_path=run/'termination'/f'{pid}.json'
            marker=read(marker_path) if marker_path.exists() else None
            frame_paths=sorted((run/'dem'/pid).glob('frame-*.json'))
            frames=[read(path) for path in frame_paths]
            contacts=[row['contact'] for row in trace]
            base={'pose_id':pid,'shard_id':name,'terrain':pose['terrain'],
                  'angle_index':pose['angle_index'],'env_index':pose['env_index'],
                  'source_family':pose['source_family'],'terrain_seed':pose['terrain_seed'],
                  'measurement_valid':True,'dem_frame_count':len(frames),
                  'actor_prephysics_complete':(run/'actor'/pid/'prephysics.json').exists(),
                  'contact_complete':all(c['bounds_valid'] and c['finite'] and
                      not c['buffer_full_or_truncation_suspected'] and c['buffer_capacity']==4096 for c in contacts),
                  'contact_count_sum_across_steps':sum(c['contact_count'] for c in contacts),
                  'contact_occupancy_max':max(c['buffer_occupancy'] for c in contacts),
                  'max_decoded_penetration_m':max(c['max_decoded_penetration_m'] for c in contacts),
                  'dem_valid_fractions':[f['valid_fraction'] for f in frames],
                  'physics_steps':len(trace)}
            if marker:
                base.update({'classification':'anchor_ineligible_physical_termination',
                    'settled_anchor_eligible':False,'candidate_bank_supply':False,
                    'ineligible_reasons':['physical_termination:'+','.join(marker['termination_reason'])],
                    'termination_reason':marker['termination_reason'],
                    'termination_step':marker['termination_step'],
                    'termination_marker_sha256':sha(marker_path),
                    'actor_terminal_complete':len(marker['pre_reset_actor'])==22,
                    'actor_step_200_complete':False,'interaction':None})
            else:
                if len(frames)!=4 or not (run/'actor'/pid/'settled_step_200.json').exists():
                    raise ValueError('complete pose DEM/Actor missing: '+pid)
                reasons=settled_exceedance_reasons(trace,[f['valid_fraction'] for f in frames])
                dem=np.asarray(frames[-1]['transformed'],dtype=float)
                interaction=classify_dem(dem[0],dem[1],float(frames[-1]['resolution_m'])) if not reasons else None
                base.update({'classification':'anchor_ineligible_numeric' if reasons else 'settled_anchor_eligible',
                    'settled_anchor_eligible':not reasons,'candidate_bank_supply':bool(interaction and interaction['supply_candidate']),
                    'ineligible_reasons':reasons,'termination_reason':[], 'termination_step':None,
                    'actor_terminal_complete':False,'actor_step_200_complete':True,'interaction':interaction,
                    'final40_max':{key:max(abs(float(r['state'][key])) for r in trace[160:]) for key in
                                   ('xy_drift_m','h1_signed_rad','linear_speed_mps','angular_speed_radps')}})
            results.append(base)
    verify_pose_coverage([p['pose_id'] for p in inventory['poses']],results)
    counts=collections.Counter(x['classification'] for x in results)
    terrain={t:{'directions':{str(x['angle_index']):x['classification'] for x in results if x['terrain']==t},
                'counts':dict(collections.Counter(x['classification'] for x in results if x['terrain']==t))}
             for t in TERRAIN_ORDER}
    shortlist=[{'pose_id':x['pose_id'],'terrain':x['terrain'],'angle_index':x['angle_index'],
                'source_family':x['source_family'],'terrain_seed':x['terrain_seed'],
                'interaction_class':x['interaction']['class'],
                'interaction_start_m':x['interaction']['interaction_start_m']}
               for x in results if x['candidate_bank_supply']]
    classes=collections.Counter(x['interaction_class'] for x in shortlist)
    flat=sum(x['settled_anchor_eligible'] and x['interaction'] and x['interaction']['class']=='flat-only' for x in results)
    missing=[c for c in ('transition','graded','differential') if classes[c]==0]
    if flat==0:missing.append('flat control')
    shortlist_families=len({x['source_family'] for x in shortlist})
    review_ready=counts['settled_anchor_eligible']>=8 and not missing and shortlist_families>=4
    if counts['anchor_ineligible_physical_termination']:
        decision='radial_inventory_complete_with_physical_terminations'
    elif counts['settled_anchor_eligible']<8:
        decision='radial_inventory_complete_insufficient_eligible_anchors'
    elif review_ready:
        decision='radial_inventory_complete_candidate_bank_review_ready'
    else:
        decision='radial_inventory_complete_second_seed_review_required'
    final={'decision':decision,'counts':dict(counts),'measurement_valid':112,
           'candidate_bank_supply':len(shortlist),'candidate_bank_review_ready':bool(review_ready),
           'second_seed_review_required':not review_ready,'missing_classes':missing,
           'source_families_total':len({x['source_family'] for x in results}),
           'source_families_in_shortlist':shortlist_families,
           'independence_note':'16 directions per terrain share one source terrain seed/family',
           'second_seed_executed':False,'candidate_path_executed':False}
    write(root/'per-pose-results.json',{'poses':results})
    write(root/'terrain-direction-summary.json',terrain)
    write(root/'termination-summary.json',{'poses':[{'pose_id':x['pose_id'],'terrain':x['terrain'],
          'angle_index':x['angle_index'],'step':x['termination_step'],'terms':x['termination_reason']}
          for x in results if x['termination_reason']]})
    write(root/'candidate-bank-shortlist.json',{'anchors':shortlist,'candidate_rollout_authorized':False})
    write(root/'resource-summary.json',{'shards':resources,'gpu_peak_mib':max(x['gpu_peak_mib'] for x in resources),
          'anon_shmem_peak_bytes':max(x['anon_shmem_peak_bytes'] for x in resources),
          'storage_peak_bytes':max(x['storage_peak_bytes'] for x in resources),
          'runtime_sum_sec':sum(x['runtime_sec'] for x in resources)})
    write(root/'sealed-shards.json',{'shards':seals})
    write(root/'final-decision.json',final)
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    grid=np.full((7,16),np.nan)
    code={'settled_anchor_eligible':0,'anchor_ineligible_numeric':1,
          'anchor_ineligible_physical_termination':2,'measurement_invalid':3}
    for item in results:grid[TERRAIN_ORDER.index(item['terrain']),item['angle_index']]=code[item['classification']]
    fig,ax=plt.subplots(figsize=(13,5));im=ax.imshow(grid,cmap='viridis',vmin=0,vmax=3,aspect='auto')
    ax.set_yticks(range(7),TERRAIN_ORDER);ax.set_xticks(range(16),range(16))
    ax.set_xlabel('direction index');ax.set_title('0 eligible, 1 numeric, 2 physical termination, 3 invalid')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(figdir/'01-classification.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(list(counts),list(counts.values()))
    ax.tick_params(axis='x',rotation=30);ax.set_title('112-pose classification')
    fig.tight_layout();fig.savefig(figdir/'02-counts.png',dpi=140);plt.close(fig)
    steps=[x['termination_step'] for x in results if x['termination_step']]
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(steps,bins=range(0,211,10));ax.set_xlabel('physics step')
    ax.set_title('Physical termination steps');fig.tight_layout();fig.savefig(figdir/'03-terminations.png',dpi=140);plt.close(fig)
    report=['# Radial 112-pose termination-aware inventory','',f'**Decision:** `{decision}`','',
            f"A eligible: {counts['settled_anchor_eligible']}; B numeric: {counts['anchor_ineligible_numeric']}; "
            f"C physical termination: {counts['anchor_ineligible_physical_termination']}; D invalid: {counts['measurement_invalid']}.",
            f'Candidate-bank shortlist: {len(shortlist)}; independent source families: {shortlist_families}; '
            f'candidate-bank review ready: {review_ready}.','',
            '| Terrain | A | B | C | D |','|---|---:|---:|---:|---:|']
    for t in TERRAIN_ORDER:
        c=terrain[t]['counts'];report.append(f"| {t} | {c.get('settled_anchor_eligible',0)} | "
            f"{c.get('anchor_ineligible_numeric',0)} | {c.get('anchor_ineligible_physical_termination',0)} | "
            f"{c.get('measurement_invalid',0)} |")
    report+=['',f'Missing interaction classes: {missing}. Physical terminations are anchor-ineligible measurements, '
             'not candidate Path outcomes, safety labels, or real-robot incidents. Contact penetration is descriptive only.',
             'The 0–0.6 m corridor is a 2 s reach proxy at 0.30 m/s; no candidate Path was run.',
             'The 16 directions within one terrain share the same seed and source family.','',
             'Old 75 poses were reused read-only with seal and SHA checks. The old partial shard-15 trace remained invalid.',
             'Second seed, Path, follower, bridge, router, Outcome, and learning were not run.','']
    (root/'report.md').write_text('\n'.join(report))


if __name__=='__main__':analyze(Path(sys.argv[1]))
