"""Verify the zero-step scan and freeze R3's second-seed pilot before physics."""
from __future__ import annotations

from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/tmp/eatnav-r3-second-seed-20261008')
WORKTREE=Path('/tmp/eatnav-vnt-r3-second-seed')
sys.path.insert(0,str(WORKTREE/'src'))
from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER,terrain_seed
from visualnav_transformer.eat_nav.anchor_placement_rules import select_r3,CLASS_ORDER
from visualnav_transformer.eat_nav.r3_second_selection import choose_pilot
from visualnav_transformer.eat_nav.r3_robustness import OFFSETS_LOCAL_M,YAW_OFFSETS_DEG,summarize_perturbations
from visualnav_transformer.eat_nav.r3_anchor_contract import requested_world


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise RuntimeError('frozen result already exists: '+str(path))
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def verify_items() -> tuple[list[dict],dict]:
    summary=json.loads((ROOT/'scan-v1/scan-summary.json').read_text())
    status=json.loads((ROOT/'scan-status.json').read_text())
    seal=json.loads((ROOT/'scan-seal.json').read_text())
    if (status['stage']!='complete' or status['native_exit_code']!=0 or status['owned_residual'] is not False or
        sha(ROOT/'scan-seal.json')!=status['seal_sha256'] or
        seal['sha_ledger_sha256']!=sha(ROOT/'scan-sha-ledger.json') or
        summary['candidate_count']!=448 or summary['dem_complete']!=448 or
        summary['robustness_complete']!=448 or summary['physics_steps_after_scene_init']!=0):
        raise RuntimeError('zero-step scan seal invalid')
    ledger=json.loads((ROOT/'scan-sha-ledger.json').read_text())['files']
    for item in ledger:
        if sha(ROOT/item['path'])!=item['sha256']:
            raise RuntimeError('scan artifact SHA mismatch: '+item['path'])
    seed=json.loads((ROOT/'seed-freeze.json').read_text())['batch_seed']
    items=[];mesh=set()
    for radius in (.30,.40,.50,.60):
        path=ROOT/f'scan-v1/radius-{radius:.2f}.json'
        payload=json.loads(path.read_text())
        if payload['radius_m']!=radius or payload['physics_steps_after_scene_init']!=0 or len(payload['candidates'])!=112:
            raise RuntimeError('radius scan geometry count/step invalid')
        mesh.add(payload['terrain_mesh_sha256'])
        for x in payload['candidates']:
            col=TERRAIN_ORDER.index(x['terrain']);angle=x['direction_index']
            expected_theta=angle*math.pi/8
            if (x['terrain_seed']!=terrain_seed(seed,col) or x['source_family']!=f"{seed}:{x['terrain']}" or
                x['env_id']!=angle*7+col or x['radius_m']!=radius or
                x['terrain_difficulty']!=.5 or x['frame_id']!='map' or
                x['axis']!='robot-relative-grid-xy' or x['resolution_m']!=.1 or
                x['dem_shape']!=[2,20,20] or not x['dem_complete'] or
                abs(x['theta_rad']-expected_theta)>1e-5 or
                len(x['robustness_conditions'])!=15):
                raise RuntimeError('second-seed pose/frame contract invalid: '+x['pose_id'])
            dem=np.asarray(x['dem'],dtype=np.float32)
            if dem.shape!=(2,20,20) or not np.isfinite(dem).all() or hashlib.sha256(dem.tobytes()).hexdigest()!=x['dem_sha256']:
                raise RuntimeError('second-seed DEM invalid: '+x['pose_id'])
            requested_world({'requested_xy_yaw':x['requested_xy_yaw'],
                'theta_rad':expected_theta,'radius_m':radius},x['env_origin_xyz'])
            conditions=x['robustness_conditions']
            for index,(offset,yaw) in enumerate((o,y) for o in OFFSETS_LOCAL_M for y in YAW_OFFSETS_DEG):
                c=conditions[index]
                if c['offset_local_m']!=list(offset) or c['yaw_offset_deg']!=yaw:
                    raise RuntimeError('perturbation grid/order changed')
            if conditions[0]['dem_sha256']!=x['dem_sha256'] or summarize_perturbations(conditions)!=x['robustness']:
                raise RuntimeError('nominal perturbation or summary drift')
            items.append(x)
    if len(items)!=448 or len({x['pose_id'] for x in items})!=448 or len(mesh)!=1 or mesh.pop()!=summary['terrain_mesh_sha256']:
        raise RuntimeError('448-pose identity/terrain mesh invalid')
    return items,summary


def main():
    items,summary=verify_items()
    selected=select_r3(items)
    pilot,missing=choose_pilot(selected)
    counts={'all':dict(Counter(x['classification'] for x in items)),
            'r3_112':dict(Counter(x['classification'] for x in selected)),
            'by_radius':{str(r):dict(Counter(x['classification'] for x in items if x['radius_m']==r)) for r in (.30,.40,.50,.60)},
            'by_terrain':{t:dict(Counter(x['classification'] for x in items if x['terrain']==t)) for t in TERRAIN_ORDER},
            'known_onset_all':sum(x['interaction_onset_m'] is not None for x in items),
            'source_families_by_class':{c:sorted({x['source_family'] for x in selected if x['classification']==c}) for c in CLASS_ORDER}}
    out=ROOT/'selection';out.mkdir(exist_ok=False)
    write_new(out/'class-support.json',counts)
    if len(selected)!=112 or missing or len(pilot)!=8:
        write_new(out/'decision.json',{'decision':'r3_second_seed_offline_class_supply_insufficient',
            'selected_count':len(selected),'missing_classes':missing,'pilot_count':len(pilot)})
        print(json.dumps(json.loads((out/'decision.json').read_text()),indent=2));return
    ranks={x['pose_id']:i for i,x in enumerate(selected,1)}
    pose_rows=[]
    for slot,x in enumerate(pilot,1):
        col=TERRAIN_ORDER.index(x['terrain']);angle=x['direction_index']
        row={'pose_id':x['pose_id'],'batch_seed':int(x['source_family'].split(':')[0]),
            'terrain':x['terrain'],'terrain_seed':x['terrain_seed'],
            'source_family':x['source_family'],'row':angle,'column':col,
            'env_index':x['env_id'],'angle_index':angle,'theta_rad':angle*math.pi/8,
            'radius_m':x['radius_m'],'requested_xy_yaw':x['requested_xy_yaw'],
            'virtual_dem_sha256':x['dem_sha256'],'virtual_class':x['classification'],
            'virtual_interaction_onset_m':x['interaction_onset_m'],
            'rank':ranks[x['pose_id']],'pilot_slot':slot,
            'robustness':x['robustness'],'robustness_conditions':x['robustness_conditions'],
            'virtual_dem_valid_fraction':x['dem_valid_fraction'],
            'virtual_corridor_known_fraction':x['corridor_known_fraction'],
            'virtual_unknown_only_difference':x['unknown_only_difference'],
            'virtual_footprint_diagnostic':x['footprint_diagnostic']}
        if x['dem_valid_fraction']<.7 or x['unknown_only_difference']:
            raise RuntimeError('R3-selected pilot DEM coverage/unknown-only gate failed')
        pose_rows.append(row)
    a,b=pose_rows[:4],pose_rows[4:]
    for shard in (a,b):
        if len({x['env_index'] for x in shard})!=4:
            raise RuntimeError('R3-selected pilot has duplicate environment within shard')
    scan_shas={f'radius-{r:.2f}.json':sha(ROOT/f'scan-v1/radius-{r:.2f}.json') for r in (.30,.40,.50,.60)}
    write_new(out/'pilot-manifest.json',{'schema':'eatnav_r3_second_pilot_v1',
        'selection_order':'R3 original rank; first distinct family per class in shard A, second in shard B',
        'selection_uses_robustness':False,'scan_sha256':scan_shas,
        'r3_source_sha256':sha(WORKTREE/'src/visualnav_transformer/eat_nav/anchor_placement_rules.py'),
        'classifier_source_sha256':sha(WORKTREE/'src/visualnav_transformer/eat_nav/radial_interaction.py'),
        'seed_freeze_sha256':sha(ROOT/'seed-freeze.json'),
        'robustness_freeze_sha256':sha(ROOT/'robustness-freeze.json'),
        'poses':pose_rows})
    reserve=[];chosen={x['pose_id'] for x in pilot}
    for rank,x in enumerate(selected,1):
        if x['pose_id'] not in chosen:
            reserve.append({'r3_rank':rank,'pose_id':x['pose_id'],'virtual_class':x['classification'],
                'source_family':x['source_family'],'status':'reserve_only_not_executable'})
    write_new(out/'reserve-rank-freeze.json',{'schema':'eatnav_r3_second_reserve_v1',
        'not_executable':True,'poses':reserve})
    write_new(out/'selection-sha.json',{'pilot_manifest_sha256':sha(out/'pilot-manifest.json'),
        'reserve_rank_sha256':sha(out/'reserve-rank-freeze.json'),
        'scan_summary_sha256':sha(ROOT/'scan-v1/scan-summary.json'),
        'class_support_sha256':sha(out/'class-support.json')})
    write_new(out/'decision.json',{'decision':'offline_preflight_pending',
        'pilot_manifest_sha256':sha(out/'pilot-manifest.json'),
        'selected_count':len(selected),'pilot_count':len(pose_rows),
        'reserve_count':len(reserve),'terrain_mesh_sha256':summary['terrain_mesh_sha256']})
    print(json.dumps({'decision':'offline_preflight_pending','class_counts':counts['all'],
        'pilot':[(x['pilot_slot'],x['rank'],x['virtual_class'],x['pose_id']) for x in pose_rows],
        'manifest_sha256':sha(out/'pilot-manifest.json')},indent=2))

if __name__=='__main__':main()
