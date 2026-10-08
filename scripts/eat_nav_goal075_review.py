#!/usr/bin/env python3
"""Offline 0.75 m common-goal candidate-bank review; no live processes."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import tarfile
from collections import Counter
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

from scripts.eat_nav_curvature_attainable import (
    verify_anchor_identity, verify_base_sources, _read_follower_config,
    _source_exact_audit, FOLLOWER_CONFIG, load_anchor_inputs, sha, write, CONFIG as PRIOR_CONFIG,
    separation,
)
from scripts.eat_nav_goal_bank_review import plain
from scripts.eat_nav_goal_bank_redesign import _forecast, _prospective_P1_P2_gates
from visualnav_transformer.eat_nav.curvature_attainable import (
    generate_goal_library, solve_distinct_modes, solve_symmetric_target,
)
from visualnav_transformer.eat_nav.goal075_offline import shared_symmetric_bank
from visualnav_transformer.eat_nav.goal_bank_offline import goal_for_bearing
from visualnav_transformer.eat_nav.goal_bank_offline import pairwise_audit, corridor_audit
from visualnav_transformer.eat_nav.goal_bank_offline import corridor_class_proxy
from visualnav_transformer.eat_nav.goal075_offline import (
    contract075, goal075, audit_candidate, side_relative_to_direct,
    variable_count_mask, prefix_row, coverage_record, generate_goal075_library,
    class_proxy_interval,
)

BASE_COMMIT='bb85939e716376017b909e564938ec00b11e849b'
ARCHIVES={
    'goal060':('eatnav-goal-bearing-bank-review-20261008-r6.tar.gz',
               'eca4650875d4972827010bbbe4cd970cb3b06b4d0b9cfcdf2516d1a204de1e6b',
               'eatnav-goal-bearing-bank-review-20261008-r6','sha-ledger.json',
               ('eight-anchor-freeze.json','state-groups.json','runtime-storage-forecast.json','final-decision.json')),
    'redesign060':('eatnav-goal-bank-redesign-20261008-r4.tar.gz',
                   '36a6b011510a633c16b0bc9ba35b8c7d941e87739dd68cb9b758308b8e619c59',
                   'eatnav-goal-bank-redesign-20261008-r4','sha-ledger.json',
                   ('selected-candidate-definitions.json','runtime-storage-forecast.json','report.md','final-decision.json')),
    'curvature060':('eatnav-curvature-attainable-20261008-r9.tar.gz',
                     'b3f0dc4a102838a8a6092a1012d550414893ca498c88c61956e543f700f3d4ac',
                     'eatnav-curvature-attainable-20261008-r9','artifact-sha-ledger.json',
                     ('source-provenance.json','attainable-set-config.json','final-decision.json',
                      'candidate-attainable-counts.json','curve-library-records.json',
                      'compatible-distinct-pairs.json','artifact-sha-ledger.json')),
}


def _archive_audit(backup_root:Path,spec:tuple) -> tuple[dict,dict]:
    name,expected,root,ledger_name,needed=spec
    archive=backup_root/name
    if sha(archive)!=expected:
        raise ValueError('prior archive SHA mismatch: '+name)
    saved={}
    with tarfile.open(archive,'r:gz') as bundle:
        ledger_bytes=bundle.extractfile(root+'/'+ledger_name).read()
        ledger=json.loads(ledger_bytes)['files']
        for relative,record in ledger.items():
            member=bundle.extractfile(root+'/'+relative)
            if member is None:
                raise ValueError('missing archived artifact: '+relative)
            body=member.read()
            digest=hashlib.sha256(body).hexdigest()
            want=record['sha256'] if isinstance(record,dict) else record
            if digest!=want or isinstance(record,dict) and len(body)!=record['size_bytes']:
                raise ValueError('archived artifact SHA/size mismatch: '+relative)
            if relative in needed:
                saved[relative]=json.loads(body) if relative.endswith('.json') else body.decode()
        if ledger_name in needed:saved[ledger_name]=json.loads(ledger_bytes)
    if set(needed)-set(saved):
        raise ValueError('archived required artifact missing: '+str(set(needed)-set(saved)))
    info={'archive':str(archive),'archive_sha256':expected,
          'ledger_sha256':hashlib.sha256(ledger_bytes).hexdigest(),
          'verified_artifact_count':len(ledger)}
    return info,saved


def verify_inputs(backup_root):
    backup_root=Path(backup_root)
    archives={};data={}
    for key,spec in ARCHIVES.items():
        archives[key],data[key]=_archive_audit(backup_root,spec)
    anchors,anchor_source=load_anchor_inputs(backup_root)
    anchor_check=verify_anchor_identity(anchors,data['goal060']['eight-anchor-freeze.json'])
    groups=data['goal060']['state-groups.json']
    if len(groups)!=40:
        raise ValueError('40 frozen StateGroups required')
    source_exact=_source_exact_audit(backup_root,_read_follower_config(FOLLOWER_CONFIG))
    root=Path(__file__).resolve().parents[1]
    previous_source=data['curvature060']['source-provenance.json']
    current_generator=sha(root/'src/visualnav_transformer/eat_nav/curvature_attainable.py')
    current_runner=sha(root/'scripts/eat_nav_curvature_attainable.py')
    generator_sha_match=(current_generator==previous_source['attainable_generator_source_sha256']
                         and current_runner==previous_source['attainable_runner_source_sha256'])
    if not generator_sha_match:
        raise ValueError('previous G0–G3 generator or runner source SHA changed')
    if data['curvature060']['final-decision.json']['decision']!='candidate_diversity_infeasible_under_frozen_follower_contract':
        raise ValueError('unexpected 0.60m curvature decision')
    base_sources=verify_base_sources(root)
    e1_gate=root/'docs/quality-identifiability-gate/candidate-coverage/e1-feasibility/e1-gates.yaml'
    provenance={'base_commit':BASE_COMMIT,'archives':archives,
                'prior_curvature_decision':data['curvature060']['final-decision.json']['decision'],
                'generator_sha_match':generator_sha_match,
                'generator_source_sha256':current_generator,
                'prior_runner_source_sha256':current_runner,
                'new_goal075_generator_sha256':sha(root/'src/visualnav_transformer/eat_nav/goal075_offline.py'),
                'new_goal075_review_runner_sha256':sha(root/'scripts/eat_nav_goal075_review.py'),
                'base_decision_source_sha256':base_sources,
                'frozen_e1_gate_sha256':sha(e1_gate),
                'source_exact_follower':source_exact,
                'anchor_identity':anchor_check,
                'anchor_source':anchor_source,
                'current_worktree_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                'current_worktree_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True).splitlines(),
                'input_roots_from_durable_backup_only':True}
    return {'anchors':anchors,'old_groups':groups,'prior':data,'provenance':provenance}


def freeze_contract(output_root):
    output_root=Path(output_root)
    if output_root.exists():
        raise FileExistsError('new output root required: '+str(output_root))
    output_root.mkdir(parents=True)
    contract={**contract075(),
              'mode_separation':PRIOR_CONFIG['mode_separation'],
              'known_terrain_rms_min_m':PRIOR_CONFIG['known_corridor_rms_min_m'],
              'known_terrain_common_bins_min':PRIOR_CONFIG['known_corridor_common_bins_min'],
              'footprint_proxies':['main_track_sprocket','initial_flipper_support',
                                   'articulated_reach','full_collision_union'],
              'primary_bank_fixed_five':True,
              'variable_count_diagnostic_only':True,
              'no_candidate_outcome_used':True}
    write(output_root/'goal-path-contract-075m.json',contract)
    return {'contract':contract,'contract_sha256':sha(output_root/'goal-path-contract-075m.json')}


def verify_g0_regression_archive(old_groups,definitions):
    if len(old_groups)!=40 or len(definitions)!=560:
        raise ValueError('unexpected frozen G0 regression count')
    by_group={row['state_group_id']:row for row in old_groups}
    libraries={bearing:generate_goal_library(goal_for_bearing(bearing))[0]
               for bearing in (-20,-10,0,10,20)}
    matched=0
    for item in definitions:
        group=by_group[item['state_group_id']]
        family='G0A' if item['bank']=='A' else 'G0B'
        variant=f"{item['bank']}-{item['side']}-{abs(float(item['signed_amplitude_or_offset_m'])):.2f}"
        curve=next((c for c in libraries[group['bearing_deg']]
                    if c['family']==family and c['variant_id']==variant),None)
        if curve is None or curve['path_sha256']!=item['path_sha256']:
            raise ValueError('prior 0.60m G0 Path SHA mismatch: '+item['state_group_id']+':'+variant)
        matched+=1
    return {'matched_selected_prior_paths':matched,'source_exact_path_sha':True}


def analyze_group_075(anchor,bearing_deg,curves,failures,follower_config):
    goal=goal075(bearing_deg)
    common=[curve for curve in curves if curve['family']!='G0B']
    direct=next((curve for curve in common if curve['family']=='G2' and curve['variant_id']=='constant'),None)
    if direct is None:
        raise ValueError('frozen forward-tangent direct G2 constant missing')
    all_records=[];compatible=[]
    for curve in common:
        result=audit_candidate(curve,goal,anchor,follower_config)
        result['side'],result['signed_relative_lateral_m']=side_relative_to_direct(curve,direct,goal)
        if curve is direct:result['side']='direct'
        all_records.append(result)
        if result['compatible']:
            compatible.append((curve,result))
    pair_rows=[];conflicts=[]
    for i,j in itertools.combinations(range(len(compatible)),2):
        ca,ra=compatible[i];cb,rb=compatible[j]
        metric=separation(ca,cb,ra['dem_full'],rb['dem_full'],goal)
        pair_rows.append({'first':ra['candidate_id'],'second':rb['candidate_id'],**metric})
        if not metric['distinct']:conflicts.append((i,j))
    maximum=solve_distinct_modes(len(compatible),conflicts)
    direct_id='G2:constant'
    direct_index=next((i for i,(_,r) in enumerate(compatible) if r['candidate_id']==direct_id),None)
    sides=[r['side'] for _,r in compatible]
    targets={str(1+2*n):solve_symmetric_target(sides,conflicts,pair_count=n,direct_index=direct_index)
             if direct_index is not None else {'feasible':False,'selected_indices':[],
                                                'solver_status':'direct_incompatible'}
             for n in (1,2,3)}
    group_id=f"{anchor['anchor_id']}-b{bearing_deg:+03d}"
    return {'state_group_id':group_id,'anchor_id':anchor['anchor_id'],
            'seed_label':anchor['seed_label'],'terrain_class':anchor['terrain_class'],
            'source_family':anchor['source_family'],'bearing_deg':bearing_deg,
            'goal_base_xy_m':goal.tolist(),'direct_id':direct_id,
            'generated_common_goal':len(common),'construction_failures':failures,
            'hard_valid_count':sum(r['hard_valid'] for r in all_records),
            'source_compatible_count':len(compatible),
            'compatible_by_family':dict(Counter(r['family'] for _,r in compatible)),
            'rejection_reasons':dict(Counter(reason for r in all_records for reason in r['reasons'])),
            'rejection_reasons_by_family':{
                family:dict(Counter(reason for r in all_records if r['family']==family
                                    for reason in r['reasons']))
                for family in ('G0A','G1','G2','G3')},
            'hard_valid_ids':[r['candidate_id'] for r in all_records if r['hard_valid']],
            'compatible_ids':[r['candidate_id'] for _,r in compatible],
            'side_by_id':{r['candidate_id']:r['side'] for _,r in compatible},
            'maximum_distinct':maximum,'symmetric_targets':targets,
            'pair_count':len(pair_rows),'conflict_count':len(conflicts),
            '_records':all_records,'_compatible':compatible,'_pair_rows':pair_rows,
            '_conflicts':conflicts}


def terrain_support_gate(groups,pairs_by_group):
    support={cls:{seed:0 for seed in ('first','second')}
             for cls in ('transition','differential','graded')}
    anchor_ids={cls:{seed:set() for seed in ('first','second')}
                for cls in support}
    for group in groups:
        cls=group['terrain_class'];seed=group['seed_label']
        if cls not in support:continue
        anchor_ids[cls][seed].add(group.get('anchor_id',group['state_group_id']))
        if any(row['distinct'] and row['primary_separated'] for row in pairs_by_group.get(group['state_group_id'],[])):
            support[cls][seed]+=1
    physical_supply=all(anchor_ids[cls][seed] for cls in support for seed in ('first','second'))
    return {'ready':physical_supply and all(support[cls][seed]>0 for cls in support for seed in ('first','second')),
            'support':support,'physical_anchor_count':{cls:{seed:len(anchor_ids[cls][seed])
                                              for seed in ('first','second')} for cls in support},
            'late_only_pairs_never_count':True}


def baseline_distinct_terrain_count(pairs):
    return sum(row['distinct'] and row['terrain_separated'] for row in pairs)


def recompute_baseline_primary_terrain(frozen):
    """Re-audit the sealed 0.60 m curves on exactly 0–0.60 m of arc length."""
    old_records=frozen['prior']['curvature060']['curve-library-records.json']
    old_pairs=frozen['prior']['curvature060']['compatible-distinct-pairs.json']
    records_by=defaultdict(list);pairs_by=defaultdict(list)
    for row in old_records:records_by[row['state_group_id']].append(row)
    for row in old_pairs:pairs_by[row['state_group_id']].append(row)
    anchor_by={a['anchor_id']:a for a in frozen['anchors']}
    libraries={b:{(c['family'],c['variant_id']):c
                  for c in generate_goal_library(goal_for_bearing(b))[0]}
               for b in (-20,-10,0,10,20)}
    result={}
    for group in frozen['old_groups']:
        identifier=group['state_group_id'];anchor=anchor_by[group['anchor_id']]
        curves=libraries[group['bearing_deg']]
        rows={}
        for record in records_by[identifier]:
            if not record['combined_compatible'] or record['family']=='G0B':continue
            curve=curves[(record['family'],record['variant_id'])]
            row=prefix_row({'dense':curve['dense'],'stations_m':curve['stations_m'],
                            'arc_length_m':curve['arc_length_m']},.6)
            dem=corridor_audit(row,anchor['dem'][0],anchor['dem'][1],anchor['proxy_bboxes'])
            rows[record['variant_id']]=(row,dem)
        counted=[];missing=[]
        for pair in pairs_by[identifier]:
            if not pair['distinct'] or pair['first_family']=='G0B' or pair['second_family']=='G0B':
                continue
            if pair['first'] not in rows or pair['second'] not in rows:
                missing.append([pair['first'],pair['second']]);continue
            a,da=rows[pair['first']];b,db=rows[pair['second']]
            metrics=pairwise_audit(a,b,da,db)
            separated=(metrics['common_known_corridor_bins']>=5 and
                       metrics['known_corridor_profile_rms_m'] is not None and
                       metrics['known_corridor_profile_rms_m']>=.01)
            counted.append({'first':pair['first'],'second':pair['second'],
                            'primary_separated':bool(separated),
                            'common_known_corridor_bins':metrics['common_known_corridor_bins'],
                            'known_corridor_profile_rms_m':metrics['known_corridor_profile_rms_m']})
        if missing:raise ValueError('old compatible pair references missing curve: '+identifier)
        result[identifier]={'arc_end_m':.6,'mode_distinct_pair_count':len(counted),
                            'primary_terrain_separated_pair_count':sum(p['primary_separated'] for p in counted),
                            'pairs':counted}
    if len(result)!=40:raise ValueError('40 baseline primary terrain audits required')
    return result


def variable_group_diagnostic(group):
    target=group['symmetric_targets']['5'] if group['symmetric_targets']['5']['feasible'] else group['symmetric_targets']['3']
    if not target['feasible']:
        return {'available':False,'reason':'fewer_than_three_balanced_distinct_modes',
                'candidate_ids':[],'valid_mask':[]}
    selected=[group['compatible_ids'][i] for i in target['selected_indices']]
    return {'available':True,**variable_count_mask(selected,group['direct_id'])}


def terrain_pair_from_rows(row_a,row_b,primary_a,primary_b,full_a,full_b,*,mode_distinct):
    def clipped_dem(dem,end):
        copy={**dem,'known_height_profile_m':[p for p in dem['known_height_profile_m'] if p[0]<=end+1e-9]}
        main=dem['corridors']['main_track_sprocket']
        copy['corridors']={**dem['corridors'],'main_track_sprocket':{
            **main,'known_corridor_profile_m':[p for p in main['known_corridor_profile_m'] if p[0]<=end+1e-9]}}
        return copy
    primary=pairwise_audit(prefix_row(row_a,.6),prefix_row(row_b,.6),
                           clipped_dem(primary_a,.6),clipped_dem(primary_b,.6))
    full=pairwise_audit(prefix_row(row_a,.75),prefix_row(row_b,.75),
                        clipped_dem(full_a,.75),clipped_dem(full_b,.75))
    def separated(metric):
        return (mode_distinct and metric['common_known_corridor_bins']>=5
                and metric['known_corridor_profile_rms_m'] is not None
                and metric['known_corridor_profile_rms_m']>=.01)
    def difference(first,second,key):
        a=first['corridors']['main_track_sprocket'].get(key)
        b=second['corridors']['main_track_sprocket'].get(key)
        return abs(float(a)-float(b)) if a is not None and b is not None else None
    return {'distinct':bool(mode_distinct),
            'primary_separated':bool(separated(primary)),
            'full_separated':bool(separated(full)),
            'primary':primary,'full_diagnostic':full,
            'primary_path_aligned_difference':{
                name:difference(primary_a,primary_b,key) for name,key in
                (('slope','corridor_maximum_slope'),('step','corridor_maximum_step_m'),
                 ('roughness','corridor_roughness_m'))},
            'full_path_aligned_difference':{
                name:difference(full_a,full_b,key) for name,key in
                (('slope','corridor_maximum_slope'),('step','corridor_maximum_step_m'),
                 ('roughness','corridor_roughness_m'))},
            'late_only':bool(separated(full) and not separated(primary))}


def enrich_group_terrain(group,anchor):
    """Attach two interval terrain and four-footprint coverage to compatible curves."""
    by_id={}
    for curve,record in group['_compatible']:
        row={'dense':curve['dense'],'stations_m':curve['stations_m'],
             'arc_length_m':curve['arc_length_m']}
        coverage=coverage_record(row,anchor['dem'][0],anchor['dem'][1],anchor['proxy_bboxes'])
        class_proxy=corridor_class_proxy(row,anchor['dem'][0],anchor['dem'][1])
        full_class_proxy=class_proxy_interval(row,anchor['dem'][0],anchor['dem'][1],.75)
        record['coverage']={key:value for key,value in coverage.items() if not key.startswith('_')}
        record['path_aligned_class_proxy']=class_proxy
        record['path_aligned_class_proxy_full_diagnostic']=full_class_proxy
        record['interaction_onset_primary_m']=coverage['_dem_primary']['interaction_onset_station_m']
        record['interaction_onset_full_m']=coverage['_dem_full']['interaction_onset_station_m']
        by_id[record['candidate_id']]=(row,coverage,record)
    pairs=[]
    for pair in group['_pair_rows']:
        first,second=pair['first'],pair['second']
        a,ac,_=by_id[first];b,bc,_=by_id[second]
        metrics=terrain_pair_from_rows(a,b,ac['_dem_primary'],bc['_dem_primary'],
                                       ac['_dem_full'],bc['_dem_full'],
                                       mode_distinct=pair['distinct'])
        pairs.append({'first':first,'second':second,**metrics})
    group['_terrain_pairs']=pairs
    group['primary_terrain_separated_pair_count']=sum(row['primary_separated'] for row in pairs)
    group['full_terrain_separated_pair_count']=sum(row['full_separated'] for row in pairs)
    group['late_only_terrain_pair_count']=sum(row['late_only'] for row in pairs)
    group['compatible_coverage_summary']={identifier:record['coverage']
                                          for identifier,(_,_,record) in by_id.items()}
    group['path_aligned_class_counts']=dict(Counter(record['path_aligned_class_proxy']['class_proxy']
                                                     for _,_,record in by_id.values()))
    return group


def decide_075(source_valid,geometry_ready,terrain_ready,coverage_ready):
    if not source_valid:
        return 'source_or_geometry_parity_invalid'
    if not geometry_ready:
        return 'five_candidate_075m_still_infeasible'
    if not coverage_ready:
        return 'five_candidate_075m_coverage_fail'
    if not terrain_ready:
        return 'five_candidate_075m_geometry_pass_terrain_support_fail'
    return 'five_candidate_075m_bank_ready_for_p1_rollout_review'


def conditional_p1_proposal(ready):
    proposal={'prepared':bool(ready),'execution_authorized':False,
              'P1_anchor_count':4,'P1_bearings_deg':[0],
              'P1_candidates_per_group':5,'direct_repeats_per_group':3,
              'non_direct_per_group':4,'processes_per_group':7,
              'P1_processes':28,'P2_bearings_deg':[-20,20],
              'P2_processes':56,'retry_count':0,
              'teacher_window_s':[0,2],
              'external_zenoh_operator_owned':True,
              'candidate_free_X2_before_pilot_clock':True,
              'P2_conditional_on_P1_review':True}
    if not ready:
        proposal['reason']='offline fixed-five geometry/terrain/coverage gates not all passed'
    return proposal


def paired_baseline(old_counts,old_records,old_pairs,new_groups,old_primary):
    """Definition-level comparison; StateGroup is the unit, never each curve."""
    previous={row['state_group_id']:row for row in old_counts}
    records=defaultdict(list);pairs=defaultdict(list)
    for row in old_records:records[row['state_group_id']].append(row)
    for row in old_pairs:pairs[row['state_group_id']].append(row)
    output=[]
    for group in new_groups:
        identifier=group['state_group_id'];old=previous[identifier]
        old_compatible=[r for r in records[identifier] if r['combined_compatible'] and r['family']!='G0B']
        new_compatible=[r for _,r in group['_compatible']]
        def stats(rows,curvature_key,command_key,old_version):
            speeds=[abs(float(r[command_key]['initial_command_estimate']['angular_z']))
                    for r in rows if r[command_key]['initial_command_estimate']['angular_z'] is not None]
            terrain=[r['terrain'] if old_version else r['coverage'] for r in rows]
            if old_version:
                classes=[r['terrain']['path_aligned_class_proxy'] for r in rows]
                known=[r['terrain']['centerline_known_fraction'] for r in rows]
                unknown=[r['terrain']['centerline_unknown_fraction'] for r in rows]
                outside=None
            else:
                classes=[r['path_aligned_class_proxy']['class_proxy'] for r in rows]
                known=[r['coverage']['full_centerline_known_fraction'] for r in rows]
                unknown=[r['coverage']['full_centerline_unknown_fraction'] for r in rows]
                outside=[r['coverage']['full_centerline_outside_fraction'] for r in rows]
            return {'max_curvature_per_m':max((r[curvature_key] for r in rows),default=None),
                    'max_abs_first_yaw_command_rad_s':max(speeds,default=None),
                    'turn_in_place_estimate_count':sum(r[command_key]['turn_in_place_entry'] for r in rows),
                    'reverse_estimate_count':sum(r[command_key]['reverse_estimate'] for r in rows),
                    'mean_centerline_known_fraction':float(np.mean(known)) if known else None,
                    'mean_centerline_unknown_fraction':float(np.mean(unknown)) if unknown else None,
                    'mean_centerline_outside_fraction':float(np.mean(outside)) if outside else None,
                    'path_aligned_class_counts':dict(Counter(str(c) for c in classes)),
                    'count':len(rows)}
        old_stats=stats(old_compatible,'analytic_or_dense_max_curvature_per_m','first_command',True)
        new_stats=stats(new_compatible,'dense_max_curvature_per_m','follower',False)
        new_pair=group['_terrain_pairs']
        old_pair=pairs[identifier]
        output.append({'state_group_id':identifier,'anchor_id':group['anchor_id'],
                       'seed_label':group['seed_label'],'source_family':group['source_family'],
                       'terrain_class':group['terrain_class'],'bearing_deg':group['bearing_deg'],
                       'baseline_060':{'compatible_common_goal':old['compatible_common_goal'],
                           'compatible_after_new_forward_tangent_screen_diagnostic':sum(
                               r['initial_tangent_error_rad']<=1e-6 for r in old_compatible),
                           'G0A_nonforward_compatible_exempted_by_frozen_baseline':sum(
                               r['family']=='G0A' and r['initial_tangent_error_rad']>1e-6
                               for r in old_compatible),
                           'maximum_distinct':old['maximum_distinct_finite_library']['maximum_mode_count'],
                           'symmetric':{str(n):old['symmetric_direct_plus_pairs'][str(n)]['feasible']
                                        for n in (3,5,7)},
                           'terrain_separated_pairs_primary_060':old_primary[identifier]['primary_terrain_separated_pair_count'],
                           'terrain_separated_pairs_archived_full_path_diagnostic':baseline_distinct_terrain_count(old_pair),
                           'terrain_separated_but_not_mode_distinct_diagnostic':sum(
                               p['terrain_separated'] and not p['distinct'] for p in old_pair),
                           'compatible_summary':old_stats,'endpoint_distance_m':.6,
                           'nominal_endpoint_time_s':2.0},
                       'revised_075':{'compatible_common_goal':group['source_compatible_count'],
                           'maximum_distinct':group['maximum_distinct']['maximum_mode_count'],
                           'symmetric':{str(n):group['symmetric_targets'][str(n)]['feasible']
                                        for n in (3,5,7)},
                           'terrain_separated_pairs_primary_060':sum(p['primary_separated'] for p in new_pair),
                           'terrain_separated_pairs_full_075':sum(p['full_separated'] for p in new_pair),
                           'late_only_pairs':sum(p['late_only'] for p in new_pair),
                           'compatible_summary':new_stats,'endpoint_distance_m':.75,
                           'nominal_endpoint_time_s':2.5},
                       'teacher_window_s':[0,2]})
    if len(output)!=40 or len({r['state_group_id'] for r in output})!=40:
        raise ValueError('paired StateGroup coverage must be exactly 40')
    strata={}
    for key in ('seed_label','source_family','terrain_class','bearing_deg'):
        strata[key]={str(value):{'state_groups':sum(r[key]==value for r in output),
                                  'old_symmetric_five':sum(r[key]==value and r['baseline_060']['symmetric']['5'] for r in output),
                                  'new_symmetric_five':sum(r[key]==value and r['revised_075']['symmetric']['5'] for r in output)}
                     for value in sorted({r[key] for r in output})}
    return {'statistical_unit':'StateGroup; curves/pairs are definitions, not independent trials',
            'paired_state_groups':output,'strata':strata,
            'screen_comparability_caveat':'Frozen 0.60 m audit exempted G0A from forward-tangent rejection; revised 0.75 m audit does not. Baseline compatible and mode counts remain frozen; same-screen compatible count is diagnostic only.',
            'planning_horizon_m':[.6,.75],'nominal_endpoint_time_s':[2.,2.5],
            'unchanged_teacher_duration_s':2.,
            'old_primary_060_recomputed_from_SHA_frozen_curves_and_anchor_DEM':True,
            'archived_old_full_path_may_exceed_060_arc':True}


def _figure_set(output,groups,libraries,baseline,bank,coverage,terrain):
    """Ten compact diagnostic figures; no simulation or measured Outcome."""
    figdir=output/'figures';figdir.mkdir()
    representative=next(a for a in groups if a['anchor_id']==groups[0]['anchor_id'] and a['bearing_deg']==0)
    new_curves={c['family']+':'+c['variant_id']:c for c in libraries[0][0]}
    old_curves={c['family']+':'+c['variant_id']:c for c in generate_goal_library(goal_for_bearing(0))[0]}
    def save(name,title,drawer):
        fig,ax=plt.subplots(figsize=(7,5));drawer(ax);ax.set_title(title)
        fig.tight_layout();fig.savefig(figdir/name,dpi=140);plt.close(fig)
    def paths(ax):
        for label,curves,style in (('0.60 m',old_curves,'--'),('0.75 m',new_curves,'-')):
            for candidate in ('G2:constant','G0A:A-left-0.12','G0A:A-right-0.12'):
                if candidate in curves:
                    xy=curves[candidate]['dense'];ax.plot(xy[:,0],xy[:,1],style,label=label+' '+candidate)
        ax.axis('equal');ax.legend(fontsize=6)
    save('01-path-lengths.png','0.60 versus 0.75 m Path geometry',paths)
    def bearings(ax):
        for bearing in (-20,-10,0,10,20):
            goal=goal075(bearing);ax.plot([0,goal[0]],[0,goal[1]],label=f'{bearing:+d}°')
        ax.axis('equal');ax.legend()
    save('02-bearings.png','Five common-goal bearings',bearings)
    def bankplot(ax):
        ids=bank.get('0',{}).get('selected_ids',[])
        if not ids:ids=representative['compatible_ids'][:5]
        for candidate in ids:
            if candidate in new_curves:
                xy=new_curves[candidate]['dense'];ax.plot(xy[:,0],xy[:,1],label=candidate)
        ax.axis('equal');ax.legend(fontsize=6)
        if not bank.get('0',{}).get('feasible'):ax.text(.02,.96,'NO FIXED FIVE BANK',transform=ax.transAxes,va='top',color='red')
    save('03-bank-or-compatible.png','Selected bank, or compatible supply if infeasible',bankplot)
    def curvature(ax):
        for candidate in representative['compatible_ids'][:5]:
            curve=new_curves[candidate]
            ax.plot(curve['stations_m'],curve['signed_curvature_profile_per_m'],label=candidate)
        ax.axhline(1.236,color='red',ls='--');ax.axhline(-1.236,color='red',ls='--')
        ax.legend(fontsize=6)
    save('04-curvature.png','Curvature profiles and frozen screen',curvature)
    def modes(ax):
        vals=[sum(g['bearing_deg']==b and g['symmetric_targets']['5']['feasible'] for g in groups)
              for b in (-20,-10,0,10,20)]
        ax.bar(range(5),vals);ax.set_xticks(range(5),[-20,-10,0,10,20]);ax.set_ylim(0,8)
        ax.set_ylabel('groups with symmetric five / 8')
    save('05-attainable-modes.png','Five-mode supply by bearing',modes)
    for number,key,title in ((6,'primary_separated','Primary 0–0.60 m terrain pairs'),
                             (7,'full_separated','Diagnostic 0–0.75 m terrain pairs')):
        def terrainplot(ax,key=key):
            xs=list(range(len(groups)));ys=[sum(p[key] for p in g['_terrain_pairs']) for g in groups]
            ax.bar(xs,ys);ax.set_xlabel('StateGroup');ax.set_ylabel('known distinct pairs')
        save(f'{number:02d}-terrain.png',title,terrainplot)
    def coverageplot(ax):
        names=['main_track_sprocket','initial_flipper_support','articulated_reach','full_collision_union']
        vals=[np.mean([v['proxies'][name]['full']['known_fraction']
                       for g in groups for v in g['compatible_coverage_summary'].values()])
              if any(g['compatible_coverage_summary'] for g in groups) else 0 for name in names]
        ax.bar(range(4),vals);ax.set_xticks(range(4),['P1','P2','P3','P4']);ax.set_ylim(0,1)
        ax.set_ylabel('mean known fraction; descriptive only')
    save('08-coverage.png','DEM footprint proxy known coverage',coverageplot)
    def supply(ax):
        a=sum(g['symmetric_targets']['5']['feasible'] for g in groups)
        b=sum(g['symmetric_targets']['3']['feasible'] for g in groups)
        ax.bar(['fixed five','variable ≥3'],[a,b]);ax.set_ylim(0,40)
        ax.set_ylabel('StateGroups / 40')
    save('09-fixed-v-variable.png','Fixed five versus variable-count diagnostic',supply)
    def flow(ax):
        ax.axis('off')
        lines=['SOURCE SHA + C0 PARITY','↓','FIXED FIVE GEOMETRY','↓','PRIMARY TERRAIN + COVERAGE','↓',
               'P1 REVIEW ONLY IF ALL PASS','↓','P2 ONLY AFTER P1 REVIEW','NO ROLLOUT AUTHORIZED']
        ax.text(.5,.5,'\n'.join(lines),ha='center',va='center',fontsize=12)
    save('10-decision-flow.png','Conditional P1/P2 decision flow',flow)
    return sorted(p.name for p in figdir.glob('*.png'))


def run_review(backup_root,output_root):
    """Run only the frozen offline review; never launch a ROS or physics process."""
    backup_root=Path(backup_root);output_root=Path(output_root)
    frozen=verify_inputs(backup_root)
    freeze=freeze_contract(output_root)
    write(output_root/'source-provenance.json',frozen['provenance'])
    write(output_root/'input-freeze.json',{
        'eight_anchor_identity':frozen['provenance']['anchor_identity'],
        'anchor_source':frozen['provenance']['anchor_source'],
        'archives':frozen['provenance']['archives'],
        'source_commit':frozen['provenance']['current_worktree_head'],
        'worktree_status_before_new_paths':frozen['provenance']['current_worktree_status'],
        'contract_sha256':freeze['contract_sha256']})
    write(output_root/'follower-source-parity.json',frozen['provenance']['source_exact_follower'])
    parity=verify_g0_regression_archive(
        frozen['old_groups'],frozen['prior']['redesign060']['selected-candidate-definitions.json'])
    write(output_root/'g0-path-regression.json',parity)
    anchor_by={a['anchor_id']:a for a in frozen['anchors']}
    libraries={bearing:generate_goal075_library(goal075(bearing)) for bearing in (-20,-10,0,10,20)}
    library_rows={str(b):{'curves':[{key:plain(value) for key,value in curve.items()}
                                    for curve in libraries[b][0]],
                          'construction_failures':libraries[b][1]}
                  for b in libraries}
    write(output_root/'candidate-library-075m.json',{
        'source':'unchanged G0A/G1–G3 generator; G0B excluded from primary',
        'bearings':library_rows,'selection_uses_Outcome':False})
    follower=_read_follower_config(FOLLOWER_CONFIG)
    groups=[]
    for old in frozen['old_groups']:
        bearing=old['bearing_deg'];anchor=anchor_by[old['anchor_id']]
        group=analyze_group_075(anchor,bearing,*libraries[bearing],follower)
        if group['state_group_id']!=old['state_group_id']:
            raise ValueError('frozen StateGroup ID mismatch: '+old['state_group_id'])
        enrich_group_terrain(group,anchor)
        groups.append(group)
        print(f"{len(groups)}/40 {group['state_group_id']} hard={group['hard_valid_count']} "
              f"compatible={group['source_compatible_count']} distinct={group['maximum_distinct']['maximum_mode_count']}",flush=True)
    if len(groups)!=40:
        raise ValueError('40/40 StateGroups required')
    bank={}
    for bearing in (-20,-10,0,10,20):
        subset=[g for g in groups if g['bearing_deg']==bearing]
        first=subset[0]
        all_sides={r['candidate_id']:r['side'] for r in first['_records']}
        for group in subset[1:]:
            if any(r['side']!=all_sides[r['candidate_id']] for r in group['_records']):
                raise ValueError('candidate side changed across frozen anchors')
        ids=list(all_sides)
        compatible={g['state_group_id']:set(g['compatible_ids']) for g in subset}
        conflicts={g['state_group_id']:[(p['first'],p['second']) for p in g['_pair_rows'] if not p['distinct']]
                   for g in subset}
        bank[str(bearing)]=shared_symmetric_bank(ids,[all_sides[i] for i in ids],compatible,conflicts,
                                                 'G2:constant',pair_count=2)
    path_shas={b:{c['family']+':'+c['variant_id']:c['path_sha256'] for c in libraries[b][0]}
               for b in libraries}
    selected_sha_unique=all(not row['feasible'] or len({path_shas[int(b)][i] for i in row['selected_ids']})==5
                            for b,row in bank.items())
    optimal=all(g['maximum_distinct'].get('optimal') and
                all(t.get('feasible') is False or t.get('solver_status')==0
                    for t in g['symmetric_targets'].values()) for g in groups)
    geometry_ready=(optimal and selected_sha_unique and
                    all(g['hard_valid_count']>=5 and g['source_compatible_count']>=5 and
                        g['symmetric_targets']['5']['feasible'] for g in groups) and
                    all(row['feasible'] for row in bank.values()))
    mode_rows=[];primary_rows=[];full_rows=[];coverage_rows=[];variable_rows=[];candidate_audits=[]
    for g in groups:
        candidate_audits.extend({'state_group_id':g['state_group_id'],
                                 'candidate_id':r['candidate_id'],
                                 'family':r['family'],'variant_id':r['variant_id'],
                                 'hard_valid':r['hard_valid'],'compatible':r['compatible'],
                                 'reasons':r['reasons'],
                                 'dense_max_curvature_per_m':r['dense_max_curvature_per_m'],
                                 'eight_point_max_curvature_per_m':r['eight_point_max_curvature_per_m'],
                                 'endpoint_error_m':r['endpoint_error_m'],
                                 'initial_tangent_error_rad':r['initial_tangent_error_rad'],
                                 'first_command':r['follower']['initial_command_estimate'],
                                 'turn_in_place_estimate':r['follower']['turn_in_place_entry'],
                                 'reverse_estimate':r['follower']['reverse_estimate'],
                                 'path_sha256':r['geometry'].get('path_sha256'),
                                 'interaction_onset_primary_m':r.get('interaction_onset_primary_m'),
                                 'interaction_onset_full_m':r.get('interaction_onset_full_m'),
                                 'path_aligned_class_proxy':r.get('path_aligned_class_proxy'),
                                 'path_aligned_class_proxy_full_diagnostic':r.get('path_aligned_class_proxy_full_diagnostic'),
                                 'coverage':r.get('coverage')}
                                for r in g['_records'])
        mode_rows.append({k:v for k,v in g.items() if k not in
                          ('_records','_compatible','_pair_rows','_conflicts','_terrain_pairs',
                           'compatible_coverage_summary')})
        primary_rows.append({'state_group_id':g['state_group_id'],
                             'pairs':[{k:v for k,v in p.items() if k not in ('full','full_diagnostic')}
                                      for p in g['_terrain_pairs']]})
        full_rows.append({'state_group_id':g['state_group_id'],
                          'pairs':[{k:v for k,v in p.items() if k not in ('primary',)}
                                   for p in g['_terrain_pairs']]})
        coverage_rows.append({'state_group_id':g['state_group_id'],
                              'compatible':g['compatible_coverage_summary']})
        variable_rows.append({'state_group_id':g['state_group_id'],
                              **variable_group_diagnostic(g)})
    write(output_root/'attainable-mode-counts.json',{
        'groups':mode_rows,'shared_bank_by_bearing':bank,
        'exact_finite_library_optimal_all_groups':optimal,
        'hard_valid_at_least_five_groups':sum(g['hard_valid_count']>=5 for g in groups),
        'source_compatible_at_least_five_groups':sum(g['source_compatible_count']>=5 for g in groups),
        'symmetric_five_groups':sum(g['symmetric_targets']['5']['feasible'] for g in groups),
        'fixed_five_geometry_ready':geometry_ready,
        'mode_rule':PRIOR_CONFIG['mode_separation'],
        'finite_library_only':True})
    write(output_root/'candidate-audit-075m.json',{
        'scope':'definition-level records; not independent experimental samples',
        'groups':40,'records':candidate_audits})
    write(output_root/'terrain-separation-primary-060m.json',{
        'interval_arc_m':[0,.6],'teacher_proxy_not_observed_travel':True,
        'groups':primary_rows})
    write(output_root/'terrain-separation-full-075m.json',{
        'interval_arc_m':[0,.75],'diagnostic_only':True,'groups':full_rows})
    write(output_root/'dem-footprint-coverage.json',{
        'P4_outside_allowed_but_retained':True,'unknown_never_counts_as_terrain_evidence':True,
        'groups':coverage_rows})
    write(output_root/'variable-count-diagnostic.json',{
        'diagnostic_only':True,'cannot_override_fixed_five':True,
        'available_groups':sum(row['available'] for row in variable_rows),
        'groups':variable_rows,
        'integration_note':'A validity mask could support 3–5 real candidates in a later model/loss revision; no training contract changed.'})
    old_primary=recompute_baseline_primary_terrain(frozen)
    write(output_root/'baseline-060-primary-terrain-recomputed.json',old_primary)
    baseline=paired_baseline(
        frozen['prior']['curvature060']['candidate-attainable-counts.json']['groups'],
        frozen['prior']['curvature060']['curve-library-records.json'],
        frozen['prior']['curvature060']['compatible-distinct-pairs.json'],groups,old_primary)
    write(output_root/'baseline-060-vs-075-comparison.json',baseline)
    library_pairs={g['state_group_id']:g['_terrain_pairs'] for g in groups}
    library_support=terrain_support_gate(groups,library_pairs)
    if geometry_ready:
        selected_pairs={g['state_group_id']:[p for p in g['_terrain_pairs']
                        if p['first'] in bank[str(g['bearing_deg'])]['selected_ids'] and
                           p['second'] in bank[str(g['bearing_deg'])]['selected_ids']]
                        for g in groups}
        selected_support=terrain_support_gate(groups,selected_pairs)
        selected_coverage=[g['compatible_coverage_summary'][candidate]
                           for g in groups for candidate in bank[str(g['bearing_deg'])]['selected_ids']]
        coverage_ready=all(row['primary_centerline_known_fraction']>0 and
                           row['full_centerline_known_fraction']>0 and
                           row['proxies']['main_track_sprocket']['primary']['known_fraction']>0
                           for row in selected_coverage)
    else:
        selected_support={'ready':False,'reason':'no fixed five bank to audit','support':None}
        coverage_ready=False
    terrain_ready=bool(selected_support['ready'])
    write(output_root/'terrain-support-gate.json',{
        'library_diagnostic':library_support,'selected_fixed_five_bank':selected_support,
        'primary_only':True,'late_only_not_counted':True})
    write(output_root/'coverage-gate.json',{
        'ready':coverage_ready if geometry_ready else None,
        'reason':None if geometry_ready else 'no fixed five bank to audit',
        'P4_full_inside_not_required':True})
    decision=decide_075(True,geometry_ready,terrain_ready,coverage_ready)
    if geometry_ready and terrain_ready and coverage_ready:
        write(output_root/'five-candidate-bank.proposal.json',{
            'candidate_ids_by_bearing':{b:row['selected_ids'] for b,row in bank.items()},
            'source_anchor_ids':[a['anchor_id'] for a in frozen['anchors']],
            'status':'offline proposal only; execution not authorized'})
    p1=conditional_p1_proposal(geometry_ready and terrain_ready and coverage_ready)
    if p1['prepared']:
        selected=[]
        for cls in ('transition','differential','graded','flat-only'):
            options=[a for a in frozen['anchors'] if a['terrain_class']==cls]
            if not options:raise ValueError('missing physical P1 anchor class '+cls)
            chosen=next((a for a in options if a['seed_label']=='first'),options[0])
            selected.append({'anchor_id':chosen['anchor_id'],'seed_label':chosen['seed_label'],
                             'source_family':chosen['source_family'],'terrain_class':cls,
                             'selection_reason':'first frozen physically supplied anchor of class, no Outcome'})
        if len({row['seed_label'] for row in selected})==1:
            alternate=next((a for a in frozen['anchors'] if a['terrain_class']=='graded' and
                            a['seed_label']!='first'),None)
            if alternate:
                selected[2]={**selected[2],
                             'anchor_id':alternate['anchor_id'],'seed_label':alternate['seed_label'],
                             'source_family':alternate['source_family'],
                             'selection_reason':'second seed graded anchor for source diversity, no Outcome'}
        p1['selected_anchors']=selected
        p1['both_seeds_represented']=len({row['seed_label'] for row in selected})==2
        p1['candidate_ids_bearing_zero']=bank['0']['selected_ids']
        p1['frozen_e1_numeric_gates']=_prospective_P1_P2_gates(Path(__file__).resolve().parents[1])
    write(output_root/'conditional-p1-proposal.json',p1)
    forecast=_forecast(frozen['prior']['goal060']['runtime-storage-forecast.json'])
    forecast['gpu_peak_plus_20_percent_mib']=1.2*forecast['gpu_peak_reference_mib']
    forecast['simulator_pss_plus_20_percent_gib']=1.2*forecast['simulator_pss_reference_gib']
    forecast['anon_shmem_plus_20_percent_gib']=1.2*forecast['anon_shmem_reference_gib']
    write(output_root/'runtime-resource-forecast.json',{
        'conditional_only':True,'prior_distribution_source':frozen['provenance']['archives']['goal060'],
        'prior_numeric_reference':frozen['prior']['goal060']['runtime-storage-forecast.json'],
        'P1_28_and_P2_56_with_20_percent_margin':forecast,
        'new_075m_live_runtime_unmeasured':True})
    figures=_figure_set(output_root,groups,libraries,baseline,bank,coverage_rows,primary_rows)
    if len(figures)!=10:raise ValueError('ten diagnostic figures required')
    result={'decision':decision,'state_groups':40,
            'fixed_five_geometry_gate':geometry_ready,
            'primary_terrain_support_gate':terrain_ready,
            'dem_footprint_coverage_gate':coverage_ready if geometry_ready else None,
            'variable_count_available_groups':sum(row['available'] for row in variable_rows),
            'P1_proposal_prepared':p1['prepared'],'P1_executed':False,
            'simulator_follower_bridge_router_started':False,
            'Outcome_collected':False,'training_started':False,
            'source_parity_valid':True,'frozen_contract_sha256':freeze['contract_sha256'],
            'figure_count':len(figures)}
    write(output_root/'final-decision.json',result)
    bearing_summary={str(b):{'compatible_per_group':[g['source_compatible_count'] for g in groups if g['bearing_deg']==b],
                             'maximum_distinct_per_group':[g['maximum_distinct']['maximum_mode_count']
                                                            for g in groups if g['bearing_deg']==b],
                             'symmetric_three_groups':sum(g['bearing_deg']==b and g['symmetric_targets']['3']['feasible'] for g in groups),
                             'symmetric_five_groups':sum(g['bearing_deg']==b and g['symmetric_targets']['5']['feasible'] for g in groups)}
                     for b in (-20,-10,0,10,20)}
    report=f'''# 0.75 m common-goal candidate bank: offline review

Decision: `{decision}`. This is a finite-library geometry and DEM review. It does not authorize P1.

The 0.75 m planning endpoint corresponds to 2.5 s at nominal 0.3 m/s. The Outcome teacher remains the first 2.0 s (40 × 0.05 s); 0–0.60 m is only an offline interaction proxy, not observed travel.

Eight SHA-verified physical anchors, 40 StateGroups, both seeds and five bearings were audited. The three production C0 fixtures and all 560 previous 0.60 m selected G0 Path SHA values matched. Follower source/config and frozen E1 gate SHA were checked. Source worktree HEAD: `{frozen['provenance']['current_worktree_head']}`. The source worktree status before generation is in `input-freeze.json`.

The fixed-five geometry gate is **{geometry_ready}**: {sum(g['hard_valid_count']>=5 for g in groups)}/40 groups have five hard-valid candidates, {sum(g['source_compatible_count']>=5 for g in groups)}/40 have five follower-compatible candidates, {sum(g['symmetric_targets']['5']['feasible'] for g in groups)}/40 admit direct plus two left/two right locally, and {sum(row['feasible'] for row in bank.values())}/5 bearings admit a shared five-identity bank across all eight anchors. Exact MILP optimal status: {optimal}. All four DEM footprint proxies and known/unknown/outside fractions are retained. The conservative P4 extent is diagnostic, and unknown cells never supply terrain evidence.

Per-bearing compatible supply, exact distinct mode counts and balanced 3/5 feasibility are in `attainable-mode-counts.json`; compact summary: `{bearing_summary}`. The frozen 0.60 m G0A baseline exempted non-forward initial tangents, while the revised hard-valid screen does not. The baseline table retains its original result and separately reports same-screen compatible counts. The 0.60 m primary terrain metric was recomputed on exactly 0–0.60 m using SHA-frozen curves and DEM; archived full-curve terrain values remain diagnostic.

Selected-bank primary terrain gate is **{terrain_ready}**; compatible-library support by class and seed is `{library_support['support']}`. Terrain separation appearing only after arc 0.60 m is excluded from the primary gate. Selected-bank coverage gate is `{coverage_ready if geometry_ready else 'not evaluable without a bank'}`. Variable 3–5 real-candidate diagnostic is available in {sum(row['available'] for row in variable_rows)}/40 groups, with masks and no padding; it does not change the fixed-five decision or model input contract.

The paired 0.60/0.75 comparison uses StateGroup as the unit; curve and candidate-pair definitions are not independent statistical samples. See `baseline-060-vs-075-comparison.json` for seed, source family, bearing and class strata. The 1.236 m^-1 value is an unchanged offline production-follower compatibility screen, not a physical mobility bound or tracking proof. Only the first command is statically estimated; no follower, router, bridge, simulator, Outcome or training ran. P1/P2 are unexecuted and conditional.

The supplemental deterministic G1/G2/G3 search covers length hypotheses through 0.92 m and preserves every failed construction with family, parameters and first failed constraint. It does not prove infeasibility over the continuous curve space. Conditional fresh-process reference forecasts with 20% margin are in `runtime-resource-forecast.json`; no P1 budget is activated by this failed geometry gate.

Large candidate definitions are kept in this output root and SHA listed in `artifact-sha-ledger.json`; compact review files and ten figures are committed separately.
'''
    (output_root/'review-report.md').write_text(report)
    files={p.relative_to(output_root).as_posix():{'sha256':sha(p),'size_bytes':p.stat().st_size}
           for p in output_root.rglob('*') if p.is_file() and p.name!='artifact-sha-ledger.json'}
    write(output_root/'artifact-sha-ledger.json',{'files':files})
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-artifact-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run_review(args.frozen_artifact_root,args.output_root),sort_keys=True))


if __name__=='__main__':main()
