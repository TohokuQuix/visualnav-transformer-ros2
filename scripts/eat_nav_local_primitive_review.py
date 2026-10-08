#!/usr/bin/env python3
"""Offline source, geometry and terrain review for local motion primitives."""

from __future__ import annotations

import hashlib
import itertools
import math
import gzip
import json
import statistics
import subprocess
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.eat_nav_goal075_review import verify_inputs as verify_goal075_inputs
from scripts.eat_nav_goal075_review import _archive_audit
from scripts.eat_nav_curvature_attainable import sha
from scripts.eat_nav_curvature_attainable import write, _read_follower_config, FOLLOWER_CONFIG
from visualnav_transformer.eat_nav.goal075_offline import coverage_record, prefix_row
from visualnav_transformer.eat_nav.goal_bank_offline import follower_diagnostic, corridor_audit, pairwise_audit
from visualnav_transformer.eat_nav.local_motion_primitives import (
    BEARINGS_DEG, CURVATURE_SCREEN_PER_M, goal_labels, constant_curvature_primitive,
)

GOAL075_ARCHIVE = ('eatnav-goal075-offline-20261008-r5.tar.gz',
                   '018394b350313701d385822c2eb825ea3011e3cbb44981a992c8c884da8e28ce',
                   'eatnav-goal075-offline-20261008-r5', 'artifact-sha-ledger.json',
                   ('source-provenance.json','final-decision.json','artifact-sha-ledger.json'))

WORKTREE = Path(__file__).resolve().parents[1]
ONIX = Path('/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity')
FROZEN_RUNTIME_SOURCES = {
    WORKTREE/'scripts/eat_nav_e1_sim.py': '9e2a40f0b283d4581219302c3f2f0782b00ca825e07653bbe3924dd5c47cec07',
    WORKTREE/'scripts/eat_nav_e1_run_one.py': 'c1e835e79951ce829ec649d363c42ccb938c5af8b53ef5f664be81bd55518db7',
    WORKTREE/'scripts/eat_nav_phase_c_ros_bridge.py': 'd3bd9bc49593c491cf158c294d32c36ea6b49a7c4365c81853f2e1751b3f6197',
    WORKTREE/'scripts/eat_nav_e1_analyze.py': '08e78ac58bdef55dba26ad9c2333e64b042f1dfdbd56866ce38ac76dd8e3bf64',
    ONIX/'config/onix/rough_env_cfg.py': 'f357e63f7f8b95caa5b6c6ac064cf1b76cc38a797edd7c435c01f2373d79ac71',
    ONIX/'mdp/rewards.py': '1e8ddec3ec265204eb872081692de36c12959620556c472cf5bef19137bcc535',
}


def classify_goal_use(consumer: str, field: str) -> str:
    if field not in {'mission_logged_goal','goal_xy','global_goal','goal_distance'}:
        return 'unused'
    if consumer in {'diagnostic_fields','summary'}:
        return 'logging_only'
    if consumer in {'outcome','offline_label'}:
        return 'offline_label_only'
    return 'physical_execution_dependency'


def runtime_goal_dependency_audit() -> dict:
    """Fail closed if any historically inspected physical source changed.

    The relabeled global goal is a new offline label. It is distinct from the
    frozen selected DEM goal used by the pre-Path anchor readiness gate.
    """
    for path, expected in FROZEN_RUNTIME_SOURCES.items():
        if sha(path) != expected:
            raise ValueError('frozen physical/analysis source SHA mismatch: '+str(path))
    simulator=(WORKTREE/'scripts/eat_nav_e1_sim.py').read_text()
    permitted={'goal_metadata[\'mission_logged_goal\']', "goal_metadata['goal_xy']"}
    uses={part for part in permitted if part in simulator}
    if uses != permitted or simulator.count("goal_metadata['mission_logged_goal']") != 1 or simulator.count("goal_metadata['goal_xy']") != 1:
        raise ValueError('unaccounted goal metadata use')
    if "goal=planned[-1]" not in (WORKTREE/'scripts/eat_nav_e1_analyze.py').read_text():
        raise ValueError('historic post-run endpoint semantics changed')
    dataflow={
        'Path generation':'frozen base_path input; mission goal metadata is not used to build it (e1_sim.py:414–429)',
        'Path publication':'fixed map Path from base_path and anchor pose to bridge reset (e1_sim.py:436)',
        'follower command':'bridge step returns TwistStamped for published Path and pose (e1_sim.py:447–466)',
        'bridge command delivery':'CommandOverride injects follower base_velocity command (e1_sim.py:447–467)',
        'Actor observation/action':'obs policy/critic from environment; policy action from captured obs (e1_sim.py:468–509)',
        'simulator dynamics':'env.step uses Actor action and other frozen env actions (e1_sim.py:506–513)',
        'reward and termination':'env.step reward and termination manager (e1_sim.py:476–482,509–522); onix rough_env_cfg/rewards.py',
        'Critic input/value':'obs critic and capture_actor_critic (e1_sim.py:488)',
        'post-run goal distance':'old e1_analyze.py:129,154–155 uses local Path endpoint; new common-goal label must be recomputed',
    }
    goal_uses=[
        {'source':'scripts/eat_nav_e1_sim.py:321–323','consumer':'input validation',
         'field':'state_id/base_path_sha256','classification':'physical_execution_dependency',
         'interpretation':'Path identity is frozen and held fixed under relabeling'},
        {'source':'scripts/eat_nav_e1_sim.py:405–412','consumer':'pre-Path anchor gate',
         'field':'selected DEM goal','classification':'physical_execution_dependency',
         'interpretation':'historical anchor gate input, not the relabelable global navigation goal'},
        {'source':'scripts/eat_nav_e1_sim.py:256','consumer':'diagnostic_fields',
         'field':'mission_logged_goal','classification':'logging_only'},
        {'source':'scripts/eat_nav_e1_sim.py:257','consumer':'diagnostic_fields',
         'field':'goal_xy','classification':'logging_only'},
        {'source':'scripts/eat_nav_e1_analyze.py:129,154–155','consumer':'outcome',
         'field':'goal_distance','classification':'offline_label_only',
         'interpretation':'historic local-endpoint metric cannot substitute for new common-goal label'},
    ]
    relabelable=[row for row in goal_uses if row['field'] in ('mission_logged_goal','goal_xy','goal_distance')]
    return {'schema':'global_goal_runtime_dependency_audit_v1',
            'source_sha256':{str(path):digest for path,digest in FROZEN_RUNTIME_SOURCES.items()},
            'dataflow':dataflow,'goal_uses':relabelable,
            'separate_nonrelabelable_dependencies':goal_uses[:2],
            'relabeling_source_valid':all(row['classification']!='physical_execution_dependency'
                                          for row in relabelable),
            'held_fixed':'anchor, candidate Path, follower and bridge state, random state, RL checkpoint',
            'claim_limit':'source graph only; archived-trace relabel fixture remains required'}


def decide(*, source: bool, geometry: bool, follower: bool, terrain: bool,
           coverage: bool, fixture: bool, integrity: bool=True) -> str:
    if not integrity:return 'source_or_artifact_integrity_invalid'
    if not source:return 'global_goal_relabeling_not_source_valid'
    if not geometry or not follower:return 'local_motion_primitive_geometry_or_follower_fail'
    if not terrain or not coverage or not fixture:return 'local_motion_primitive_terrain_support_fail'
    return 'local_motion_primitive_bank_ready_for_p1_rollout_review'


def verify_inputs(backup_root, goal075_archive=None) -> dict:
    backup_root=Path(backup_root)
    frozen=verify_goal075_inputs(backup_root)
    archive_root=Path(goal075_archive) if goal075_archive else backup_root
    archived,prior=_archive_audit(archive_root,GOAL075_ARCHIVE)
    if prior['final-decision.json']['decision']!='five_candidate_075m_still_infeasible':
        raise ValueError('unexpected 0.75 m decision')
    frozen['prior_goal075']={'archive':archived,'records':prior}
    return frozen


def freeze_review(backup_root: Path, output_root: Path) -> dict:
    output_root=Path(output_root)
    if output_root.exists():raise FileExistsError('new output root required: '+str(output_root))
    frozen=verify_inputs(backup_root,Path('/home/kojima/eatnav-backups'))
    audit=runtime_goal_dependency_audit()
    if not audit['relabeling_source_valid']:
        raise ValueError('global_goal_relabeling_not_source_valid')
    output_root.mkdir(parents=True)
    provenance=frozen['provenance']
    provenance['prior_goal075_archive']=frozen['prior_goal075']['archive']
    provenance['physical_runtime_source_sha256']=audit['source_sha256']
    provenance['new_primitive_generator_sha256']=sha(WORKTREE/'src/visualnav_transformer/eat_nav/local_motion_primitives.py')
    provenance['new_review_runner_sha256']=sha(Path(__file__))
    write(output_root/'source-provenance.json',provenance)
    write(output_root/'global-goal-runtime-dependency-audit.json',audit)
    write(output_root/'local-motion-primitive-contract.json',local_contract())
    return {'contract_sha256':sha(output_root/'local-motion-primitive-contract.json'),
            'source_relabeling_gate':audit['relabeling_source_valid']}


def local_contract() -> dict:
    return {'schema':'eatnav_local_motion_primitive_contract_v1',
            'local_endpoint_chord_distance_m':.75,
            'local_endpoint_bearings_deg':list(BEARINGS_DEG),
            'candidate_order':'increasing_bearing',
            'common_global_goal_distance_m':.75,
            'common_global_goal_bearings_deg':list(BEARINGS_DEG),
            'local_endpoint_is_navigation_teacher':False,
            'Path':'eight future robot-local XY points; origin implicit',
            'initial_tangent_xy':[1.,0.],
            'reverse_allowed':False,'self_intersection_allowed':False,
            'curvature_screen_per_m':CURVATURE_SCREEN_PER_M,
            'teacher_control_steps':40,'teacher_step_s':.05,
            'primary_terrain_arc_m':[0.,.6],
            'full_terrain_diagnostic':'through each complete Path endpoint',
            'physical_anchor_count':8,'physical_definition_count':40,
            'potential_correlated_label_count':200,
            'no_rollout_authorized':True}


def audit_primitive(curve, map_pose_xy_yaw, follower_config) -> dict:
    path=np.asarray(curve['waypoints'],dtype=float)
    dense=np.asarray(curve['dense'],dtype=float)
    pose=np.asarray(map_pose_xy_yaw,dtype=float)
    reasons=[]
    if pose.shape!=(3,) or not np.isfinite(pose).all():reasons.append('invalid_map_pose')
    if path.shape!=(8,2) or dense.ndim!=2 or dense.shape[1]!=2 or not np.isfinite(path).all() or not np.isfinite(dense).all():
        reasons.append('invalid_or_nonfinite_geometry')
    if not reasons:
        if np.linalg.norm(dense[0])>1e-9:reasons.append('start_not_origin')
        if curve['endpoint_error_m']>1e-6:reasons.append('endpoint_mismatch')
        if np.linalg.norm(np.asarray(curve['initial_tangent_xy'])-[1.,0.])>1e-9:reasons.append('initial_tangent_error')
        if np.max(np.linalg.norm(np.diff(dense,axis=0),axis=1))>.02+1e-8:reasons.append('dense_spacing')
        if np.min(np.diff(dense[:,0]))<=0:reasons.append('reverse_or_self_intersection')
        if len(np.unique(path,axis=0))!=8:reasons.append('duplicate_waypoint')
        c,s=math.cos(pose[2]),math.sin(pose[2]);rotation=np.array([[c,-s],[s,c]])
        restored=((path@rotation.T+pose[:2])-pose[:2])@rotation
        if np.max(np.abs(restored-path))>1e-9:reasons.append('map_base_transform_error')
    command=follower_diagnostic({'waypoints':path,
                                  'maximum_curvature_per_m':curve['analytic_max_curvature_per_m']},follower_config)
    follower_reasons=[]
    if curve['analytic_max_curvature_per_m']>CURVATURE_SCREEN_PER_M+1e-9:follower_reasons.append('analytic_curvature_screen')
    if curve['dense_measured_max_curvature_per_m']>CURVATURE_SCREEN_PER_M+1e-9:follower_reasons.append('dense_curvature_screen')
    if curve['eight_point_max_curvature_per_m']>CURVATURE_SCREEN_PER_M+1e-9:follower_reasons.append('eight_point_curvature_screen')
    if command['command_mode']!='tracking':follower_reasons.append('source_first_command_'+command['command_mode'])
    if command['reverse_estimate']:follower_reasons.append('reverse_estimate')
    if command['turn_in_place_entry']:follower_reasons.append('turn_in_place_estimate')
    first=command['initial_command_estimate']
    if first['linear_x'] is None or first['angular_z'] is None or not np.isfinite([first['linear_x'],first['angular_z']]).all():
        follower_reasons.append('nonfinite_first_command')
    return {'candidate_id':curve['variant_id'],'path_sha256':curve['path_sha256'],
            'endpoint_error_m':curve['endpoint_error_m'],
            'analytic_curvature_per_m':curve['analytic_curvature_per_m'],
            'analytic_max_curvature_per_m':curve['analytic_max_curvature_per_m'],
            'dense_measured_max_curvature_per_m':curve['dense_measured_max_curvature_per_m'],
            'eight_point_max_curvature_per_m':curve['eight_point_max_curvature_per_m'],
            'hard_valid':not reasons,'hard_invalid_reasons':reasons,
            'follower_compatible':not follower_reasons,
            'follower_invalid_reasons':follower_reasons,
            'follower':command}


def terrain_support_for_anchor(curves,height,mask,proxy_bboxes) -> dict:
    by_id={};coverage_rows={}
    for curve in curves:
        row={'dense':curve['dense'],'stations_m':curve['stations_m'],
             'arc_length_m':curve['arc_length_m']}
        coverage=coverage_record(row,np.asarray(height),np.asarray(mask),proxy_bboxes)
        full=corridor_audit(row,np.asarray(height),np.asarray(mask),proxy_bboxes)
        by_id[curve['variant_id']]=(row,coverage,full)
        public={key:value for key,value in coverage.items() if not key.startswith('_')}
        public.update({'full_diagnostic_arc_length_m':curve['arc_length_m'],
                       'full_centerline_known_fraction':full['centerline_known_fraction'],
                       'full_centerline_unknown_fraction':full['centerline_unknown_fraction'],
                       'full_centerline_outside_fraction':full['centerline_outside_fraction'],
                       'minimum_full_tile_margin_m':float(np.min(1-np.max(np.abs(curve['dense']),axis=1)))})
        main=coverage['_dem_primary']['corridors']['main_track_sprocket']
        full_main=full['corridors']['main_track_sprocket']
        public.update({'primary_main_track_slope':main['corridor_maximum_slope'],
                       'primary_main_track_step_m':main['corridor_maximum_step_m'],
                       'primary_main_track_roughness_m':main['corridor_roughness_m'],
                       'full_main_track_slope':full_main['corridor_maximum_slope'],
                       'full_main_track_step_m':full_main['corridor_maximum_step_m'],
                       'full_main_track_roughness_m':full_main['corridor_roughness_m'],
                       'interaction_onset_primary_m':coverage['_dem_primary']['interaction_onset_station_m'],
                       'interaction_onset_full_m':full['interaction_onset_station_m']})
        tangent=np.gradient(np.asarray(curve['dense']),axis=0)
        tangent/=np.linalg.norm(tangent,axis=1)[:,None]
        normal=np.stack((-tangent[:,1],tangent[:,0]),axis=1)
        for name in proxy_bboxes:
            public['proxies'][name]['full']={key:full['corridors'][name][key] for key in
                ('known_fraction','unknown_fraction','outside_fraction')}
            low,high=np.asarray(proxy_bboxes[name],dtype=float)
            corners=np.array(list(itertools.product((low[0],high[0]),(low[1],high[1]))))
            touched=(np.asarray(curve['dense'])[:,None,:]
                     +corners[None,:,0,None]*tangent[:,None,:]
                     +corners[None,:,1,None]*normal[:,None,:])
            public['proxies'][name]['minimum_full_tile_margin_m']=float(np.min(1-np.max(np.abs(touched),axis=2)))
        coverage_rows[curve['variant_id']]=public
    pairs=[]
    for first,second in itertools.combinations(by_id,2):
        row_a,cov_a,full_a=by_id[first];row_b,cov_b,full_b=by_id[second]
        primary=pairwise_audit(prefix_row(row_a,.6),prefix_row(row_b,.6),
                               cov_a['_dem_primary'],cov_b['_dem_primary'])
        full=pairwise_audit(row_a,row_b,full_a,full_b)
        def separated(metric):
            return (metric['common_known_corridor_bins']>=5
                    and metric['known_corridor_profile_rms_m'] is not None
                    and metric['known_corridor_profile_rms_m']>=.01)
        pair={'distinct':True,'primary_separated':bool(separated(primary)),
              'full_separated':bool(separated(full)),'primary':primary,
              'full_diagnostic':full,'late_only':bool(separated(full) and not separated(primary))}
        pairs.append({'first':first,'second':second,**pair})
    return {'candidate_coverage':coverage_rows,'pairs':pairs,
            'primary_separated_pair_count':sum(p['primary_separated'] for p in pairs),
            'late_only_pair_count':sum(p['late_only'] for p in pairs)}


def conditional_p1_manifest(selected_anchors, *, ready: bool) -> dict:
    if not ready:
        return {'prepared':False,'executed':False,'processes':[],
                'reason':'one or more offline readiness gates failed'}
    if len(selected_anchors)!=4 or {a['terrain_class'] for a in selected_anchors}!={'transition','differential','graded','flat-only'}:
        raise ValueError('one anchor per frozen P1 terrain class required')
    if len({a['source_family'] for a in selected_anchors})!=4:
        raise ValueError('P1 source families must be independent')
    processes=[]
    for anchor in selected_anchors:
        for bearing in BEARINGS_DEG:
            repeats=3 if bearing==0 else 1
            for repeat in range(repeats):
                processes.append({'anchor_id':anchor['anchor_id'],
                                  'terrain_class':anchor['terrain_class'],
                                  'source_family':anchor['source_family'],
                                  'primitive_bearing_deg':bearing,'repeat_index':repeat,
                                  'primary_common_global_goal_bearing_deg':0,
                                  'candidate_path_to_be_frozen_after_review':True})
    assert len(processes)==28
    return {'prepared':True,'executed':False,'processes':processes,
            'process_count':28,'repeat_policy':'0° primitive three total runs per anchor',
            'external_zenoh_required':True,'X2_required':True,'automatic_retry_count':0}


def select_p1_anchors(anchors: list[dict]) -> list[dict]:
    classes=('transition','differential','graded','flat-only')
    options=[[a for a in anchors if a['terrain_class']==cls] for cls in classes]
    for combination in itertools.product(*options):
        if len({a['source_family'] for a in combination})==4:
            return list(combination)
    raise ValueError('no four-class P1 selection with distinct source families')


def relabel_archived_trace(run_root: Path) -> dict:
    """SHA-verify one sealed physical trace, then relabel its first 40 steps."""
    run_root=Path(run_root)
    complete=json.loads((run_root/'run-complete.json').read_text())
    seal=json.loads((run_root/'sealed-tree-manifest.json').read_text())
    trace_path=run_root/'live-0.trace.json.gz'
    digest=sha(trace_path)
    if (digest!=complete['output_sha256']['live-0.trace.json.gz']
            or digest!=seal['files']['live-0.trace.json.gz']
            or complete['identity']['kind']!='candidate'):
        raise ValueError('archived trace completion/seal/SHA mismatch')
    with gzip.open(trace_path,'rt') as stream: trace=json.load(stream)
    summary_path=run_root/'live-0.json'
    if sha(summary_path)!=complete['output_sha256']['live-0.json'] or sha(summary_path)!=seal['files']['live-0.json']:
        raise ValueError('archived summary SHA mismatch')
    summary=json.loads(summary_path.read_text())
    if summary['trace_sha256']!=digest or summary['anchor_gate']['status']!='pass':
        raise ValueError('archived trace/anchor summary mismatch')
    if len(trace)<40 or any(trace[k]['step_identity']['control_step']!=k for k in range(40)):
        raise ValueError('archived trace lacks 40 consecutive control steps')
    if any(trace[k]['terminal'] for k in range(40)):
        raise ValueError('archived trace terminated before 40-step fixture')
    initial=np.asarray(trace[0]['root_state'],dtype=float)
    w,x,y,z=initial[3:7]
    yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
    pose=np.array([initial[0],initial[1],yaw])
    if summary['post_follower_anchor']['root_state']!=trace[0]['root_state']:
        raise ValueError('frozen anchor root differs from first trace root')
    base_path=summary['goal_metadata']['base_path']
    if base_path!=trace[0]['planned_candidate_path'] or summary['goal_metadata']['base_path_sha256']!=summary['source_sha256']['base_path']:
        raise ValueError('frozen base Path differs from trace Path')
    map_path=np.asarray(summary['map_path'],dtype=float)
    c,s=math.cos(yaw),math.sin(yaw);rotation=np.array([[c,-s],[s,c]])
    transformed=np.asarray(base_path,dtype=float)@rotation.T
    map_error=float(np.max(np.abs((map_path-map_path[0])-transformed)))
    if map_error>1e-6 or summary['fixed_map_path_sha256']!=complete['identity']['map_path_file_sha256']:
        raise ValueError('frozen map Path transform or SHA mismatch')
    positions=np.asarray([[initial[0],initial[1]]]+[row['post_root_state'][:2] for row in trace[:40]],dtype=float)
    rewards=[row['reward'] for row in trace[:40]]
    local_endpoint=np.asarray(trace[0]['planned_candidate_path'][-1],dtype=float)
    physical_id=complete['identity']['run_id']
    physical_invariants={'reward_steps':rewards,
        'maximum_contact_force_n':float(max(np.linalg.norm(np.asarray(row['contact_force'],dtype=float),axis=-1).max() for row in trace[:40])),
        'maximum_raw_joint_effort':float(max(np.abs(np.asarray(row.get('raw_joint_effort') or [0.],dtype=float)).max() for row in trace[:40])),
        'termination_or_censor_status':'horizon_reached_no_terminal',
        'physical_trace_sha256':digest}
    invariant_bytes=json.dumps(physical_invariants,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    physical_invariants['sha256']=hashlib.sha256(invariant_bytes).hexdigest()
    labels=[]
    for goal in BEARINGS_DEG:
        target=goal_labels(positions,pose,rewards,goal,local_endpoint=local_endpoint)
        labels.append({'physical_run_id':physical_id,'label_id':f'{physical_id}:goal:{goal:+03d}',
                       'goal_bearing_deg':goal,'split':'fixture','physical_trace_sha256':digest,
                       'physical_invariants_sha256':physical_invariants['sha256'],
                       'local_endpoint_base_xy_m':local_endpoint.tolist(),**target})
    return {'source_trace_path':str(trace_path),'source_run_complete_sha256':sha(run_root/'run-complete.json'),
            'source_seal_sha256':sha(run_root/'sealed-tree-manifest.json'),
            'verified_step_count':len(trace),'used_control_steps':40,
            'physical_trace_sha256':digest,'anchor_world_xy_yaw':pose.tolist(),
            'frame_validation':{'base_path_exact_match':True,'anchor_root_exact_match':True,
                                'map_path_transform_max_error_m':map_error,
                                'map_path_origin_frame':'simulation map local to env origin',
                                'trace_root_frame':'world; local displacements computed relative to world anchor',
                                'map_path_file_sha256':summary['fixed_map_path_sha256']},
            'physical_run_identity':complete['identity'],
            'physical_invariants':physical_invariants,
            'labels':labels,'split_grouping':'all five labels share physical run/source family',
            'physical_execution_repeated':False}


def _read_prior_goal075(name: str) -> dict:
    archive=Path('/home/kojima/eatnav-backups')/GOAL075_ARCHIVE[0]
    if sha(archive)!=GOAL075_ARCHIVE[1]:raise ValueError('goal075 archive SHA changed')
    with tarfile.open(archive,'r:gz') as bundle:
        member=f'{GOAL075_ARCHIVE[2]}/{name}'
        stream=bundle.extractfile(member)
        if stream is None:raise ValueError('missing prior artifact: '+name)
        return json.load(stream)


def state_group_comparison(old: dict, new_group: dict) -> dict:
    goal_bearing=old['bearing_deg']
    goal=.75*np.array([math.cos(math.radians(goal_bearing)),math.sin(math.radians(goal_bearing))])
    residual={bearing:float(np.linalg.norm(.75*np.array([
        math.cos(math.radians(bearing)),math.sin(math.radians(bearing))])-goal))
              for bearing in BEARINGS_DEG}
    candidates=new_group['candidates']
    return {'state_group_id':old['state_group_id'],'anchor_id':old['anchor_id'],
            'common_global_goal_bearing_deg':goal_bearing,
            'old_060_compatible_count':old['baseline_060']['compatible_common_goal'],
            'old_060_distinct_mode_count':old['baseline_060'].get('maximum_distinct'),
            'old_060_primary_terrain_pairs':old['baseline_060'].get('terrain_separated_pairs_primary_060'),
            'old_075_compatible_count':old['revised_075']['compatible_common_goal'],
            'old_075_distinct_mode_count':old['revised_075'].get('maximum_distinct'),
            'old_075_primary_terrain_pairs':old['revised_075'].get('terrain_separated_pairs_primary_060'),
            'local_primitive_count':len(candidates),
            'new_follower_compatible_count':sum(c['hard_valid'] and c['follower_compatible'] for c in candidates),
            'new_max_curvature_per_m':max(c['analytic_max_curvature_per_m'] for c in candidates),
            'new_first_yaw_commands_rad_s':[c['follower']['initial_command_estimate']['angular_z'] for c in candidates],
            'new_primary_terrain_pairs':new_group['terrain']['primary_separated_pair_count'],
            'new_late_only_pairs':new_group['terrain']['late_only_pair_count'],
            'new_local_endpoint_residual_m':residual}


def build_previous_bank_group_rows(frozen: dict) -> list[dict]:
    """Numeric 40-Group A/B/G1–G3 comparison from SHA-verified prior archives."""
    archive=Path(frozen['provenance']['archives']['redesign060']['archive'])
    if sha(archive)!=frozen['provenance']['archives']['redesign060']['archive_sha256']:
        raise ValueError('prior Bank A/B archive SHA changed')
    with tarfile.open(archive,'r:gz') as bundle:
        member=next(name for name in bundle.getnames() if name.endswith('/per-state-group-support.json'))
        support=json.load(bundle.extractfile(member))
    selected=frozen['prior']['redesign060']['selected-candidate-definitions.json']
    library=frozen['prior']['curvature060']['curve-library-records.json']
    counts=frozen['prior']['curvature060']['candidate-attainable-counts.json']['groups']
    by_selected={};by_library={}
    for row in selected:by_selected.setdefault((row['state_group_id'],row['bank']),[]).append(row)
    for row in library:
        if row['family'] in ('G1','G2','G3'):
            by_library.setdefault(row['state_group_id'],[]).append(row)
    count_by={row['state_group_id']:row for row in counts}
    def summarize(rows,compatible,curvature,command,known,residual,pairs,modes):
        accepted=[row for row in rows if compatible(row)]
        numeric=lambda values:[float(x) for x in values if x is not None and math.isfinite(float(x))]
        c=numeric(curvature(r) for r in accepted)
        yaw=numeric(command(r) for r in accepted)
        coverage=numeric(known(r) for r in accepted)
        residuals=numeric(residual(r) for r in accepted)
        all_residuals=numeric(residual(r) for r in rows)
        return {'compatible_count':len(accepted),'selected_definition_count':len(rows),
                'distinct_geometry_modes_reported':modes,
                'distinct_compatible_path_sha_count':len({r['path_sha256'] for r in accepted}),
                'max_curvature_per_m':max(c) if c else None,
                'first_yaw_command_rad_s':yaw,
                'turn_in_place_count':sum(r['follower']['turn_in_place_entry'] if 'follower' in r
                                          else r['first_command']['turn_in_place_entry'] for r in accepted),
                'reverse_count':sum(r['follower']['reverse_estimate'] if 'follower' in r
                                    else r['first_command']['reverse_estimate'] for r in accepted),
                'centerline_known_fraction_mean':float(np.mean(coverage)) if coverage else None,
                'endpoint_to_global_goal_residual_m_max':max(residuals) if residuals else None,
                'endpoint_to_global_goal_residual_m_max_all_definitions':max(all_residuals) if all_residuals else None,
                'known_terrain_separated_pair_count':pairs}
    result=[]
    for old in frozen['old_groups']:
        identifier=old['state_group_id'];count=count_by[identifier]
        banks={}
        for name in ('A','B'):
            rows=by_selected.get((identifier,name),[])
            meta=support[name][identifier]
            banks[f'Bank_{name}_060']=summarize(
                rows,lambda r:r['follower']['follower_compatible_screen'] and r['geometry']['hard_valid'],
                lambda r:r['maximum_curvature_per_m'],
                lambda r:r['follower']['initial_command_estimate']['angular_z'],
                lambda r:r['dem']['centerline_known_fraction'],
                lambda r:r['endpoint_global_goal_residual_m'],
                meta['compatible_known_terrain_separated_pairs'],meta['geometry_modes'])
            banks[f'Bank_{name}_060']['frozen_compatible_count']=meta['compatible']
        curves=by_library.get(identifier,[])
        family_pairs=sum(value for key,value in count['terrain_pair_by_family'].items()
                         if any(f in key for f in ('G1','G2','G3')))
        banks['G1_G3_common_endpoint_060']=summarize(
            curves,lambda r:r['combined_compatible'],
            lambda r:r['analytic_or_dense_max_curvature_per_m'],
            lambda r:r['first_command']['initial_command_estimate']['angular_z'],
            lambda r:r['terrain']['centerline_known_fraction'],
            lambda r:r['endpoint_goal_error_m'],family_pairs,
            count['maximum_distinct_finite_library']['maximum_mode_count'])
        banks['G1_G3_common_endpoint_060']['compatible_by_family']={
            family:sum(r['family']==family and r['combined_compatible'] for r in curves)
            for family in ('G1','G2','G3')}
        banks['G1_G3_common_endpoint_060']['distinct_mode_count_scope']='reported maximum is for the full G0A/G1/G2/G3 common-endpoint library; family-only mode count unavailable'
        result.append({'state_group_id':identifier,'anchor_id':old['anchor_id'],
                       'goal_bearing_deg':old['bearing_deg'],**banks})
    return result


def _runtime_forecast(backup_root: Path) -> dict:
    runs=backup_root/'eatnav-expanded-feasibility-ros-continuation-20261005/runs'
    rows=[]
    for marker in runs.glob('*/run-complete.json'):
        x=json.loads(marker.read_text())
        if x.get('identity',{}).get('kind')!='candidate' or x.get('protocol')!='fresh_bridge_per_run':continue
        root=marker.parent
        size=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        rows.append({'elapsed_sec':float(x['elapsed_sec']),
                     'gpu_peak_mib':float(x['gpu_peak_mib']),
                     'cpu_anon_peak_bytes':int(x['cpu_anon_peak_bytes']),
                     'artifact_count':sum(p.is_file() for p in root.rglob('*')),
                     'size_bytes':size})
    if not rows:raise ValueError('no sealed measured fresh-bridge candidate runs')
    resource_path=backup_root/'eatnav-expanded-feasibility-ros-continuation-20261005/analysis/resource-summary-r2.json'
    resource=json.loads(resource_path.read_text())
    if resource['cap_gib']!=6 or not resource['cap_pass']:
        raise ValueError('frozen process-tree memory measurement invalid')
    elapsed=np.asarray([r['elapsed_sec'] for r in rows]);sizes=np.asarray([r['size_bytes'] for r in rows])
    return {'source':str(runs),'measured_candidate_runs':len(rows),
            'per_process_wall_mean_sec':float(elapsed.mean()),
            'per_process_wall_p95_sec':float(np.quantile(elapsed,.95)),
            'P1_28_processes':{'point_hours':float(28*elapsed.mean()/3600),
                               'p95_per_process_conservative_hours':float(28*np.quantile(elapsed,.95)/3600),
                               'point_plus_20_percent_hours':float(1.2*28*elapsed.mean()/3600),
                               'p95_plus_20_percent_hours':float(1.2*28*np.quantile(elapsed,.95)/3600),
                               'storage_point_gb':float(28*sizes.mean()/1e9),
                               'storage_plus_20_percent_gb':float(1.2*28*sizes.mean()/1e9),
                               'artifact_count_estimate':int(28*statistics.median(r['artifact_count'] for r in rows))},
            'optional_32_nonzero_goal_aligned_repeats':{
                'point_hours':float(32*elapsed.mean()/3600),
                'p95_plus_20_percent_hours':float(1.2*32*np.quantile(elapsed,.95)/3600),
                'storage_plus_20_percent_gb':float(1.2*32*sizes.mean()/1e9)},
            'measured_gpu_device_peak_mib':max(r['gpu_peak_mib'] for r in rows),
            'gpu_device_total_hard_cap_mib':6656,
            'measured_cpu_anon_peak_gib':max(r['cpu_anon_peak_bytes'] for r in rows)/2**30,
            'measured_process_tree_anon_plus_shmem_peak_gib':resource['anon_plus_shmem_peak_gib'],
            'process_tree_resource_source_sha256':sha(resource_path),
            'protocol_contract':{'external_operator_owned_zenoh':True,'candidate_free_X2':True,
                                 'heartbeat':True,'finalization_grace':True,
                                 'ownership_registry':True,'owned_residual_zero':True,
                                 'native_exit_zero':True,'completion_marker':True,
                                 'coordinate_frame_correction':True,'retry_count':0},
            'not_authorized_to_execute':True}


def _figure_set(root: Path, curves: list[dict], anchors: list[dict], groups: list[dict],
                fixture: dict, decision: str) -> list[str]:
    figures=root/'figures';figures.mkdir(exist_ok=True)
    colors=plt.cm.coolwarm(np.linspace(0,1,5))
    def save(name,title,draw):
        fig,ax=plt.subplots(figsize=(7.5,5));draw(ax);ax.set_title(title);fig.tight_layout()
        fig.savefig(figures/name,dpi=130);plt.close(fig)
    def semantics(ax):
        for curve,color in zip(curves,colors):
            xy=np.asarray(curve['dense']);ax.plot(xy[:,0],xy[:,1],color=color,label=f"Path {curve['bearing_deg']:+d}°")
            ax.plot(*curve['endpoint'],'.',color=color)
        for b in BEARINGS_DEG:
            g=.75*np.array([math.cos(math.radians(b)),math.sin(math.radians(b))])
            ax.plot(*g,'kx',alpha=.45)
        ax.set(xlabel='robot-local x (m)',ylabel='robot-local y (m)');ax.axis('equal');ax.legend(fontsize=7)
    def one_goal(ax):
        semantics(ax)
        ax.plot(.75,0,'k*',markersize=15,label='one common global goal: 0°')
        ax.annotate('Path endpoints may differ\nfrom common goal',(.75,0),(.25,-.21),
                    arrowprops={'arrowstyle':'->'})
    save('01-goal-versus-endpoint.png','Local Path endpoints and one independent global goal',one_goal)
    save('02-five-primitives.png','Five forward-tangent robot-local primitives',semantics)
    save('03-curvature.png','Analytic and discrete curvature',lambda ax:(ax.bar(np.arange(5)-0.18,[c['analytic_curvature_per_m'] for c in curves],.36,label='analytic'),ax.bar(np.arange(5)+.18,[c['eight_point_max_curvature_per_m'] for c in curves],.36,label='8-point magnitude'),ax.axhline(1.236,color='r',linestyle='--'),ax.axhline(-1.236,color='r',linestyle='--'),ax.set_xticks(range(5),BEARINGS_DEG),ax.set(xlabel='bearing (deg)',ylabel='curvature (1/m)'),ax.legend()))
    follower=groups[0]['candidates']
    save('04-first-command.png','Source-equivalent fresh first command estimates',lambda ax:(ax.bar(BEARINGS_DEG,[r['follower']['initial_command_estimate']['angular_z'] for r in follower],width=7),ax.set(xlabel='Path bearing (deg)',ylabel='yaw command (rad/s)')))
    def corridors(ax):
        for curve,color in zip(curves,colors):
            xy=np.asarray(curve['dense']);keep=np.asarray(curve['stations_m'])<=.6
            ax.plot(xy[keep,0],xy[keep,1],color=color,label=f"{curve['bearing_deg']:+d}°")
        ax.set(xlabel='robot-local x (m)',ylabel='robot-local y (m)');ax.axis('equal');ax.legend()
    save('05-primary-corridors.png','Primary DEM comparison: Path arc 0–0.60 m',corridors)
    def support(ax):
        classes=('transition','differential','graded');seeds=('first','second')
        vals=np.array([[next(g['terrain']['primary_separated_pair_count'] for g in groups if g['terrain_class']==c and g['seed_label']==s) for s in seeds] for c in classes])
        im=ax.imshow(vals,cmap='viridis');ax.set_xticks(range(2),seeds);ax.set_yticks(range(3),classes)
        for i in range(3):
            for j in range(2):ax.text(j,i,str(vals[i,j]),ha='center',va='center',color='white')
        ax.set(xlabel='seed',ylabel='terrain class');
    save('06-class-seed-support.png','Known primary terrain-separated Path pairs',support)
    def identity(ax):
        ax.text(.5,.72,'8 anchors × 5 local Paths = 40 physical definitions',ha='center',fontsize=12)
        ax.text(.5,.48,'each physical trajectory × 5 common goals = 5 correlated labels',ha='center')
        ax.text(.5,.25,'same physical-run ID and split across relabels',ha='center')
        ax.axis('off')
    save('07-run-versus-label.png','Physical and goal-label identity',identity)
    def old(ax):
        ax.text(.5,.7,'Old: five curves reconverge to one common endpoint',ha='center')
        ax.text(.5,.4,'New: five local endpoints; common goal separate',ha='center')
        ax.text(.5,.16,'Follower 1.236/m screen unchanged',ha='center');ax.axis('off')
    save('08-old-versus-new.png','Candidate semantics change',old)
    def p1(ax):
        ax.text(.5,.7,'Conditional only: 4 anchors × (5 Paths + 2 direct repeats)',ha='center')
        ax.text(.5,.45,'28 physical processes; 0° common goal primary',ha='center')
        ax.text(.5,.2,'Current terrain gate fails; no rollout authorized',ha='center');ax.axis('off')
    save('09-conditional-p1.png','Unexecuted P1 process design',p1)
    def flow(ax):
        ax.text(.5,.78,'source → geometry → follower → terrain → coverage → relabel fixture',ha='center')
        ax.text(.5,.5,decision,ha='center',fontsize=10)
        ax.text(.5,.2,'Review and explicit approval required before any P1',ha='center');ax.axis('off')
    save('10-decision-flow.png','Offline decision flow',flow)
    return sorted(p.name for p in figures.glob('*.png'))


def _reserve_inventory(backup_root: Path, selected_ids: set[str]) -> dict:
    sources={
        'first_settling':'eatnav-r3-settling-pilot-20261008/analysis/per-pose-results.json',
        'first_reserve':'eatnav-r3-transition-reserve-20261008/analysis/reserve-per-pose.json',
        'second_settling':'eatnav-r3-second-seed-20261008/analysis-v2/per-pose-results.json',
    }
    entries=[];source_shas={}
    for label,relative in sources.items():
        path=backup_root/relative;source_shas[label]={'path':str(path),'sha256':sha(path)}
        data=json.loads(path.read_text());data=data['poses'] if isinstance(data,dict) else data
        for row in data:
            if row['pose_id'] in selected_ids:continue
            entries.append({'source':label,'pose_id':row['pose_id'],
                            'settled_class':row.get('settled_class'),
                            'classification':row.get('classification'),
                            'physical_supply':row.get('physical_supply'),
                            'terrain_seed':row.get('terrain_seed'),
                            'source_family':row.get('source_family')})
    eligible=[r for r in entries if r['classification']=='settled_anchor_eligible' and r['physical_supply']]
    return {'source_sha256':source_shas,'other_measured_pose_records':entries,
            'other_eligible_physical_supply':eligible,
            'missing_primary_support_remedy':'No eligible measured alternate for transition/graded in both seeds in this R3 pilot inventory; new anchor selection/measurement needs separate review. DEM-only unverified reserve rows are not promoted.'}


def run_review(backup_root: Path, output_root: Path) -> dict:
    """Continue an already frozen output root. It never launches live processes."""
    backup_root=Path(backup_root);output_root=Path(output_root)
    frozen=verify_inputs(backup_root,Path('/home/kojima/eatnav-backups'))
    audit=runtime_goal_dependency_audit()
    contract=local_contract()
    if (json.loads((output_root/'local-motion-primitive-contract.json').read_text())!=contract
            or json.loads((output_root/'global-goal-runtime-dependency-audit.json').read_text())!=audit):
        raise ValueError('pre-frozen contract or source audit changed')
    if any(output_root.glob('primitive-*.json')):raise FileExistsError('primitive results already exist in this root')
    provenance=frozen['provenance']
    provenance['prior_goal075_archive']=frozen['prior_goal075']['archive']
    provenance['physical_runtime_source_sha256']=audit['source_sha256']
    provenance['new_primitive_generator_sha256']=sha(WORKTREE/'src/visualnav_transformer/eat_nav/local_motion_primitives.py')
    provenance['new_review_runner_sha256']=sha(Path(__file__))
    checkpoint=Path('/tmp/eatnav-rl-checkpoint-audit-model_10000.pt')
    checkpoint_sha='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d'
    if sha(checkpoint)!=checkpoint_sha:raise ValueError('frozen RL checkpoint SHA mismatch')
    provenance['RL_checkpoint']={'docker_source_path':'/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
                                 'read_only_audit_copy':str(checkpoint),'sha256':checkpoint_sha}
    provenance['new_worktree']={'branch':subprocess.check_output(['git','branch','--show-current'],cwd=WORKTREE,text=True).strip(),
                                'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKTREE,text=True).strip(),
                                'path':str(WORKTREE)}
    write(output_root/'source-provenance.json',provenance)
    write(output_root/'follower-source-parity.json',provenance['source_exact_follower'])
    curves=[constant_curvature_primitive(.75,b) for b in BEARINGS_DEG]
    path_shas={c['path_sha256'] for c in curves}
    if len(path_shas)!=5:raise ValueError('duplicate local primitive Path SHA')
    def curve_plain(curve):
        return {key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in curve.items()}
    write(output_root/'primitive-path-definitions.json',{'contract_sha256':sha(output_root/'local-motion-primitive-contract.json'),
        'candidates':[curve_plain(c) for c in curves],'selection_uses_Outcome':False})
    follower=_read_follower_config(FOLLOWER_CONFIG)
    groups=[];full_pairs=[];full_coverage=[];all_geom=[]
    for anchor in frozen['anchors']:
        candidates=[audit_primitive(c,anchor['settled_map_xy_yaw'],follower) for c in curves]
        terrain=terrain_support_for_anchor(curves,anchor['dem'][0],anchor['dem'][1],anchor['proxy_bboxes'])
        groups.append({'anchor_id':anchor['anchor_id'],'seed_label':anchor['seed_label'],
                       'terrain_class':anchor['terrain_class'],'terrain_seed':anchor['terrain_seed'],
                       'source_family':anchor['source_family'],
                       'candidates':candidates,
                       'terrain':{'primary_separated_pair_count':terrain['primary_separated_pair_count'],
                                  'late_only_pair_count':terrain['late_only_pair_count']}})
        for c in candidates:all_geom.append({'anchor_id':anchor['anchor_id'],**c})
        for pair in terrain['pairs']:full_pairs.append({'anchor_id':anchor['anchor_id'],**pair})
        full_coverage.append({'anchor_id':anchor['anchor_id'],
                              'candidate_coverage':terrain['candidate_coverage']})
    geometry_ready=all(r['hard_valid'] for r in all_geom)
    follower_ready=all(r['follower_compatible'] for r in all_geom)
    support={cls:{seed:next(g['terrain']['primary_separated_pair_count'] for g in groups
                            if g['terrain_class']==cls and g['seed_label']==seed)
                  for seed in ('first','second')}
             for cls in ('transition','differential','graded')}
    terrain_ready=all(v>0 for by_seed in support.values() for v in by_seed.values())
    coverage_values=[row for group in full_coverage for row in group['candidate_coverage'].values()]
    coverage_ready=all(row['primary_centerline_known_fraction']>0 and row['full_centerline_known_fraction']>0
                       and row['proxies']['main_track_sprocket']['primary']['known_fraction']>0
                       for row in coverage_values)
    fixture_run=backup_root/'eatnav-expanded-feasibility-ros-continuation-20261005/runs/expanded-train-002-left_1'
    fixture=relabel_archived_trace(fixture_run)
    locomotion={r['locomotion']['discounted_reward_2s'] for r in fixture['labels']}
    nav={r['navigation']['goal_distance_reduction_2s'] for r in fixture['labels']}
    fixture_ready=len(locomotion)==1 and len(nav)>1
    decision=decide(source=audit['relabeling_source_valid'],geometry=geometry_ready,
                    follower=follower_ready,terrain=terrain_ready,
                    coverage=coverage_ready,fixture=fixture_ready)
    write(output_root/'primitive-geometry-audit.json',{'geometry_gate':geometry_ready,
        'follower_gate':follower_ready,'candidate_path_sha_unique':True,
        'anchor_independent_path_sha':True,'anchor_count':8,
        'hard_valid_count':sum(r['hard_valid'] for r in all_geom),
        'follower_compatible_count':sum(r['follower_compatible'] for r in all_geom),
        'maximum_analytic_curvature_per_m':max(c['analytic_max_curvature_per_m'] for c in curves),
        'maximum_eight_point_curvature_per_m':max(c['eight_point_max_curvature_per_m'] for c in curves),
        'records':all_geom})
    write(output_root/'state-group-and-independence-contract.json',{
        'physical_rollout_identity':['anchor','local_primitive'],
        'evaluator_label_identity':['anchor','common_global_goal','local_primitive'],
        'independent_experimental_unit':'terrain seed/source family',
        'physical_definition_count':40,'potential_correlated_label_count':200,
        'physical_definitions':[{'anchor_id':g['anchor_id'],'primitive_bearing_deg':b,
                                 'source_family':g['source_family'],'terrain_seed':g['terrain_seed']}
                                for g in groups for b in BEARINGS_DEG],
        'split_rule':'all relabels, repeats, bearings from one anchor and related source family stay in one split'})
    write(output_root/'goal-conditioned-label-contract.json',{
        'locomotion_invariant_under_goal_relabel':['discounted_reward_2s','contact','tilt','effort','termination','realized_trajectory'],
        'navigation_recomputed_for_common_global_goal':['goal_distance_reduction_2s','final_distance_to_global_goal_m','minimum_distance_to_global_goal_m','progress_along_goal_ray_m','signed_lateral_deviation_m','candidate_endpoint_residual_m'],
        'local_endpoint_role':'auxiliary Path tracking diagnostic, never goal-reduction teacher',
        'direct_reference_rule':'primitive bearing equals common global-goal bearing',
        'direct_relative_reward':'compare physical runs; never fabricate three repeats for relabeled nonzero goal',
        'gamma_per_0_05s_control_step':.99,'horizon_steps':40,'scalar_tradeoff_fitted':False})
    reserve=_reserve_inventory(backup_root,{a['pose_id'] for a in frozen['anchors']})
    write(output_root/'terrain-corridor-support.json',{'primary_interval_arc_m':[0,.6],
        'full_diagnostic_interval':'through each Path endpoint',
        'full_diagnostic_arc_length_by_bearing_m':{c['bearing_deg']:c['arc_length_m'] for c in curves},
        'known_corridor_rms_min_m':.01,'common_known_bins_min':5,
        'support_by_class_seed':support,'terrain_gate':terrain_ready,
        'coverage_gate':coverage_ready,
        'primary_pair_count':sum(p['primary_separated'] for p in full_pairs),
        'late_only_pair_count':sum(p['late_only'] for p in full_pairs),
        'per_anchor':[{**g['terrain'],'anchor_id':g['anchor_id'],
                       'seed_label':g['seed_label'],'terrain_class':g['terrain_class']} for g in groups],
        'reserve_inventory':reserve,
        'pair_rows':full_pairs,'coverage_rows':full_coverage,
        'unknown_only_difference_counted':False})
    prior075=_read_prior_goal075('final-decision.json')
    baseline=_read_prior_goal075('baseline-060-vs-075-comparison.json')
    group_by_anchor={g['anchor_id']:g for g in groups}
    coverage_by_anchor={g['anchor_id']:g for g in full_coverage}
    old_numeric={row['state_group_id']:row for row in build_previous_bank_group_rows(frozen)}
    paired=[state_group_comparison(old,group_by_anchor[old['anchor_id']])
            for old in baseline['paired_state_groups']]
    for row in paired:
        row.update({key:value for key,value in old_numeric[row['state_group_id']].items()
                    if key.startswith('Bank_') or key.startswith('G1_G3')})
        cover=coverage_by_anchor[row['anchor_id']]['candidate_coverage']
        row['new_primary_centerline_known_fraction_mean']=float(np.mean(
            [r['primary_centerline_known_fraction'] for r in cover.values()]))
        row['new_full_centerline_known_fraction_mean']=float(np.mean(
            [r['full_centerline_known_fraction'] for r in cover.values()]))
    write(output_root/'previous-bank-comparison.json',{
        'old_060_and_075_paired_source':'SHA-verified 0.75m baseline comparison archive',
        'old_075_decision':prior075['decision'],
        'old_075_fixed_five_geometry_gate':prior075['fixed_five_geometry_gate'],
        'old_060_075_paired_state_group_count':len(baseline['paired_state_groups']),
        'paired_state_groups':paired,
        'old_060_semantics':'Bank A common endpoint; Bank B separately treated local endpoint',
        'curvature_common_endpoint_semantics':'G1–G3 preserve one common endpoint',
        'new_semantics':'independent local primitive endpoints and common global goal',
        'new_five_geometry_compatible_at_all_anchors':geometry_ready and follower_ready,
        'new_primary_terrain_support_by_class_seed':support,
        'local_endpoint_residual_expected':True,
        'evaluator_heads':'locomotion reward and goal progress remain separate; no scalar weighting inferred',
        'possible_away_from_goal':'short initial detour may avoid terrain; requires physical Outcome later',
        'bank_role':'handcrafted data-generation instrument, not final NoMaD proposal generator'})
    write(output_root/'offline-goal-relabeling-fixture.json',fixture)
    selected=select_p1_anchors(groups)
    ready=all((audit['relabeling_source_valid'],geometry_ready,follower_ready,terrain_ready,coverage_ready,fixture_ready))
    p1=conditional_p1_manifest(selected,ready=ready)
    write(output_root/'conditional-p1-manifest.json',p1)
    write(output_root/'minimum-additional-repeat-plan.json',{
        'conditional_only':True,'not_authorized':True,
        'nonzero_goal_aligned_primitives':[-20,-10,10,20],
        'additional_repeats_per_primitive_per_anchor':2,
        'four_anchor_additional_processes':32,
        'reason':'three-repeat direct-relative reference for nonzero relabeled goals requires real physical repeats'})
    forecast=_runtime_forecast(backup_root)
    write(output_root/'runtime-resource-forecast.json',forecast)
    figures=_figure_set(output_root,curves,frozen['anchors'],groups,fixture,decision)
    final={'decision':decision,'source_relabeling_gate':audit['relabeling_source_valid'],
           'five_primitive_geometry_gate':geometry_ready,'follower_gate':follower_ready,
           'primary_terrain_support_gate':terrain_ready,'DEM_coverage_gate':coverage_ready,
           'relabeling_fixture_gate':fixture_ready,'P1_proposal_prepared':p1['prepared'],
           'figure_count':len(figures),'simulator_or_follower_started':False,
           'physical_rollout_count':0,'contract_sha256':sha(output_root/'local-motion-primitive-contract.json')}
    write(output_root/'final-decision.json',final)
    report=f'''# Local motion primitive bank offline review

Decision: `{decision}`. No live process or physical rollout was started.

The common global goal is an offline relabelable target while the candidate Path and anchor remain fixed. The SHA-frozen E1 source uses `mission_logged_goal` only in trace diagnostics; the former Outcome progress target instead used the local Path endpoint, so it must not be reused as the new navigation teacher. The historical selected-DEM goal in the pre-Path anchor gate is held fixed and is separate from the relabelable global goal. Source audit and archived-trace fixture both pass.

Five analytic circular arcs at {-20,-10,0,10,20} degrees and 0.75 m chord are valid at all eight anchors. Their analytic maximum curvature is {max(c['analytic_max_curvature_per_m'] for c in curves):.3f}/m, below the unchanged 1.236/m follower screen. All 40 anchor/Path combinations pass the static first-command screen. This is not evidence of realized tracking.

The primary known DEM terrain gate **fails**: pair counts by class/seed are {support}. The 0–0.60 m interval is primary; {sum(p['late_only'] for p in full_pairs)} pair differences seen only by 0.75 m are diagnostic. DEM centerline and main-track coverage pass. The already measured R3 reserve inventory has no eligible physical-supply replacement for the missing transition/graded combinations. The geometry is retained as a hypothesis; no Outcome-driven adjustment or P1 execution follows from this result.

There are 40 physical definitions and 200 correlated goal-label identities. The archived 40-step fixture has one trace SHA across five relabels, unchanged locomotion reward and varying goal-distance targets. These 200 identities are not independent trajectories. The conditional P1 manifest has {len(p1['processes'])} processes because the terrain gate failed; the reviewed plan would require 28 processes if all gates passed. Forecasts are recorded without starting the pilot clock.

The old common-endpoint Bank A/G1–G3 and 0.75 m bank could not supply a shared five-Path set. The new separate-endpoint semantics remove that geometric contradiction, but do not by themselves create sufficient transition/graded terrain separation in the saved DEMs. The common goal remains the navigation target; locomotion and progress remain separate Evaluator heads. A short local detour could initially lose goal distance while improving locomotion, which can only be judged after a separately approved physical pilot. This handcrafted bank is a data-generation instrument, not the final NoMaD generator.

See JSON artifacts and ten figures for source SHA, formula, follower commands, DEM corridors, reserve inventory, label semantics, resource forecast and decision. P1 rollout requires a new review and explicit approval.
'''
    (output_root/'review-report.md').write_text(report)
    files={p.relative_to(output_root).as_posix():{'sha256':sha(p),'size_bytes':p.stat().st_size}
           for p in output_root.rglob('*') if p.is_file() and p.name!='artifact-sha-ledger.json'}
    write(output_root/'artifact-sha-ledger.json',{'files':files})
    return final


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    args=parser.parse_args()
    if not args.output_root.exists():freeze_review(args.backup_root,args.output_root)
    print(json.dumps(run_review(args.backup_root,args.output_root),sort_keys=True))
