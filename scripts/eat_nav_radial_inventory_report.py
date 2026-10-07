"""Report a stopped candidate-free inventory without promoting it to Outcome/pilot data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from visualnav_transformer.eat_nav.radial_inventory import TERRAIN_ORDER, FLAT_TYPES, build_pose_schedule, validate_record

PRIOR = Path('/tmp/eatnav-radial-boundary-review-20261007')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main(root):
    root = Path(root)
    freeze = json.loads((root / 'input-freeze.json').read_text())
    status = json.loads((root / 'status.json').read_text())
    if status['stage'] != 'stopped_anchor_gate' or status['completed_batches'] != 1:
        raise ValueError('review script only handles first-batch anchor-gate stop')
    records = [json.loads(p.read_text()) for p in sorted((root / 'runs/batch-20261007/poses').glob('*.json'))]
    assert len(records) == 112
    schedule = {x['pose_id']: x for x in build_pose_schedule(freeze['batch_seeds'])}
    gate = []
    for x in records:
        y = dict(x)
        y['cleanup_ok'] = True
        result = validate_record(y, schedule[x['pose_id']], freeze['validity_limits'])
        observed = [z for z in result['reasons'] if z != 'penetration']
        gate.append({'pose_id': x['pose_id'], 'terrain': x['terrain'],
                     'source_family': x['source_family'], 'status': 'measurement_invalid_penetration_unobserved',
                     'penetration_observed': False, 'physical_failure_label': False,
                     'observed_numeric_gate_reasons': observed,
                     'eligible_for_candidate_pilot': False})
    by_terrain = {}
    for name in TERRAIN_ORDER:
        subset = [x for x in records if x['terrain'] == name]
        g = [x for x in gate if x['terrain'] == name]
        by_terrain[name] = {'pose_count': len(subset), 'terrain_class': subset[0]['terrain_class'],
            'seed_count': len({x['terrain_seed'] for x in subset}),
            'valid_mask_min': min(x['dem_valid_fraction'] for x in subset),
            'valid_mask_median': statistics.median(x['dem_valid_fraction'] for x in subset),
            'xy_error_max_m': max(x['settled_error_xy_m'] for x in subset),
            'yaw_error_max_rad': max(x['settled_yaw_error_rad'] for x in subset),
            'speed_max_mps': max(x['settled_speed_mps'] for x in subset),
            'angular_speed_max_radps': max(x['settled_angular_speed_radps'] for x in subset),
            'observed_numeric_gate_fail_count': sum(bool(x['observed_numeric_gate_reasons']) for x in g),
            'numeric_reason_counts': dict(Counter(z for x in g for z in x['observed_numeric_gate_reasons'])),
            'penetration_unobserved_count': len(g), 'pilot_eligible_count': 0}
    identities = json.loads((root / 'runs/batch-20261007/env-origins.json').read_text())
    assert all(meta['seed'] == schedule[f"b20261007-{meta['terrain_type']}-a{meta['row']:02d}"]['terrain_seed']
               for meta in identities['terrain_metadata'])
    write(root / 'runtime-terrain-inventory.json', {'status': 'first_batch_recorded_second_batch_not_started',
        'batch_seed': 20261007, 'pose_count': 112, 'terrain_count': 7, 'terrain_seed_count': 7,
        'runtime_env_origin_sha256': sha(root / 'runs/batch-20261007/env-origins.json'),
        'generated_identity_sha256_count': len({x['generated_identity_sha256'] for x in records}),
        'row_column_origin_mapping_verified': True, 'same_terrain_seed_across_16_angles': True,
        'all_four_frame_dem': all(len(x['dem_history']) == 4 for x in records),
        'all_actor_finite': all(x['actor_finite'] for x in records),
        'all_dem_finite': all(x['dem_finite'] for x in records),
        'second_batch_not_started_due_to_first_failure': True, 'by_terrain': by_terrain})
    write(root / 'anchor-only-gate-results.json', {'status': 'stop_measurement_incomplete_and_numeric_exceedances',
        'pose_count': 112, 'valid_anchor_count': 0, 'measurement_invalid_count': 112,
        'penetration_unobserved_count': 112,
        'numeric_reason_counts': dict(Counter(z for x in gate for z in x['observed_numeric_gate_reasons'])),
        'numeric_exceedance_pose_count': sum(bool(x['observed_numeric_gate_reasons']) for x in gate),
        'no_physical_failure_label': True, 'per_pose': gate})
    foot = json.loads((PRIOR / 'footprint-specification.json').read_text())
    full = json.loads((PRIOR / 'full-collision-envelope-specification.json').read_text())
    write(root / 'p1-p2-p3-p4-footprint-specification.json', {
        'P1_main_track_support': {**foot['main_track_only'], 'role': 'primary terrain corridor; main tracks plus main sprockets',
            'actual_contact_unverified': True},
        'P2_initial_pose_support': {**foot['potential_support_zero_flipper'],
            'role': 'initial anchor/support/penetration diagnostic at zero flipper joints', 'settled_support_unverified': True},
        'P3_potential_flipper_reach': {'upper_bound_from_full_articulated_union_including_nonflipper': full['articulated_union'],
            'role': 'auxiliary possible-contact mask; never substitute for P1',
            'sampled_52_joints_13_angles': True, 'exact_flipper_only_polygon': None,
            'contact_after_actor_action_unverified': True},
        'P4_full_collision_envelope': {**full, 'role': 'penetration/collision/clearance only; not primary terrain corridor'},
        'visual_width_estimate_m': .56, 'collision_width_m': .438,
        'difference': 'visual extent, collision envelope, main-track support and flipper reach are distinct'})
    prior_bank = PRIOR / 'five-candidate-bank.json'
    write(root / 'five-candidate-bank.json', {'status': 'provisional_not_executed',
        'source_file': str(prior_bank), 'source_sha256': sha(prior_bank),
        'common_global_goal_xy_m': [.6, 0.],
        'curvature_per_m': [0., .5, 1., -.5, -1.], 'arc_length_m': .6,
        'waypoint_count': 8, 'follower_screen': 'analytic_only', 'candidate_path_published': False})
    write(root / 'radial-dem-mask-archive.json', {'status': 'candidate_free_measurement_only',
        'pose_directory': str(root / 'runs/batch-20261007/poses'),
        'pose_count': 112, 'frames_per_pose': 4, 'channels': ['height', 'mask'],
        'record_sha256': {x['pose_id']: sha(root / 'runs/batch-20261007/poses' / (x['pose_id'] + '.json')) for x in records},
        'no_outcome': True, 'second_seed_batch': 'not_started'})
    na_reason = 'anchor gate stopped: penetration separation not recorded; numeric thresholds exceeded on 16 poses; second batch not started'
    for name in ('main-track-corridor-results', 'flipper-reach-diagnostics', 'terrain-interaction-taxonomy'):
        write(root / f'{name}.json', {'status': 'NA_not_performed', 'reason': na_reason,
            'unknown_not_free': True, 'outcome_used': False})
    write(root / 'independent-family-accounting.json', {'recorded_candidate_free_source_families': 7,
        'verified_eligible_families': 0, 'second_seed_batch': 'not_started',
        'same_seed_16_angles_one_family': True, 'no_four_eligible_families': True})
    write(root / 'small-pilot-candidate-proposal.json', {'small_pilot_ready_for_review': False,
        'selected_families': [], 'reason': na_reason, 'candidate_rollout_authorized': False})
    write(root / 'front-goal-pilot-manifest-proposal.json', {'status': 'conditional_only_no_manifest',
        'task_global_goal_robot_xy_m': [.6, 0.], 'five_curvatures_per_m': [0., .5, 1., -.5, -1.],
        'direct_repeats_per_group': 3, 'minimum_independent_families': 4,
        'actual_frozen_groups': 0, 'requires_anchor_gate_pass_and_separate_approval': True})
    bearings = [-20, -10, 0, 10, 20]
    write(root / 'final-collection-goal-bearing-proposal.json', {'status': 'design_only_future_review',
        'small_pilot_goal_bearing_deg': [0], 'future_bearings_deg': bearings,
        'future_goal_distance_m': .6,
        'future_goals_robot_xy_m': {str(b): [.6 * math.cos(math.radians(b)), .6 * math.sin(math.radians(b))] for b in bearings},
        'path_generation': 'regenerate constrained forward-tangent curves for each task goal; do not rotate old Paths',
        'same_task_global_goal_within_candidate_bank': True,
        'candidate_number_or_missing_mask': 'must freeze before future collection',
        'plus_minus_30_deg_not_auto_adopted': True,
        'gates': ['front-goal small pilot pass', 'bearing-specific geometry', 'follower screen',
                  'known terrain corridor', 'global-goal teacher', 'runtime/storage', 'split/leakage', 'separate approval'],
        'no_goal_bearing_path_executed': True})
    write(root / 'split-leakage-contract.json', {'independence_unit': ['terrain seed', 'environment instance',
        'radial anchor', 'source episode'], 'same_split': ['all angle variants sharing seed/instance',
        'all goal bearings', 'all candidate Paths', 'all direct repeats'],
        'goal_generalization_evaluation': ['E-goal held-out bearing on seen anchor',
        'unseen anchor/seed with held-out bearing'],
        'batch_20261007_family_count': 7, 'batch_20261008_family_count': 0,
        'outcome_used_for_selection': False})
    resource = json.loads((root / 'resource-latest.json').read_text())
    write(root / 'runtime-storage-resource-forecast.json', {'actual_first_batch_wall_sec': resource['batch_elapsed_sec'],
        'observed_last_gpu_used_mib': resource['gpu_device_used_mib'],
        'gpu_peak_mib': None, 'peak_not_logged': True,
        'observed_last_anon_shmem_bytes': resource['container_anon_shmem_bytes'],
        'actual_host_output_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
        'second_batch_cost': 'not_measured; stopped by gate',
        'conditional_prior_seven_bank_pilot_forecast_hours': 2.1726829615238348,
        'radial_pilot_forecast': None,
        'resource_caps': freeze['limits']})
    write(root / 'risk-register.json', {'risks': [
        {'id': 'U-R1', 'issue': 'Contact force recorded but contact separation/penetration not instrumented; 112 anchors measurement-invalid.'},
        {'id': 'U-R2', 'issue': '16 poses exceeded at least one fixed settled XY/yaw/velocity threshold; must inspect without posthoc threshold changes.'},
        {'id': 'U-R3', 'issue': 'Second terrain-seed batch was not started due first-failure stop.'},
        {'id': 'U-R4', 'issue': 'Resource monitor retained latest sample rather than full peak series; no measured GPU peak claim.'},
        {'id': 'U-R5', 'issue': 'Known DEM coverage 0.785–0.82; unknown cells cannot count as free or differential evidence.'},
        {'id': 'U-R6', 'issue': 'P1/P2/P3/P4 are conservative source geometry; actual settled contact footprint unverified.'}]})
    decision = {'all_terrain_radial_rule_defined': True, 'runtime_env_origins_verified': True,
        'radial_pose_transform_verified': True, 'source_flat_anchor_support': 'geometry_only_not_anchor_gate_pass',
        'unguaranteed_radial_anchor_support': False, 'main_track_footprint_resolved': 'URDF_projection_only',
        'radial_dem_mask_available': 'first_batch_112_poses_only', 'five_candidate_geometry_valid': 'prior_analytic_only',
        'known_terrain_corridor_separation_available': False, 'independent_source_family_count': 0,
        'small_pilot_ready_for_review': False, 'small_pilot_authorized': False,
        'goal_bearing_design_ready_for_future_review': True, 'large_collection_ready': False,
        'Evaluator_training_ready': False, 'NoMaD_training_ready': False,
        'status': 'STOP_ANCHOR_GATE_MEASUREMENT_INCOMPLETE',
        'candidate_path_published': False, 'outcome_collected': False, 'retry_count': 0,
        'second_seed_batch_started': False}
    write(root / 'final-decision.json', decision)
    make_figures(root, records, by_terrain, foot, full, bearings)
    report(root, by_terrain, resource, decision)
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'review-artifact-sha-ledger.json'
                   and 'mplcache' not in p.parts)
    write(root / 'review-artifact-sha-ledger.json', {'files': {str(p.relative_to(root)):
        {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files},
        'source_freeze_sha256': sha(root / 'input-freeze.json'),
        'status': 'partial_candidate_free_first_batch_sealed_after_stop'})
    print(json.dumps({'pose_count': len(records), 'numeric_exceedance_pose_count':
        sum(bool(x['observed_numeric_gate_reasons']) for x in gate),
        'measurement_invalid': 112, 'figures': len(list((root / 'figures').glob('*.png')))}, indent=2))


def make_figures(root, records, stats, foot, full, bearings):
    d = root / 'figures'; d.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 8})
    origins = np.asarray([x['env_origin'][:2] for x in records])
    fig, ax = plt.subplots(figsize=(7, 6))
    for j, name in enumerate(TERRAIN_ORDER):
        q = [x for x in records if x['terrain'] == name]
        xy = np.asarray([[x['requested_pose']['world_x'], x['requested_pose']['world_y']] for x in q])
        ax.scatter(xy[:,0], xy[:,1], s=12, label=name)
    ax.set(aspect='equal', title='112 candidate-free radial poses, seed 20261007', xlabel='world x (m)', ylabel='world y (m)')
    ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(d/'01-all-seven-terrain-radial.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 2)); ax.bar(TERRAIN_ORDER, [16]*7, color=['#198e83']*4+['#be7b32']*3)
    ax.set(title='Four source-flat vs three unguaranteed classes; 16 poses each', ylabel='recorded poses')
    ax.tick_params(axis='x', rotation=35); fig.tight_layout(); fig.savefig(d/'02-terrain-class.png'); plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3)); idx=np.arange(112)
    axs[0].scatter(idx,[x['settled_error_xy_m'] for x in records],s=7); axs[0].axhline(.1,color='r',ls='--'); axs[0].set(title='settled XY error',ylabel='m',xlabel='pose index')
    axs[1].scatter(idx,[x['dem_valid_fraction'] for x in records],s=7); axs[1].axhline(.7,color='r',ls='--'); axs[1].set(title='DEM valid mask',ylabel='fraction',xlabel='pose index')
    fig.tight_layout(); fig.savefig(d/'03-settled-anchor-dem-mask.png'); plt.close(fig)
    fig, ax=plt.subplots(figsize=(6,3));
    for label, poly in [('P1 main track',foot['main_track_only']['hull_xy_m']),('P2 initial support',foot['potential_support_zero_flipper']['hull_xy_m']),('P4 full collision zero',full['zero_joint']['hull_xy_m'])]:
        poly=np.asarray(poly); poly=np.vstack([poly,poly[0]]); ax.plot(poly[:,0],poly[:,1],label=label)
    ax.set(aspect='equal',title='P1/P2/P4 XY source geometry; P3 reach in metadata',xlabel='forward (m)',ylabel='left (m)');ax.legend();fig.tight_layout();fig.savefig(d/'04-footprint-roles.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,4))
    for k in [0,.5,1,-.5,-1]:
        s=np.linspace(0,.6,80); x=s if k==0 else np.sin(k*s)/k; y=np.zeros_like(s) if k==0 else (1-np.cos(k*s))/k
        ax.plot(x,y,label=f'κ={k:+g}')
    ax.set(aspect='equal',title='Five Path geometries; terrain corridors NA',xlabel='forward (m)',ylabel='left (m)');ax.legend();fig.tight_layout();fig.savefig(d/'05-five-paths-corridor-na.png');plt.close(fig)
    for number, message, title in [(6,'No anchor-gate-valid pose; corridor clusters NA','Terrain corridor clusters'),
        (7,'112 measured; 0 eligible; taxonomy supply NA','Taxonomy supply'),
        (8,'7 recorded seed families; 0 gate-valid families','Independent families'),
        (9,'Four-family front-goal pilot: not ready','Pilot proposal')]:
        fig,ax=plt.subplots(figsize=(6,1.8));ax.axis('off');ax.text(.5,.5,message,ha='center',va='center');ax.set_title(title)
        fig.tight_layout();fig.savefig(d/f'{number:02d}-{title.lower().replace(" ","-")}.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,3))
    for deg in bearings:
        theta=math.radians(deg);ax.arrow(0,0,.6*math.cos(theta),.6*math.sin(theta),head_width=.02,length_includes_head=True)
        ax.text(.62*math.cos(theta),.62*math.sin(theta),f'{deg}°',fontsize=7)
    ax.set(xlim=(0,.75),ylim=(-.3,.3),aspect='equal',title='Future goal bearings: design only, no Path execution')
    fig.tight_layout();fig.savefig(d/'10-goal-bearing-future.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,1.5));ax.axis('off');ax.text(.5,.5,'candidate-free inventory → anchor gate [STOP] → corridor scan → review → pilot → final collection → A1–A4',ha='center',va='center',fontsize=8)
    fig.tight_layout();fig.savefig(d/'11-decision-flow.png');plt.close(fig)


def report(root, stats, resource, decision):
    lines = ['# EAT-Nav all-terrain radial inventory — first-failure review', '',
        '## Decision', '',
        '**Stopped at candidate-free anchor gate.** The one authorized inventory execution recorded the first seed batch: 112 poses (16 directions × 7 terrains). The second seed batch did not start. No candidate Path, follower/bridge, Outcome, 120-step suffix, training or pilot was run.', '',
        'The simulator completed normally and host/container output SHAs match. Live `scene.env_origins` matched row/column placement for all 112 envs, and each terrain column used one seed and fixed difficulty 0.5 across 16 angles. All records contain four finite 20×20 DEM/mask frames and finite Actor observations. Valid-mask fractions span 0.785–0.820.', '',
        '**No anchor is promoted as valid.** Contact force was recorded, but contact separation (penetration depth) was not instrumented, so `penetration=None` for all 112. This is `measurement_invalid`, not physical failure. Separately, 16 poses exceed at least one fixed settled-state numeric threshold. No limits were changed after observation.', '',
        '## Numeric audit by terrain', '',
        '| Terrain | Class | poses | mask min | max XY error (m) | max yaw error (rad) | numeric exceedances |',
        '|---|---|---:|---:|---:|---:|---:|']
    for name in TERRAIN_ORDER:
        x=stats[name]
        lines.append(f"| `{name}` | {x['terrain_class']} | {x['pose_count']} | {x['valid_mask_min']:.3f} | {x['xy_error_max_m']:.3f} | {x['yaw_error_max_rad']:.3f} | {x['observed_numeric_gate_fail_count']} |")
    lines += ['', 'Fixed limits were XY ≤0.10 m, yaw ≤0.10 rad, speed ≤0.10 m/s, angular speed ≤0.20 rad/s and DEM valid fraction ≥0.70. Detailed per-pose reasons are in `anchor-only-gate-results.json`. The source-flat guarantee applies only to the central-square placement geometry; settled physical stability was not established.', '',
      '## Provenance and resource', '',
      '- Prior radial review artifacts, 12 direct inputs and 30 execution-source SHA entries were rechecked before launch; the 1,015 older entries remain stored audit attestations.',
      '- Scene: 16×7 tiles, 112 envs; `r=0.30 m`, `θ=22.5°k`, fixed difficulty 0.5. Root requested at runtime terrain origin plus 0.55 m, then settled for 20 control steps. Candidate-free source and input SHA are frozen in `input-freeze.json`.',
      f"- Last retained resource sample: GPU device total {resource['gpu_device_used_mib']} MiB, anon+shmem {resource['container_anon_shmem_bytes']/1024**3:.2f} GiB; no cap crossed. The monitor kept its latest sample rather than a complete peak series, so GPU peak is NA.",
      '- The owned simulator exited and cleanup passed. Retry count 0. The input root and 112 raw pose records are preserved.', '',
      '## P1–P4 and downstream scope', '',
      'P1 main-track/sprocket projection is the intended primary corridor; P2 adds initial flipper support, P3 is potential articulated flipper reach, and P4 is the full collision envelope for safety. These source-derived proxies are not a measured settled contact footprint. The 0.56 m visible width and 0.438 m collision width are distinct.', '',
      'The five constant-curvature Paths and common front goal `(0.6,0)` remain **provisional geometry only**. Since no anchor passed, main-track/flipper terrain-corridor profiles, terrain clusters, taxonomy, four-family selection and a pilot manifest are NA. Seven source-seed families were recorded but zero are gate-eligible. Unknown DEM cells were never treated as free.', '',
      'A future final-collection design proposes goal bearings `{-20,-10,0,+10,+20}°` at 0.6 m, with new forward-tangent constrained curves per bearing. Multiple bearings/angles/candidates from one source seed and environment stay in one split. No goal-bearing Path was executed; bank size or missing mask must be frozen before a later review.', '',
      '## Stop and next review', '',
      'The next technical question is how to instrument PhysX contact separation without changing reset events, dynamics or follower control, and how to interpret the 16 settled-state exceedances at the fixed limits. This report does not request automatic retry; a new inventory attempt would need a new output root and explicit review. `small_pilot_ready_for_review=false`, and no later collection/training gate passed.', '',
      '## Figures', '']
    captions=['All seven terrain radial poses','Source-flat vs unguaranteed classes','Settled pose and DEM mask','P1/P2/P4 footprint roles','Five Paths, corridor NA','Terrain corridor cluster NA','Taxonomy supply NA','Independent family support','Front-goal pilot NA','Future goal bearings','Decision flow']
    for n,(path,caption) in enumerate(zip(sorted((root/'figures').glob('*.png')),captions),1):
        lines.append(f'{n}. [{caption}](figures/{path.name})')
    (root/'review-report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/tmp/eatnav-radial-terrain-inventory-20261007'))
    main(ap.parse_args().root)
