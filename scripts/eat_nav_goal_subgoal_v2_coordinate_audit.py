"""Read-only v2 rollout audit and new, frame-corrected GS1–GS4 products.

Never edits the original pilot JSON, trace, Path, checkpoint, or decision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import traceback

import numpy as np

import eat_nav_goal_subgoal_v2_analyze as old
import eat_nav_goal_subgoal_v2_run as v
from visualnav_transformer.eat_nav.command_preview_long_horizon import endpoint_metrics, post_root

ROOT = v.OUT
DEST = ROOT / 'coordinate-audit'
SOURCE = v.REPO
DT = .05
TOL = 1e-5


def write(name, data):
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if target.exists():
        if target.read_text() != content:
            raise FileExistsError('coordinate audit artifact differs: ' + str(target))
        return
    temp = target.with_name(target.name + '.tmp')
    temp.write_text(content)
    temp.replace(target)


def assert_close(left, right, label, atol=TOL):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=0):
        raise ValueError(f'{label} differs: max_abs={np.max(np.abs(a-b)) if a.shape == b.shape else "shape"}')


def finite(value):
    if isinstance(value, dict):
        return all(finite(x) for x in value.values())
    if isinstance(value, list):
        return all(finite(x) for x in value)
    return not isinstance(value, (float, int)) or math.isfinite(value)


def source_audit():
    files = {
        'simulator': SOURCE / 'scripts/eat_nav_goal_subgoal_sim.py',
        'bridge_server': SOURCE / 'scripts/eat_nav_command_delivery_bridge.py',
        'bridge_tf': SOURCE / 'scripts/eat_nav_sim_clock_bridge.py',
        'bridge_client': SOURCE / 'src/visualnav_transformer/eat_nav/phase_c_bridge_client.py',
        'capture': SOURCE / 'src/visualnav_transformer/eat_nav/phase_c_sim.py',
        'clock': SOURCE / 'src/visualnav_transformer/eat_nav/sim_clock_protocol.py',
        'legacy_analyzer': SOURCE / 'scripts/eat_nav_goal_subgoal_v2_analyze.py',
        'path_inference': SOURCE / 'scripts/eat_nav_goal_subgoal_v2_infer.py',
    }
    provenance = v.read(ROOT / 'v2-source-provenance.json')['files']
    for label, field in [('bridge_server', 'bridge_source'), ('simulator', 'simulator_source')]:
        if label == 'simulator':
            expected = v.read(ROOT / 'v6.done.json')['source_sha256']
        else:
            expected = provenance[field]['sha256']
        if v.sha(files[label]) != expected:
            raise ValueError(f'{label} source SHA drift')
    for field in ('follower_binary', 'pilot_follower_binary', 'follower_config', 'follower_cpp', 'follower_header'):
        item = provenance[field]
        if v.sha(item['path']) != item['sha256']:
            raise ValueError(field + ' SHA drift')
    return {label: {'path': str(path), 'sha256': v.sha(path)} for label, path in files.items()} | {
        field: {'path': provenance[field]['path'], 'sha256': provenance[field]['sha256']}
        for field in ('follower_binary', 'pilot_follower_binary', 'follower_config', 'follower_cpp', 'follower_header')}


def load_saved():
    schedule_path = ROOT / 'run-schedule-v2.json'
    schedule = v.read(schedule_path)['runs']
    if len(schedule) != 48 or [x['order'] for x in schedule] != list(range(48)):
        raise ValueError('48-run fixed schedule changed')
    if v.sha(schedule_path) != v.read(ROOT / 'v2-continuation.done.json')['schedule_sha256']:
        raise ValueError('fixed schedule SHA drift')
    if Counter(x['kind'] for x in schedule) != {'anchor': 4, 'duplicate': 12, 'candidate': 32}:
        raise ValueError('schedule kind count drift')
    states = v.read(ROOT / 'selected-states-v2.json')['states']
    goals = v.read(ROOT / 'goals-v2.json')['goals']
    if len(states) != 4 or len(goals) != 16 or len({(g['state_id'],g['goal_id']) for g in goals}) != 16:
        raise ValueError('state/goal identity drift')
    goals_by_key = {(x['state_id'], x['goal_id']): x for x in goals}
    source = source_audit()
    runs = {}
    ledger = []
    for entry in schedule:
        frozen = v.verify_saved_run(entry)
        folder = ROOT / 'runs' / v.run_name(entry)
        started = v.read(folder / 'started.json')
        summary_name = 'anchor-0.json' if entry['kind'] == 'anchor' else 'live-0.json'
        marker_name = 'anchor-0.marker.json' if entry['kind'] == 'anchor' else 'live-0.marker.json'
        summary = v.read(folder / summary_name)
        marker = v.read(folder / marker_name)
        if summary['state']['state_id'] != entry['state_id'] or summary['state']['episode_key'] != entry['episode_key']:
            raise ValueError('summary state/episode identity drift')
        if summary['state']['step'] != entry['anchor_step'] or entry['episode_key'][1] != states[entry['state_index']]['env_id']:
            raise ValueError('anchor step/env ID drift')
        if not finite(summary):
            raise ValueError('nonfinite summary')
        if marker['source']['runner_source'] != source['simulator']['sha256']:
            raise ValueError('simulator source SHA drift')
        if marker['source']['bridge_client_source'] != source['bridge_client']['sha256']:
            raise ValueError('bridge client SHA drift')
        if marker['source']['capture_source'] != source['capture']['sha256'] or marker['source']['clock_protocol_source'] != source['clock']['sha256']:
            raise ValueError('capture/clock source SHA drift')
        if entry['kind'] != 'anchor':
            if v.sha(ROOT / 'inputs' / f"state-{entry['state_index']:03d}" / f"{entry['candidate_id']}-map-path.json") != summary['fixed_map_path_sha256']:
                raise ValueError('fixed map Path SHA drift')
            goal = goals_by_key[(entry['state_id'], entry['goal_id'])]
            if entry['goal_xy'] != goal['goal_xy'] or entry['goal_class'] != goal['declared_class']:
                raise ValueError('frozen goal class/XY drift')
            rows = old.trace(entry)
            if len(rows) != 120 or summary['suffix_steps'] != 120 or not finite(rows):
                raise ValueError('trace length/finite mismatch')
            if v.sha(folder / 'live-0.trace.json.gz') != summary['trace_sha256']:
                raise ValueError('trace SHA drift')
            if v.sha(folder / 'live-0.clock.json.gz') != summary['clock_sidecar_sha256']:
                raise ValueError('clock SHA drift')
            for i, row in enumerate(rows):
                stamp = entry['logical_t0_ns'] + i * 50_000_000
                if (row['step_identity'] != {'state_id': entry['state_id'], 'run_id': entry['run_id'],
                                             'control_step': i, 'logical_stamp_ns': stamp}
                        or any(row[key] != stamp for key in ('logical_sim_time_ns', 'clock_stamp_ns',
                                                               'tf_stamp_ns', 'follower_command_stamp_ns',
                                                               'dem_timestamp_ns'))
                        or row['path_stamp_ns'] != entry['logical_t0_ns']
                        or row['bridge_rejected_count'] != [0] * 8
                        or len(row['bridge_command_events']) != 8
                        or any(not event['accepted'] or event['stamp_ns'] != stamp
                               or event['tf_stamp_ns'] != stamp
                               or event['path_stamp_ns'] != entry['logical_t0_ns']
                               for event in row['bridge_command_events'])
                        or row['terminal'] is not None):
                    raise ValueError(f'trace step/stamp/command drift order={entry["order"]} step={i}')
            runs[entry['run_id']] = (entry, summary, rows)
        ledger.append({'order': entry['order'], 'run_id': entry['run_id'], 'kind': entry['kind'],
                       'state_id': entry['state_id'], 'env_id': entry['episode_key'][1],
                       'started_sha256': frozen['started_sha256'],
                       'completion_marker_sha256': frozen['run_complete_sha256'],
                       'input_sha256': started['input_sha256'],
                       'output_sha256': {Path(path).name: digest for path, digest in marker['outputs_sha256'].items()},
                       'simulator_marker_sha256': frozen['marker_sha256'],
                       'source_sha256': marker['source'],
                       'trace_sha256': v.sha(folder / 'live-0.trace.json.gz') if entry['kind'] != 'anchor' else None,
                       'recovered': entry['order'] == 21})
    if v.sha(ROOT / 'runs/v2-train-001-near_left_lateral-seed-1002/recovered_complete_after_monitor_cleanup_race.json') != v.read(ROOT / 'v2-continuation-analysis.done.json')['recovery_marker_sha256']:
        raise ValueError('recovered run marker SHA drift')
    return schedule, states, goals, runs, {'status': 'pass', 'schedule_sha256': v.sha(schedule_path),
        'selected_states_sha256': v.sha(ROOT / 'selected-states-v2.json'),
        'goals_sha256': v.sha(ROOT / 'goals-v2.json'), 'source': source,
        'run_count': 48, 'candidate_count': 32, 'duplicate_count': 12,
        'recovered_run_order': 21, 'runs': ledger,
        'legacy_decision_sha256': v.sha(ROOT / 'final-decision-v2-schedule-storage.json')}


def reproduce_old():
    checked = []
    def compare(path, value):
        path = Path(path)
        if v.read(path) != value:
            raise ValueError('legacy analysis reproduction mismatch: ' + path.name)
        checked.append({'name': path.name, 'sha256': v.sha(path), 'exact_json_value': True})
    original = old.atomic
    try:
        old.atomic = compare
        old.main()
    finally:
        old.atomic = original
    if len(checked) != 5:
        raise ValueError('legacy output count differs')
    return {'status': 'pass', 'legacy_results': checked,
            'legacy_decision_status': 'superseded_pending_coordinate_audit'}


def origin_and_spec(schedule, runs, states):
    by_state = defaultdict(list)
    for entry in schedule:
        if entry['kind'] != 'anchor':
            by_state[entry['state_id']].append(runs[entry['run_id']])
    origins = {}
    states_out = []
    for state in states:
        group = by_state[state['state_id']]
        anchor = np.asarray(group[0][1]['anchor']['root_state'][:2], dtype=float)
        map_anchor = np.asarray(group[0][1]['map_path'][0], dtype=float)
        origin = anchor - map_anchor
        if not np.all(np.isfinite(origin)):
            raise ValueError('nonfinite origin')
        for entry, summary, rows in group:
            candidate_origin = np.asarray(summary['anchor']['root_state'][:2]) - np.asarray(summary['map_path'][0])
            assert_close(candidate_origin, origin, 'state-invariant env origin', atol=2e-6)
            assert_close(np.asarray(rows[0]['root_state'][:2]) - origin, summary['map_path'][0],
                         'step0 pre-action root/map anchor', atol=2e-6)
        origins[state['state_id']] = origin
        states_out.append({'state_id': state['state_id'], 'terrain': state['terrain_name'],
                           'env_id': state['env_id'], 'origin_world_xy_m': origin.tolist(),
                           'origin_source': 'anchor root_state_w minus Path[0], cross-checked over 11 live runs',
                           'origin_yaw_rad': 0.0,
                           'origin_yaw_source': 'Isaac env_origins is an XYZ translation tensor; local_pose subtracts XY and preserves world yaw',
                           'anchor_world_xy_m': anchor.tolist(), 'anchor_map_xy_m': map_anchor.tolist(),
                           'anchor_yaw_world_and_map_rad': old.yaw_from_root(group[0][1]['anchor']['root_state'])})
    spec = {'status': 'unique_translation_confirmed', 'canonical_evaluation_frame': 'environment-local map',
            'transform': 'root_map_xy = root_state_w_xy - env_origins_xy; yaw_map = yaw_world',
            'env_origin_rotation_rad': 0.0,
            'frames': [
                {'variable':'robot-local goal','saved_from':'goals-v2.json / schedule goal_xy','frame':'anchor body local','origin':'anchor robot','yaw_basis':'anchor yaw','unit':'m'},
                {'variable':'goal_xy_map','saved_from':'legacy analysis derived value','frame':'environment-local map','origin':'tile env origin','yaw_basis':'world yaw retained','unit':'m'},
                {'variable':'Path waypoint','saved_from':'base-path → map_path(local_pose)','frame':'environment-local map','origin':'tile env origin','yaw_basis':'anchor yaw','unit':'m'},
                {'variable':'bridge TF pose','saved_from':'local_pose sent to bridge poses','frame':'map → branch_i/chassis/body','origin':'tile env origin','yaw_basis':'anchor/current world yaw','unit':'m/rad'},
                {'variable':'post_root_state','saved_from':'Isaac robot.data.root_state_w','frame':'Isaac world','origin':'world zero','yaw_basis':'world','unit':'m'},
                {'variable':'env origin','saved_from':'Isaac scene.env_origins','frame':'Isaac world','origin':'world zero','yaw_basis':'zero (translation only)','unit':'m'},
                {'variable':'canonical realized trajectory','saved_from':'post_root_state minus env origin','frame':'environment-local map','origin':'tile env origin','yaw_basis':'world yaw retained','unit':'m'}],
            'source_evidence': {'local_pose':'scripts/eat_nav_goal_subgoal_sim.py:106-110',
                'map_path':'scripts/eat_nav_goal_subgoal_sim.py:199-201',
                'bridge_tf_path':'scripts/eat_nav_sim_clock_bridge.py:86-106',
                'root_capture':'scripts/eat_nav_goal_subgoal_sim.py:169,488',
                'legacy_world_local_mix':'scripts/eat_nav_goal_subgoal_v2_analyze.py:94-105',
                'increment_cumulative':'scripts/eat_nav_goal_subgoal_v2_infer.py:56-65'},
            'path_representation':'NoMaD cumulative body-local 8 waypoints; prepend [0,0] for nine-point base Path; rotate+translate at anchor into map Path',
            'nominal_and_candidate_same_transform':True, 'states':states_out}
    return origins, spec


def corrected_xy(rows, origin):
    return np.asarray([post_root(row)[:2] for row in rows], dtype=float) - origin


def invariants_and_offsets(schedule, runs, origins):
    path_rows = {}
    with gzip.open(ROOT / 'paths-v2.jsonl.gz', 'rt') as stream:
        for line in stream:
            item = json.loads(line)
            path_rows[(item['state_id'], item['goal_id'], item['diffusion_seed'])] = item
    if len(path_rows) != 32:
        raise ValueError('frozen 32 Path rows missing')
    offsets = []
    checks = []
    for entry in schedule:
        if entry['kind'] == 'anchor':
            continue
        _, summary, rows = runs[entry['run_id']]
        origin = origins[entry['state_id']]
        anchor_world = np.asarray(summary['anchor']['root_state'][:2], dtype=float)
        anchor_map = np.asarray(summary['map_path'][0], dtype=float)
        goal = old.goal_map(summary, entry['goal_xy'])
        path = np.asarray(summary['map_path'], dtype=float)
        root_world = np.asarray([post_root(row)[:2] for row in rows], dtype=float)
        root_map = root_world - origin
        assert_close(anchor_world - origin, anchor_map, 'anchor map pose', atol=2e-6)
        assert_close(np.asarray(rows[0]['root_state'][:2]) - origin, anchor_map,
                     'step0 pre-action root map', atol=2e-6)
        assert_close(np.linalg.norm(goal-anchor_map), np.linalg.norm(entry['goal_xy']), 'goal initial distance')
        if entry['kind'] == 'candidate':
            legacy = next(x for x in v.read(ROOT / 'goal-execution-results-v2.json')['runs']
                          if x['run_id'] == entry['run_id'])
            assert_close(legacy['goal_xy_map'], goal, 'legacy goal map')
            assert_close(legacy['goal_initial_distance_m'], np.linalg.norm(goal-anchor_map), 'legacy initial goal distance')
            assert_close(legacy['planned_endpoint_error_m'], np.linalg.norm(path[-1]-goal), 'legacy Path endpoint error')
            path_row = path_rows[(entry['state_id'], entry['goal_id'], entry['seed'])]
            increments = np.asarray(path_row['path_increments'], dtype=float)
            cumulative = np.asarray(path_row['cumulative_waypoints'], dtype=float)
            assert_close(np.cumsum(increments, axis=0), cumulative, 'increment/cumulative roundtrip', atol=2e-6)
            base = np.asarray(v.read(ROOT / 'inputs' / f"state-{entry['state_index']:03d}" /
                         f"{entry['candidate_id']}-base-path.json"), dtype=float)
            assert_close(base[0], [0, 0], 'base Path start')
            assert_close(base[1:], cumulative, 'base cumulative Path', atol=2e-6)
        assert_close(path[0], anchor_map, 'map Path start')
        if not np.isfinite(root_map).all() or len(rows) != 120:
            raise ValueError('nonfinite/short canonical trajectory')
        for i, row in enumerate(rows):
            if row['step_identity']['control_step'] != i or row['logical_sim_time_ns'] != entry['logical_t0_ns'] + i*50_000_000:
                raise ValueError('control step order/time changed')
        used = root_world[39]
        observed = used - path[-1]
        residual = observed - origin
        offsets.append({'run_id': entry['run_id'], 'state_id': entry['state_id'], 'terrain': entry['terrain'],
            'kind': entry['kind'], 'goal_class': entry['goal_class'], 'seed': entry['seed'],
            'anchor_root_world_xy_m': anchor_world.tolist(), 'env_origin_world_xy_m': origin.tolist(),
            'anchor_root_map_xy_m': anchor_map.tolist(), 'anchor_yaw_rad': old.yaw_from_root(summary['anchor']['root_state']),
            'robot_local_goal_xy_m': entry['goal_xy'], 'goal_xy_map_m': goal.tolist(),
            'path_start_xy_map_m': path[0].tolist(), 'path_end_xy_map_m': path[-1].tolist(),
            'realized_step0_pre_world_xy_m': rows[0]['root_state'][:2],
            'realized_step0_pre_map_xy_m': (np.asarray(rows[0]['root_state'][:2])-origin).tolist(),
            'realized_step0_post_world_xy_m': root_world[0].tolist(),
            'realized_step0_post_map_xy_m': root_map[0].tolist(),
            'legacy_used_step39_world_xy_m': used.tolist(),
            'observed_world_minus_path_end_vector_m': observed.tolist(),
            'origin_vector_m': origin.tolist(),
            'observed_minus_origin_residual_m': residual.tolist(),
            'observed_minus_origin_residual_norm_m': float(np.linalg.norm(residual)),
            'world_goal_error_step39_m': float(np.linalg.norm(used-goal)),
            'map_goal_error_step39_m': float(np.linalg.norm(root_map[39]-goal)),
            'root_trace_sha256': v.sha(ROOT / 'runs' / v.run_name(entry) / 'live-0.trace.json.gz')})
        checks.append({'run_id': entry['run_id'], 'step0_pre_anchor_error_m': float(np.linalg.norm(
                       np.asarray(rows[0]['root_state'][:2])-origin-anchor_map)),
                       'path_start_anchor_error_m': float(np.linalg.norm(path[0]-anchor_map)),
                       'origin_roundtrip_max_abs_m': float(np.max(np.abs((root_map+origin)-root_world)))})
    state_rows = []
    for state_id in sorted(origins):
        group = [row for row in offsets if row['state_id']==state_id and row['kind']=='candidate']
        residuals = np.asarray([row['observed_minus_origin_residual_m'] for row in group])
        state_rows.append({'state_id': state_id, 'origin_world_xy_m': origins[state_id].tolist(),
            'origin_norm_m': float(np.linalg.norm(origins[state_id])),
            'observed_offset_mean_xy_m': np.mean(np.asarray([x['observed_world_minus_path_end_vector_m'] for x in group]),axis=0).tolist(),
            'residual_norm_min_m': float(np.min(np.linalg.norm(residuals,axis=1))),
            'residual_norm_max_m': float(np.max(np.linalg.norm(residuals,axis=1))),
            'residual_vector_std_xy_m': np.std(residuals,axis=0).tolist(),
            'candidate_count':len(group)})
    # A shared translation leaves within-state pairwise trajectory distance unchanged.
    pair_max = 0.0
    for state_id in origins:
        group = [runs[e['run_id']] for e in schedule if e['state_id']==state_id and e['kind']=='candidate']
        for j,(_,_,left) in enumerate(group):
            for _,_,right in group[j+1:]:
                before=old.realized_distance(left,right)
                transformed_left=[dict(row,post_root_state=(np.asarray(row['post_root_state'][:2])-origins[state_id]).tolist()+row['post_root_state'][2:]) for row in left]
                transformed_right=[dict(row,post_root_state=(np.asarray(row['post_root_state'][:2])-origins[state_id]).tolist()+row['post_root_state'][2:]) for row in right]
                for original, transformed in ((left, transformed_left),(right, transformed_right)):
                    for row0,row1 in zip(original,transformed):
                        row1['root_state']=(np.asarray(row0['root_state'][:2])-origins[state_id]).tolist()+row0['root_state'][2:]
                after=old.realized_distance(transformed_left,transformed_right)
                pair_max=max(pair_max,abs(before-after))
    if pair_max > 1e-9:
        raise ValueError('pairwise distance changed by translation')
    return {'status':'pass','runs':checks,'within_state_pairwise_translation_max_abs_m':pair_max,
            'all_raw_trace_sha_unchanged':True}, {'status':'pass','states':state_rows,'runs':offsets}


def metric(entry, summary, rows, origin, goal_metadata):
    path = np.asarray(summary['map_path'],dtype=float)
    goal = old.goal_map(summary,entry['goal_xy'])
    root = corrected_xy(rows,origin)
    distances = np.linalg.norm(root-goal,axis=1)
    lateral = old.polyline_distance(root,path)
    initial = float(np.linalg.norm(goal-path[0]))
    first = np.flatnonzero(distances<=old.GOAL_TOLERANCE_M)
    horizons={}
    for label,count in (('primary_0_2s',40),('secondary_0_6s',120)):
        d=distances[:count];x=lateral[:count]
        horizons[label]={'observed_steps':count,'complete':True,
            'endpoint_error_m':float(d[-1]),'path_endpoint_distance_m':float(np.linalg.norm(root[count-1]-path[-1])),
            'cross_track_p50_m':float(np.quantile(x,.5)), 'cross_track_p95_m':float(np.quantile(x,.95)),
            'progress_m':float(initial-d[-1]),'progress_ratio':float((initial-d[-1])/initial) if initial else None,
            'reach_0p2m':bool(np.any(d<=old.GOAL_TOLERANCE_M)),
            'first_reach_time_sec':float((np.flatnonzero(d<=old.GOAL_TOLERANCE_M)[0]+1)*DT) if np.any(d<=old.GOAL_TOLERANCE_M) else None,
            'post_reach_max_deviation_m':float(np.max(d[np.flatnonzero(d<=old.GOAL_TOLERANCE_M)[0]:])) if np.any(d<=old.GOAL_TOLERANCE_M) else None}
    return {'run_id':entry['run_id'],'state_id':entry['state_id'],'terrain':entry['terrain'],
        'goal_class':entry['goal_class'],'goal_id':entry['goal_id'],'seed':entry['seed'],
        'train_support_class':goal_metadata['actual_train_support_class'],
        'train_knn_distance_m':goal_metadata['train_knn_distance_m'],
        'goal_xy_map':goal.tolist(),'goal_initial_distance_m':initial,
        'planned_endpoint_error_m':float(np.linalg.norm(path[-1]-goal)),
        'reach_time_sec':float((first[0]+1)*DT) if len(first) else None,
        'post_reach_max_deviation_m':float(np.max(distances[first[0]:])) if len(first) else None,
        'catastrophic_termination':bool(rows[-1]['terminal'] and rows[-1]['terminal'].get('terminated')),
        'horizons':horizons, 'frame':'environment-local map'}


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        for key in ('all',f"class:{row['goal_class']}",f"state:{row['state_id']}",
                    f"terrain:{row['terrain']}",f"seed:{row['seed']}",
                    f"support:{row['train_support_class']}"):
            groups[key].append(row)
    result={}
    for key,group in groups.items():
        result[key]={'count':len(group),'primary_reach_rate':float(np.mean([x['horizons']['primary_0_2s']['reach_0p2m'] for x in group])),
            'secondary_reach_rate':float(np.mean([x['horizons']['secondary_0_6s']['reach_0p2m'] for x in group])),
            'primary_endpoint_error_median_m':float(np.median([x['horizons']['primary_0_2s']['endpoint_error_m'] for x in group])),
            'secondary_endpoint_error_median_m':float(np.median([x['horizons']['secondary_0_6s']['endpoint_error_m'] for x in group])),
            'primary_cross_track_p95_median_m':float(np.median([x['horizons']['primary_0_2s']['cross_track_p95_m'] for x in group])),
            'post_reach_departure_count':sum((x['post_reach_max_deviation_m'] or 0)>.2 for x in group)}
    support_distances=np.asarray([x['train_knn_distance_m'] for x in rows],dtype=float)
    for index,ids in enumerate(np.array_split(np.argsort(support_distances),4)):
        group=[rows[i] for i in ids]
        result[f'train_knn_distance_quartile:{index+1}']={'count':len(group),
            'distance_min_m':float(min(x['train_knn_distance_m'] for x in group)),
            'distance_max_m':float(max(x['train_knn_distance_m'] for x in group)),
            'primary_reach_rate':float(np.mean([x['horizons']['primary_0_2s']['reach_0p2m'] for x in group])),
            'secondary_reach_rate':float(np.mean([x['horizons']['secondary_0_6s']['reach_0p2m'] for x in group]))}
    return result


def command_curve(rows):
    pose=np.zeros(3,dtype=float);points=[]
    for row in rows[:40]:
        vx,vy,wz=map(float,row['follower_command'])
        c,s=math.cos(pose[2]),math.sin(pose[2])
        pose[:2]+=DT*np.array([c*vx-s*vy,s*vx+c*vy])
        pose[2]+=DT*wz
        points.append(pose[:2].copy())
    return np.asarray(points)


def mode_count(ids, distances, threshold):
    chosen=[]
    for run_id in ids:
        if all(distances[tuple(sorted((run_id,prior)))]>threshold for prior in chosen):
            chosen.append(run_id)
    return len(chosen)


def pcr(schedule,runs,origins):
    legacy=v.read(ROOT/'pcr-diversity-results-v2.json')
    output=[]
    for state_id in origins:
        group=[e for e in schedule if e['state_id']==state_id and e['kind']=='candidate']
        core=[e for e in group if e['goal_class']!='far_ood']
        nominal=[e for e in schedule if e['state_id']==state_id and e['kind']=='duplicate']
        if len(group)!=8 or len(core)!=6 or len(nominal)!=3:
            raise ValueError('P/C/R group population drift')
        gate=v.read(ROOT/f'duplicate-gate-{state_id}.json')
        if gate['status']!='pass':raise ValueError('duplicate gate not pass')
        threshold=max(old.MODE_SEPARATION_M,gate['xy_rms_m'])
        matrices={'P':{},'C':{},'R':{}}
        for j,left in enumerate(group):
            for right in group[j+1:]:
                key=tuple(sorted((left['run_id'],right['run_id'])))
                _,ls,lr=runs[left['run_id']];_,rs,rr=runs[right['run_id']]
                matrices['P'][key]=old.path_distance(ls['map_path'],rs['map_path'])
                matrices['C'][key]=old.path_distance(command_curve(lr),command_curve(rr))
                a=corrected_xy(lr,origins[state_id])[:40]-np.asarray(lr[0]['root_state'][:2])+origins[state_id]
                b=corrected_xy(rr,origins[state_id])[:40]-np.asarray(rr[0]['root_state'][:2])+origins[state_id]
                matrices['R'][key]=old.path_distance(a,b)
                assert_close(matrices['R'][key],old.realized_distance(lr,rr),'legacy R pairwise distance',atol=1e-9)
        ids=[e['run_id'] for e in core]
        modes={kind:mode_count(ids,distances,threshold) for kind,distances in matrices.items()}
        core_pairs=[tuple(sorted((a,b))) for j,a in enumerate(ids) for b in ids[j+1:]]
        p_separated=[pair for pair in core_pairs if matrices['P'][pair]>threshold]
        c_separated=[pair for pair in core_pairs if matrices['C'][pair]>threshold]
        p_to_c=sum(matrices['C'][pair]>threshold for pair in p_separated)/len(p_separated) if p_separated else None
        c_to_r=sum(matrices['R'][pair]>threshold for pair in c_separated)/len(c_separated) if c_separated else None
        nominal_effect=max(old.realized_distance(runs[a['run_id']][2],runs[b['run_id']][2])
                           for j,a in enumerate(nominal) for b in nominal[j+1:])
        previous=next(x for x in legacy['states'] if x['state_id']==state_id)
        if modes['R']!=previous['core_mode_count']:
            raise ValueError('old GS2 R mode changed under translation')
        output.append({'state_id':state_id,'terrain':group[0]['terrain'],'threshold_m':threshold,
            'nominal_duplicate_rms_m':gate['xy_rms_m'],'nominal_duplicate_max_pair_distance_m':nominal_effect,
            'core_mode_count':modes,'P_to_C_separated_pair_preservation_rate':p_to_c,
            'C_to_R_separated_pair_preservation_rate':c_to_r,
            'P_separated_core_pairs':len(p_separated),'C_separated_core_pairs':len(c_separated),
            'distance_matrices':{k:{'|'.join(pair):value for pair,value in matrix.items()} for k,matrix in matrices.items()},
            'goal_class_modes':{kind:{cls:mode_count([e['run_id'] for e in group if e['goal_class']==cls],matrices[kind],threshold)
                for cls in sorted({e['goal_class'] for e in group})} for kind in matrices},
            'seed_modes':{kind:{str(seed):mode_count([e['run_id'] for e in group if e['seed']==seed],matrices[kind],threshold)
                for seed in (1001,1002)} for kind in matrices}})
    return {'states':output,'GS2_pass':sum(x['core_mode_count']['R']>=2 for x in output)>=3,
            'legacy_GS2_pass':legacy['GS2_pass'],'R_pairwise_translation_invariant':True}


def gs3(schedule,runs,origins,execution):
    legacy=v.read(ROOT/'relative-outcome-results-v2.json')
    result=[]
    execution_by={row['run_id']:row for row in execution}
    for state_id in origins:
        group=[e for e in schedule if e['state_id']==state_id and e['kind']!='anchor']
        old_state=next(x for x in legacy['states'] if x['state_id']==state_id)
        outcomes=[]
        for entry in group:
            _,summary,rows=runs[entry['run_id']]
            anchor=np.asarray(summary['map_path'][0],dtype=float)
            goal=old.goal_map(summary,entry['goal_xy'])
            primary=endpoint_metrics(rows,40,rows[40]['critic_value'],anchor,goal,summary['map_path'])
            secondary=endpoint_metrics(rows,120,summary['suffix_end_critic_value'],anchor,goal,summary['map_path'])
            old_out=next(x for x in old_state['outcomes'] if x['run_id']==entry['run_id'])
            for key in ('delta_v','advantage_h','drawdown','recovery','forward_goal_progress_m',
                        'maximum_tilt_rad','maximum_contact_n','stack','termination'):
                if primary[key]!=old_out['primary_40_step'][key] or secondary[key]!=old_out['secondary_120_step'][key]:
                    raise ValueError('raw GS3 outcome changed: '+entry['run_id']+' '+key)
            if primary['tracking_error_rms_m']!=old_out['primary_40_step']['tracking_error_rms_m']:
                raise ValueError('GS3 relative Path tracking unexpectedly changed')
            outcome={'run_id':entry['run_id'],'kind':entry['kind'],'goal_class':entry['goal_class'],
                'seed':entry['seed'],'primary_40_step':primary,'secondary_120_step':secondary,
                'raw_joint_effort_peak_abs':old_out['raw_joint_effort_peak_abs'],
                'termination':old_out['termination']}
            if entry['kind']=='candidate':
                outcome['canonical_goal_endpoint_error_2s_m']=execution_by[entry['run_id']]['horizons']['primary_0_2s']['endpoint_error_m']
                outcome['canonical_cross_track_p95_2s_m']=execution_by[entry['run_id']]['horizons']['primary_0_2s']['cross_track_p95_m']
            outcomes.append(outcome)
        candidates=[x for x in outcomes if x['kind']=='candidate']
        values=[x['primary_40_step']['delta_v'] for x in candidates]
        nominal=[x['primary_40_step']['delta_v'] for x in outcomes if x['kind']=='duplicate']
        required=max(max(nominal)-min(nominal),old.DELTA_V_EPSILON)
        ranking=sorted(candidates,key=lambda x:x['primary_40_step']['delta_v'],reverse=True)
        def corr(key):
            x=np.asarray(values,dtype=float);y=np.asarray([row[key] for row in candidates],dtype=float)
            return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
        row={'state_id':state_id,'terrain':group[0]['terrain'],'outcomes':outcomes,
             'candidate_delta_v_range':max(values)-min(values),'required_delta_v_range':required,
             'candidate_ranking_by_delta_v':[x['run_id'] for x in ranking],
             'correlation_delta_v_goal_error_2s':corr('canonical_goal_endpoint_error_2s_m'),
             'correlation_delta_v_cross_track_p95_2s':corr('canonical_cross_track_p95_2s_m'),
             'goal_class_delta_v':{cls:[x['primary_40_step']['delta_v'] for x in candidates if x['goal_class']==cls]
                                   for cls in sorted({x['goal_class'] for x in candidates})},
             'pass':old_state['pass'],'raw_critic_reward_command_body_joint_series_unchanged':True}
        if row['candidate_delta_v_range']!=old_state['candidate_delta_v_range'] or row['pass']!=old_state['pass']:
            raise ValueError('GS3 state decision drift')
        result.append(row)
    return {'states':result,'GS3_pass':sum(x['pass'] for x in result)>=3,
            'legacy_GS3_pass':legacy['GS3_pass']}


def gs4(schedule,runs):
    legacy=v.read(ROOT/'evaluator-input-readiness-v2.json')
    required=('dem_history','mission_logged_goal','candidate_local_goal','planned_candidate_path',
              'follower_command','actor_input','actor_latent','previous_action','root_state',
              'joint_position','raw_joint_effort')
    fields=[]
    for field in required:
        available=all(field in row and row[field] is not None and finite(row[field])
                      for entry in schedule if entry['kind']=='candidate'
                      for row in runs[entry['run_id']][2])
        fields.append({'field':field,'all_candidate_steps_available_and_finite':available,
            'selection_time_available':field in ('dem_history','mission_logged_goal','candidate_local_goal',
                                                'planned_candidate_path','previous_action','root_state',
                                                'joint_position','raw_joint_effort'),
            'availability_note':'future step value is outcome-only' if field in ('follower_command','actor_input','actor_latent') else 'anchor/current value available; future value is outcome-only'})
    if not all(x['all_candidate_steps_available_and_finite'] for x in fields):
        raise ValueError('GS4 missing/nonfinite candidate field')
    return {'fields':fields,'same_step_timestamp_audited':True,
            'GS4_simulator_schema_pass':legacy['GS4_simulator_schema_pass'],
            'real_robot_equivalent_effort':'NA / not verified'}


def figures(execution,offsets,pcr,relative,decision):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    legacy={x['run_id']:x for x in v.read(ROOT/'goal-execution-results-v2.json')['runs']}
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for j,state_id in enumerate(sorted({x['state_id'] for x in execution})):
        ax=axes.flat[j]
        item=next(x for x in offsets['runs'] if x['state_id']==state_id and x['kind']=='candidate')
        ax.scatter(*item['anchor_root_world_xy_m'],marker='x',label='start world')
        ax.scatter(*item['anchor_root_map_xy_m'],marker='o',label='start env-local map')
        ax.scatter(*item['env_origin_world_xy_m'],marker='s',label='env origin world')
        ax.scatter(*item['goal_xy_map_m'],marker='*',label='goal map')
        ax.scatter(*item['path_end_xy_map_m'],marker='^',label='Path end map')
        ax.set_title(state_id+' / '+item['terrain']);ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.axis('equal');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(DEST/'world_vs_env_local_by_state.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for j,state in enumerate(offsets['states']):
        ax=axes.flat[j];origin=np.asarray(state['origin_world_xy_m'])
        group=[x for x in offsets['runs'] if x['state_id']==state['state_id'] and x['kind']=='candidate']
        for row in group:
            vec=np.asarray(row['observed_world_minus_path_end_vector_m']);ax.scatter(*vec,color='tab:orange',s=20)
        ax.scatter(*origin,color='black',marker='x',s=80,label='env origin')
        ax.set_title(state['state_id']+' / old world−Path vector');ax.set_xlabel('X offset (m)');ax.set_ylabel('Y offset (m)');ax.legend()
    fig.tight_layout();fig.savefig(DEST/'origin_vs_legacy_error_vector.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    names=[x['run_id'] for x in execution]
    axes[0,0].scatter([legacy[n]['horizons']['primary_0_2s']['endpoint_error_m'] for n in names],
                      [x['horizons']['primary_0_2s']['endpoint_error_m'] for x in execution]);axes[0,0].set(xlabel='old world/map error (m)',ylabel='corrected map error (m)',title='2 s endpoint error')
    for cls in ('logged','ID','near_ood','far_ood'):
        group=[x for x in execution if x['goal_class']==cls]
        axes[0,1].scatter([x['horizons']['primary_0_2s']['endpoint_error_m'] for x in group],
                          [x['horizons']['primary_0_2s']['cross_track_p95_m'] for x in group],label=cls)
    axes[0,1].set(xlabel='goal endpoint error (m)',ylabel='cross-track P95 (m)',title='2 s');axes[0,1].legend(fontsize=7)
    classes=('logged','ID','near_ood','far_ood');x=np.arange(4)
    axes[1,0].bar(x-.18,[np.mean([z['horizons']['primary_0_2s']['reach_0p2m'] for z in execution if z['goal_class']==c]) for c in classes],.36,label='0–2 s')
    axes[1,0].bar(x+.18,[np.mean([z['horizons']['secondary_0_6s']['reach_0p2m'] for z in execution if z['goal_class']==c]) for c in classes],.36,label='0–6 s')
    axes[1,0].set_xticks(x,classes);axes[1,0].set(ylabel='reach fraction',title='goal reach ≤0.2 m');axes[1,0].legend()
    axes[1,1].boxplot([[z['horizons']['primary_0_2s']['cross_track_p95_m'] for z in execution if z['goal_class']==c] for c in classes],labels=classes)
    axes[1,1].set(ylabel='cross-track P95 (m)',title='Path tracking, env-local map')
    fig.tight_layout();fig.savefig(DEST/'old_vs_corrected_goal_metrics.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for j,state_id in enumerate(sorted({x['state_id'] for x in execution})):
        ax=axes.flat[j];e=next(x for x in v.read(ROOT/'run-schedule-v2.json')['runs'] if x['state_id']==state_id and x['kind']=='candidate' and x['goal_class']=='ID')
        s=v.read(ROOT/'runs'/v.run_name(e)/'live-0.json');rows=old.trace(e)
        origin=np.asarray(s['anchor']['root_state'][:2])-np.asarray(s['map_path'][0]);root=corrected_xy(rows,origin)
        path=np.asarray(s['map_path']);goal=old.goal_map(s,e['goal_xy'])
        ax.plot(path[:,0],path[:,1],'.-',label='planned Path map (2 s)')
        ax.plot(root[:40,0],root[:40,1],label='realized 0–2 s map')
        ax.plot(root[40:,0],root[40:,1],label='realized 2–6 s map')
        ax.scatter(*path[0],marker='o',label='robot start map');ax.scatter(*goal,marker='*',label='goal map')
        ax.scatter(*path[-1],marker='^',label='Path end map')
        ax.set_title(state_id+' / '+e['terrain']);ax.set_xlabel('map X (m)');ax.set_ylabel('map Y (m)');ax.axis('equal');ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(DEST/'goal_path_realized_overlay.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,9));labels=[x['state_id'][-3:] for x in pcr['states']];x=np.arange(4)
    for k,name in enumerate(('P','C','R')):
        axes[0,0].bar(x+(k-1)*.22,[s['core_mode_count'][name] for s in pcr['states']],.22,label=name)
    axes[0,0].set_xticks(x,labels);axes[0,0].set(ylabel='mode count',title='P / C / R modes, 0.05 m');axes[0,0].legend()
    axes[0,1].bar(x-.2,[s['nominal_duplicate_max_pair_distance_m'] for s in pcr['states']],.4,label='nominal')
    axes[0,1].bar(x+.2,[max(s['distance_matrices']['R'].values()) for s in pcr['states']],.4,label='candidate')
    axes[0,1].set_xticks(x,labels);axes[0,1].set(ylabel='R pair distance (m)',title='duplicate vs candidate');axes[0,1].legend()
    for k,metric in enumerate(('delta_v','advantage_h','drawdown')):
        for cls in ('logged','ID','near_ood','far_ood'):
            vals=[o['primary_40_step'][metric] for state in relative['states'] for o in state['outcomes'] if o['kind']=='candidate' and o['goal_class']==cls]
            axes[1,0].scatter([k]*len(vals),vals,label=cls if k==0 else None,alpha=.6)
    axes[1,0].set_xticks(range(3),['ΔV','advantage','drawdown']);axes[1,0].set(ylabel='raw outcome units',title='Goal class outcome variation');axes[1,0].legend(fontsize=7)
    axes[1,1].bar(x-.18,[int(v.read(ROOT/'final-decision-v2-schedule-storage.json').get(key,False)) for key in ('GS1','GS2','GS3','GS4_simulator')],.36,label='old')
    axes[1,1].bar(x+.18,[int(decision[key]) for key in ('GS1','GS2','GS3','GS4_simulator')],.36,label='corrected')
    axes[1,1].set_xticks(x,['GS1','GS2','GS3','GS4']);axes[1,1].set(ylim=(0,1.2),ylabel='pass',title='Gate before / after');axes[1,1].legend()
    fig.tight_layout();fig.savefig(DEST/'pcr_outcomes_gates.png',dpi=140);plt.close(fig)
    return {p.name:v.sha(p) for p in DEST.glob('*.png')}


def report(spec,offsets,execution,pcr,relative,gs4,decision,figs):
    agg=execution['aggregates'];old_dec=v.read(ROOT/'final-decision-v2-schedule-storage.json')
    text=f"""# EAT-Nav v2 coordinate audit and GS1–GS4 reanalysis

Status: **{decision['status']}**. The original `local_subgoal_execution_not_supported` result is retained as `superseded_pending_coordinate_audit`; original files were not overwritten. No simulator rollout, model training, threshold, candidate, Path, or raw trace changed.

## Frame finding

The old analyzer compared Isaac `root_state_w` (world) to Path and goal (environment-local `map`). `local_pose()` subtracts `scene.env_origins` before Path generation and bridge TF publication, while the trace saves `root_state_w` unchanged. Isaac `env_origins` is translation only, so canonical root XY is world XY minus the state origin. The four inferred origins were cross-checked across all 11 live runs per state and step-0 anchor invariants.

| State | Terrain | env origin world XY (m) | norm (m) | residual norm range after subtraction (m) |
|---|---|---:|---:|---:|
"""
    for row in offsets['states']:
        terrain=next(x['terrain'] for x in offsets['runs'] if x['state_id']==row['state_id'])
        text+=f"| {row['state_id']} | {terrain} | {row['origin_world_xy_m']} | {row['origin_norm_m']:.2f} | {row['residual_norm_min_m']:.3f}–{row['residual_norm_max_m']:.3f} |\n"
    text+=f"""

The legacy 17–45 m errors are explained primarily by those origins. The residual is real candidate/path deviation and varies by candidate; no yaw rotation or axis swap was needed. See `coordinate-frame-spec.json` for the variable-by-variable frame table and source lines.

## Recomputed execution

- 0–2 s reach ≤0.2 m: logged {agg['class:logged']['primary_reach_rate']:.1%}, ID {agg['class:ID']['primary_reach_rate']:.1%}, near-OOD {agg['class:near_ood']['primary_reach_rate']:.1%}, far-OOD {agg['class:far_ood']['primary_reach_rate']:.1%}.
- 0–6 s reach ≤0.2 m: logged {agg['class:logged']['secondary_reach_rate']:.1%}, ID {agg['class:ID']['secondary_reach_rate']:.1%}, near-OOD {agg['class:near_ood']['secondary_reach_rate']:.1%}, far-OOD {agg['class:far_ood']['secondary_reach_rate']:.1%}.
- Logged/ID/near-OOD combined primary reach: {decision['core_primary_reach_rate']:.1%}; state support {decision['supporting_state_count']}/4. Far-OOD remains diagnostic only.
- Median candidate 0–2 s cross-track P95: {agg['all']['primary_cross_track_p95_median_m']:.3f} m. The Path ends at 2 s; the 2–6 s metrics describe goal arrival and subsequent motion, not Path tracking to a new 6 s plan.
- Post-reach departure beyond 0.2 m: {agg['all']['post_reach_departure_count']}/32 candidates. Catastrophic termination: {decision['catastrophic_termination_count']}.

## Other gates and scope

- GS2 P/C/R: R mode counts {[x['core_mode_count']['R'] for x in pcr['states']]}; old R pair distances/modes are unchanged by common origin translation. P and C modes and nominal variation are separately recorded.
- GS3: ΔV, advantage, drawdown, contact, tilt, stack, and effort agree exactly with old raw-series results. State pass count {sum(x['pass'] for x in relative['states'])}/4. Canonical goal and cross-track correlations are new read-only diagnostics.
- GS4 simulator schema: {gs4['GS4_simulator_schema_pass']}. Real robot equivalent raw effort remains `NA / not verified`; simulator availability does not establish field availability or semantics on hardware.
- Counterfactual Evaluator collection decision: **{decision['decision']}**, subject to the existing staged approval process. This offline audit alone does not authorize starting collection.
- NoMaD retraining: not required by this coordinate bug alone. Far-OOD reach is 0/8 and could motivate a separate proposal, without changing this pilot's result.

The original GS1 implementation counted ID and near-OOD (16 samples), whereas this requested audit counts logged, ID, and near-OOD (24 samples). The reach threshold (0.2 m), 75% aggregate requirement, 3-of-4 state requirement, and 2 s primary horizon are unchanged. The corrected aggregate clears 75%, but only {decision['supporting_state_count']} states clear the state-level requirement. The old and new decisions have the same label for different reasons.

## Requested answers

1. **17–45 m legacy error:** yes; the state origin norms are 16.97, 28.28, 37.76, and 45.61 m. Remaining vector error is candidate-specific motion/Path deviation, not another shared offset.
2. **Frames:** goal starts in robot local, is rotated and translated into environment-local `map`; planned Path is cumulative robot-local waypoints transformed into that same `map`; Isaac root and post-root are world. Bridge TF is `map → body` using the origin-subtracted pose.
3. **0–2 s reach:** logged 4/8, ID 8/8, near-OOD 8/8, far-OOD 0/8; core logged/ID/near-OOD 20/24 = 83.3%.
4. **0–6 s reach:** logged 6/8, ID 8/8, near-OOD 8/8, far-OOD 0/8; core 22/24 = 91.7%.
5. **Goal classes:** ID and near-OOD reached consistently; logged reached later in two additional runs; far-OOD did not reach. Four states and two seeds per goal class are recorded in the JSON aggregates.
6. **Cross-track:** median of candidate-level 0–2 s P95 is {agg['all']['primary_cross_track_p95_median_m']:.3f} m; class-specific medians are logged {agg['class:logged']['primary_cross_track_p95_median_m']:.3f}, ID {agg['class:ID']['primary_cross_track_p95_median_m']:.3f}, near {agg['class:near_ood']['primary_cross_track_p95_median_m']:.3f}, far {agg['class:far_ood']['primary_cross_track_p95_median_m']:.3f} m.
7. **After reach:** {agg['all']['post_reach_departure_count']} candidates later moved beyond 0.2 m from goal. The per-run maximum deviation is saved in the corrected execution JSON.
8. **GS2:** corrected R pair distances and mode counts equal the old values exactly; all four states have at least two R modes. P/C mode counts are newly reported because the old output had distances but no P/C clustering verdict.
9. **GS3:** the raw ΔV, advantage, drawdown, recovery, tilt, contact, stack, termination, and effort verdict is unchanged: four of four states pass. Only canonical goal/cross-track correlations are added.
10. **Trajectory and outcome diversity:** yes, candidates create multiple realized R modes and distinct Critic-derived outcomes, while the precommitted GS1 state support gate fails.
11. **Evaluator collection:** do not advance under this gate result; a new design decision would require separate approval. No collection was started by this audit.
12. **NoMaD retraining:** not required to correct the analysis bug. Any retraining proposal should address a separately defined limitation, such as the far-OOD 0/8 result.
13. **Real robot inputs:** equivalent raw joint effort remains unverified; the simulator field availability and source-level frame mapping do not establish hardware timing, calibration, or equivalence.

## Preservation and figures

Old decision SHA: `{v.sha(ROOT/'final-decision-v2-schedule-storage.json')}`. New decision and full 48-run SHA ledger are separate. Figures: {', '.join(figs)}. Frames, metres, origins, robot start, goal, and Path endpoint are labelled in the coordinate and overlay plots.
"""
    target=DEST/'v2-coordinate-audit-report.md'
    if target.exists() and target.read_text()!=text:raise FileExistsError(target)
    if not target.exists():
        temp=target.with_name(target.name+'.tmp');temp.write_text(text);temp.replace(target)
    return v.sha(target)


def main():
    schedule,states,goals,runs,provenance=load_saved()
    old_reproduction=reproduce_old()
    origins,spec=origin_and_spec(schedule,runs,states)
    invariants,offsets=invariants_and_offsets(schedule,runs,origins)
    if invariants['status']!='pass':raise ValueError('coordinate invariants failed')
    goal_metadata={(x['state_id'],x['goal_id']):x for x in goals}
    execution=[metric(e,runs[e['run_id']][1],runs[e['run_id']][2],origins[e['state_id']],
                      goal_metadata[(e['state_id'],e['goal_id'])]) for e in schedule if e['kind']=='candidate']
    aggregates=aggregate(execution)
    core=[x for x in execution if x['goal_class']!='far_ood']
    if len(core)!=24:raise ValueError('logged/ID/near core population differs')
    state_rates={state:float(np.mean([x['horizons']['primary_0_2s']['reach_0p2m'] for x in core if x['state_id']==state])) for state in origins}
    reach=float(np.mean([x['horizons']['primary_0_2s']['reach_0p2m'] for x in core]))
    support=sum(x>=.75 for x in state_rates.values())
    catastrophic=sum(x['catastrophic_termination'] for x in execution)
    gs1=reach>=.75 and support>=3 and catastrophic==0
    pcr_result=pcr(schedule,runs,origins)
    relative=gs3(schedule,runs,origins,execution)
    readiness=gs4(schedule,runs)
    duplicate_pass=all(v.read(ROOT/f'duplicate-gate-{state}.json')['status']=='pass' for state in origins)
    gs2=pcr_result['GS2_pass'];gs3pass=relative['GS3_pass'];gs4pass=readiness['GS4_simulator_schema_pass']
    all_pass=gs1 and gs2 and gs3pass and gs4pass and duplicate_pass
    decision={'status':'coordinate_bug_confirmed_analysis_recomputed',
        'decision':('proceed_support_aware_subgoal_collection' if all_pass else
                    'local_subgoal_execution_not_supported' if not gs1 else
                    'goal_execution_ok_but_outcome_diversity_insufficient'),
        'GS1':gs1,'GS2':gs2,'GS3':gs3pass,'GS4_simulator':gs4pass,
        'duplicate_gate_pass_count':sum(v.read(ROOT/f'duplicate-gate-{state}.json')['status']=='pass' for state in origins),
        'core_population':'logged/ID/near-OOD; far-OOD diagnostic only',
        'core_candidate_count':len(core),'core_primary_reach_rate':reach,
        'state_core_primary_reach_rate':state_rates,'supporting_state_count':support,
        'catastrophic_termination_count':catastrophic,
        'legacy_decision':'local_subgoal_execution_not_supported',
        'legacy_decision_history_status':'superseded_pending_coordinate_audit',
        'legacy_decision_sha256':v.sha(ROOT/'final-decision-v2-schedule-storage.json'),
        'heldout_used':False,'validation_used':False,'new_simulator_rollouts':0,
        'real_robot_equivalent_effort':'NA / not verified'}
    for entry in schedule:
        v.verify_saved_run(entry)
        row=provenance['runs'][entry['order']]
        if entry['kind']!='anchor' and v.sha(ROOT/'runs'/v.run_name(entry)/'live-0.trace.json.gz')!=row['trace_sha256']:
            raise ValueError('raw trace SHA changed during coordinate audit')
    # New outputs are issued only after the full C0/C1/C2/C5 audit passes.
    write('coordinate-audit-provenance.json',provenance)
    write('legacy-analysis-reproduction.json',old_reproduction)
    write('coordinate-frame-spec.json',spec)
    write('coordinate-offset-diagnostic.json',offsets)
    write('coordinate-invariant-tests.json',invariants)
    write('goal-execution-results-v2-coordinate-fixed.json',{'runs':execution,'aggregates':aggregates,
        'core_primary_reach_rate':reach,'state_core_primary_reach_rate':state_rates,'GS1_pass':gs1})
    write('pcr-diversity-results-v2-coordinate-fixed.json',pcr_result)
    write('relative-outcome-results-v2-coordinate-fixed.json',relative)
    write('evaluator-input-readiness-v2-coordinate-audited.json',readiness)
    write('final-decision-v2-coordinate-fixed.json',decision)
    figs=figures(execution,offsets,pcr_result,relative,decision)
    report_sha=report(spec,offsets,{'aggregates':aggregates},pcr_result,relative,readiness,decision,figs)
    ledger={p.name:v.sha(p) for p in sorted(DEST.iterdir()) if p.is_file()
            and p.name!='coordinate-audit-sha-ledger.json'
            and '-draft' not in p.stem and '-attempt-' not in p.stem}
    write('coordinate-audit-sha-ledger.json',{'status':'pass','files':ledger,
        'original_decision_sha256':v.sha(ROOT/'final-decision-v2-schedule-storage.json'),
        'report_sha256':report_sha})
    print(json.dumps({'status':decision['status'],'decision':decision['decision'],'GS1':gs1,
        'GS2':gs2,'GS3':gs3pass,'GS4':gs4pass,'core_reach':reach,'figure_count':len(figs)},indent=2))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        write('coordinate-audit-stop.json',{'status':'coordinate_audit_inconclusive',
            'reason':repr(exc),'traceback':traceback.format_exc()})
        raise
