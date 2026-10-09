"""Read-only aggregate for sealed P1 minimal candidate-free physics probes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.p1_batch_golden import verify_backup
from visualnav_transformer.eat_nav.p1_prepath_divergence import pair_metrics, gate_checks


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def verify_run(root):
    root = Path(root)
    seal, complete = load(root/'host-seal.json'), load(root/'output/complete.json')
    if sha(root/'manifest.json') != seal['manifest_sha256'] or sha(root/'output/complete.json') != seal['complete_sha256']:
        raise ValueError(f'host seal mismatch: {root}')
    for name, key in (('physics-trace.json','trace_sha256'), ('actor-observations.json','actors_sha256'),
                      ('dem-frames.json','dem_sha256')):
        if sha(root/'output'/name) != complete[key]:
            raise ValueError(f'output SHA mismatch: {root}/{name}')
    trace = load(root/'output/physics-trace.json')
    actor = load(root/'output/actor-observations.json')
    dem = load(root/'output/dem-frames.json')
    terrain = load(root/'output/terrain.json')
    manifest = load(root/'manifest.json')
    if (len(trace) != 200 or len(actor) != 20 or len(dem) != 4 or
            [x['physics_step'] for x in trace] != list(range(1,201)) or
            [x['physics_step'] for x in dem] != [50,100,150,200] or
            not all(y['finite'] for row in actor for y in row['post_delay_actor_input']) or
            not all(y['finite'] for row in dem for y in row['dem']) or
            complete['Path_publications'] != 0):
        raise ValueError(f'protocol incomplete: {root}')
    canonical = [x['tile_local_mesh_sha256'] for x in terrain['canonical_tiles']]
    if len(canonical) != 3 or len(set(canonical)) != 1:
        raise ValueError(f'tile geometry mismatch: {root}')
    return {'root': root, 'seal': seal, 'manifest': manifest, 'trace': trace,
            'actor': actor, 'dem': dem, 'terrain': terrain,
            'after_reset': load(root/'output/after-reset.json'),
            'before_first': load(root/'output/before-first-physics.json')}


def record(run, i):
    return {'root_state_w': run['trace'][-1]['states'][i]['root_state_w'],
            'target_tile_origin_xy': run['terrain']['world_tile_origins'][i][:2],
            'joint_position': run['trace'][-1]['states'][i]['joint_position'],
            'joint_velocity': run['trace'][-1]['states'][i]['joint_velocity'],
            'actor_input': run['actor'][-1]['post_delay_actor_input'][i]['values'],
            'dem_frames': [x['dem'][i] for x in run['dem']]}


def first_difference(series_a, series_b, extractor, tolerance):
    first_nonzero = first_gate = None
    maximum = 0.0
    for step, (a,b) in enumerate(zip(series_a, series_b), 1):
        x,y = np.asarray(extractor(a, 0), dtype=float), np.asarray(extractor(b, 1), dtype=float)
        diff = float(np.max(np.abs(x-y)))
        maximum = max(maximum, diff)
        if diff > 0 and first_nonzero is None:
            first_nonzero = {'step': step, 'max_abs': diff}
        if diff > tolerance and first_gate is None:
            first_gate = {'step': step, 'max_abs': diff}
    return {'first_nonzero': first_nonzero, 'first_frozen_threshold_crossing': first_gate,
            'max_abs_over_200': maximum}


def local_root_position(state, origin_xy):
    xyz = np.asarray(state['root_state_w'][:3], dtype=float).copy()
    xyz[:2] -= np.asarray(origin_xy, dtype=float)[:2]
    return xyz


def pair_timeline(run, pair, tolerance):
    a, b = pair
    origins = np.asarray(run['terrain']['world_tile_origins'])
    rows_a = [r['states'][a] for r in run['trace']]
    rows_b = [r['states'][b] for r in run['trace']]
    specs = {
        'root_local_position_m': (lambda s, index: local_root_position(s, origins[(a,b)[index]]),
                                  tolerance['root_position_m']),
        'root_velocity': (lambda s, index: s['root_state_w'][7:13], tolerance['root_velocity_mps_or_radps']),
        'joint_position': (lambda s, index: s['joint_position'], tolerance['joint_position_rad']),
        'joint_velocity': (lambda s, index: s['joint_velocity'], tolerance['joint_velocity_radps']),
        'joint_position_target': (lambda s, index: s['joint_position_target'], tolerance['joint_position_rad']),
        'applied_torque_diagnostic': (lambda s, index: s['applied_torque'], 1e-4),
        'command': (lambda s, index: s['command'], 1e-6),
    }
    return {name: first_difference(rows_a, rows_b, fn, bound) for name,(fn,bound) in specs.items()}


def golden_timeline(gold_trace, run, env, tolerance, source_origin):
    """Compare physical state with the historical source at each settling step."""
    target_origin = np.asarray(run['terrain']['world_tile_origins'][env])
    source_origin = np.asarray(source_origin)
    target_trace = [row['states'][env] for row in run['trace']]
    specs = {
        'root_local_position_m': (
            lambda state, side: local_root_position(
                state, source_origin if side == 0 else target_origin),
            tolerance['root_position_m']),
        'root_velocity': (lambda state, side: state['root_state_w'][7:13],
                          tolerance['root_velocity_mps_or_radps']),
        'joint_position': (lambda state, side: state['joint_position'],
                           tolerance['joint_position_rad']),
        'joint_velocity': (lambda state, side: state['joint_velocity'],
                           tolerance['joint_velocity_radps']),
    }
    return {name: first_difference(gold_trace, target_trace, extract, bound)
            for name, (extract, bound) in specs.items()}


def main(args):
    golden = Path(args.golden)
    verified = verify_backup(golden)
    freeze = load(args.tolerance)
    tol = freeze['pre_path_tolerance']
    runs = {name: verify_run(Path(path)) for name,path in
            (('baseline', args.baseline), ('R2_flipper',args.r2),
             ('R3_contact',args.r3), ('R4_global_shift',args.r4))}
    source_run = golden/'runs/p1-01-second-differential-b+00-r00/run'
    pose = 'b3773891777-k_rail_step_field-r0.60-a00'
    gold_trace = [json.loads(line)['state'] for line in (source_run/'trace'/f'{pose}.jsonl').read_text().splitlines()]
    gold_state = gold_trace[-1]
    gold_actor = load(source_run/'actor'/pose/'settled_step_200.json')['values']
    gold_record = {'root_state_w': gold_state['root_state_w'],
                   'target_tile_origin_xy': [-60.150003051757814,24.149999618530273],
                   'joint_position': gold_state['joint_position'],
                   'joint_velocity': gold_state['joint_velocity'],
                   'actor_input': gold_actor,
                   'dem_frames': [load(source_run/'dem'/pose/f'frame-{i:02d}.json') for i in range(4)]}
    results = {}
    for name, run in runs.items():
        d1 = []
        for i in range(3):
            metrics = pair_metrics(gold_record, record(run,i))
            d1.append({'env':i, 'metrics':metrics, 'frozen_criteria_checks':gate_checks(metrics,tol),
                       'historical_physics_timeline': golden_timeline(
                           gold_trace, run, i, tol, gold_record['target_tile_origin_xy'])})
        d2_metrics = pair_metrics(record(run,1),record(run,2))
        results[name] = {
            'manifest_sha256': sha(run['root']/'manifest.json'),
            'host_seal_sha256': sha(run['root']/'host-seal.json'),
            'terrain_mesh_sha256': run['terrain']['canonical_tiles'][0]['tile_local_mesh_sha256'],
            'world_tile_origins': run['terrain']['world_tile_origins'],
            'D1_golden_diagnostic': d1,
            'D2_direct_pair': {'metrics':d2_metrics,'frozen_criteria_checks':gate_checks(d2_metrics,tol)},
            'pair_0_1_timeline': pair_timeline(run,(0,1),tol),
            'pair_1_2_timeline': pair_timeline(run,(1,2),tol),
            'initial_states': run['before_first'],
            'resource': load(run['root']/'host-status.json'),
        }
    r3 = runs['R3_contact']['trace']
    r4 = runs['R4_global_shift']['trace']
    contact = {}
    for name, trace in (('R3_contact',r3),('R4_global_shift',r4)):
        contact[name] = []
        for i in range(3):
            count_step = next((r['physics_step'] for r in trace if r['states'][i]['physx_contact']['contact_count']>0),None)
            force_step = next((r['physics_step'] for r in trace if r['states'][i]['physx_contact']['force_max_n']>0),None)
            contact[name].append({'env':i,'first_contact_candidate_step':count_step,
                                  'first_nonzero_force_step':force_step,
                                  'step_38':trace[37]['states'][i]['physx_contact']})
    exact_r2_r3 = all(runs['R2_flipper']['trace'][k]['states'][i]['root_state_w'] == r3[k]['states'][i]['root_state_w']
                      for k in range(200) for i in range(3))
    shift = []
    for i in range(3):
        root_velocity = next(({'step':k+1,'max_abs':float(np.max(np.abs(
            np.asarray(r3[k]['states'][i]['root_state_w'][7:]) -
            np.asarray(r4[k]['states'][i]['root_state_w'][7:]))))}
            for k in range(200) if np.max(np.abs(
            np.asarray(r3[k]['states'][i]['root_state_w'][7:]) -
            np.asarray(r4[k]['states'][i]['root_state_w'][7:]))) > tol['root_velocity_mps_or_radps']),None)
        shift.append({'env':i,'first_root_velocity_frozen_threshold_crossing':root_velocity})
    out = {'schema':'p1_minimal_prepath_diagnosis_v1','golden_seal_sha256':verified['backup_seal_sha256'],
           'frozen_tolerance_sha256':sha(args.tolerance), 'runs':results,
           'R1':'not_run_baseline_step_0_root_z_and_velocity_match_source_fixture',
           'R3_contact_timeline':contact,'R2_to_R3_root_trace_exact_all_200':exact_r2_r3,
           'R4_global_shift_sensitivity':shift,
           'limitations':['3-env scene and tile placement differ from historical 112-env',
                          'historical source tile collision mesh was not stored per tile',
                          'Actor pre-noise and pre-delay observations were not captured',
                          'per-env latency and some actuator/solver internals remain different or unobservable'],
           'Path_publications':0,'mixed_candidate_rollouts':0}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'analysis.json').write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--golden',type=Path,required=True)
    p.add_argument('--tolerance',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--r2',type=Path,required=True)
    p.add_argument('--r3',type=Path,required=True)
    p.add_argument('--r4',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
