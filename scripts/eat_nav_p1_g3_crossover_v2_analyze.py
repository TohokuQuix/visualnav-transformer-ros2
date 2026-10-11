"""Offline paired Path-effect analysis of sealed G3 crossover batches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from eat_nav_p1_g1_lifecycle import verify_seal
from visualnav_transformer.eat_nav.p1_g3_material_gate_v2 import SCHEMA
from visualnav_transformer.eat_nav.p1_g3_crossover_runtime import paired_effects
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import extract_trial


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixed_two_second_load(rows: list[dict]) -> dict:
    if len(rows)<40:
        raise ValueError('fixed two-second exposure incomplete')
    forces=[];torques=[]
    for row in rows[:40]:
        vectors=row['contact_net_forces_w']
        torque=row['applied_torque']
        if not vectors or not torque:
            raise ValueError('fixed two-second body load missing')
        forces.append(statistics.mean(math.sqrt(sum(x*x for x in vector))
                                      for vector in vectors))
        torques.append(statistics.mean(abs(x) for x in torque))
    return {'contact_force_fixed_2s_mean_N':statistics.mean(forces),
            'joint_torque_fixed_2s_mean_abs':statistics.mean(torques),
            'fixed_window_steps':40}


def _path_geometry(points: list[list[float]]) -> dict:
    lengths=[math.dist(a,b) for a,b in zip(points,points[1:])]
    if not lengths or any(length<=0 for length in lengths):
        raise ValueError('invalid Path segment length')
    curvature=[]
    for a,b,c in zip(points,points[1:],points[2:]):
        ac=math.dist(a,c)
        if ac<=0:raise ValueError('invalid Path curvature baseline')
        cross=(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
        curvature.append(2*cross/(math.dist(a,b)*math.dist(b,c)*ac))
    return {'path_length_m':sum(lengths),
            'path_max_abs_discrete_curvature_m_inv':max(map(abs,curvature),default=0.0)}


def _trial(rows: list[dict], arrival: dict, publication: dict,
           origin: list[float], identity: dict) -> dict:
    local=extract_trial(rows,arrival,publication['map_xy'],origin,identity)
    common=publication['common_goal_map_xy']
    def distance(state):
        return math.hypot(state[0]-origin[0]-common[0],
                          state[1]-origin[1]-common[1])
    local['common_goal_progress_2s_m']=(distance(rows[0]['root_state'])-
                                         distance(rows[39]['post_root_state']))
    local['local_endpoint_progress_2s_m']=local['two_second_progress_m']
    local['path_role']=publication['path_role']
    local['local_endpoint_map_xy_m']=publication['map_xy'][-1]
    local['common_goal_map_xy_m']=common
    local['endpoint_common_goal_distance_m']=math.dist(publication['map_xy'][-1],common)
    local.update(_path_geometry(publication['map_xy']))
    local.update(_fixed_two_second_load(rows))
    return local


def analyze(root: Path, manifest: dict) -> dict:
    assignment=json.loads((root/'g3-assignment.json').read_text())
    if sha(root/'g3-assignment.json')!=manifest['assignment_sha256'] or \
            len(assignment['processes'])!=18:
        raise ValueError('frozen crossover assignment drift')
    trials=[];seals={}
    pilot_ids=set(manifest['pilot_process_ids'])
    for process in assignment['processes']:
        name=(f"batch-{process['anchor_id']}-{process['batch_index']}-"
              f"period-{process['execution_index']}")
        if name not in pilot_ids:
            continue
        batch=root/name
        seal=verify_seal(batch)
        if seal['stage']!='G3_crossover_batch_protocol_valid':
            raise ValueError('batch not protocol-valid: '+name)
        seals[name]=sha(batch/'host-seal.json')
        output=batch/'output'
        complete=json.loads((output/'complete.json').read_text())
        summary=json.loads((output/'pilot-summary.json').read_text())
        publication=json.loads((output/'path-publication.json').read_text())
        terrain=json.loads((output/'terrain.json').read_text())
        if complete['protocol_version']!=SCHEMA or complete['Path_publications']!=1 or \
                len(publication['trial_paths'])!=4 or len(summary['arrivals'])!=4:
            raise ValueError('crossover completion invalid: '+name)
        for slot in range(4):
            path=output/f'trace-slot-{slot}.jsonl'
            if sha(path)!=complete['trace_sha256'][slot]:
                raise ValueError('trace SHA drift: '+name)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            declared=publication['trial_paths'][slot]
            assigned=process['trials'][slot]
            if (declared['path_sha256']!=assigned['path_sha256'] or
                    declared['path_role']!=assigned['path_role'] or
                    len(rows)!=complete['control_steps']):
                raise ValueError('Path/trace assignment drift: '+name)
            identity={'anchor_id':process['anchor_id'],'batch_seed':process['batch_seed'],
                      'env_id':slot,'execution_index':process['execution_index'],
                      'assignment_period':process['assignment_period'],
                      'path_sha256':assigned['path_sha256']}
            trials.append(_trial(rows,summary['arrivals'][slot],declared,
                                 terrain['world_tile_origins'][slot],identity))
    if len(trials)!=12 or len(seals)!=3:
        raise ValueError('3-process/12-trial pilot completeness')
    return {'schema':SCHEMA+'_aggregate_v1',
            'independent_unit':'StateGroup × seed bundle; env rows nested within seed',
            'protocol_valid_batches':3,'trial_count':12,
            'manifest_sha256':sha(root/'manifest.json'),
            'batch_seals':seals,'paired':paired_effects(trials),
            'trials':trials,
            'interpretation':'Path-induced local execution response; navigation quality not inferred from local arrival alone'}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest=json.loads((args.root/'manifest.json').read_text())
    if manifest['schema']!=SCHEMA or sha(Path(__file__))!=manifest['analysis_source_sha256']:
        raise ValueError('analysis source/contract SHA drift')
    result=analyze(args.root,manifest)
    from eat_nav_p1_g3_direct_repeat_stage import durable_json
    args.output.parent.mkdir(parents=True,exist_ok=True)
    durable_json(args.output,result)


if __name__=='__main__':
    main()
