"""Read-only paired analysis of three separately sealed ROS-free crossover periods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_g3_crossover_v2_analyze import _trial
from eat_nav_p1_g3_rosfree_crossover_analyze_v2 import add_display_mode
from visualnav_transformer.eat_nav.p1_g3_crossover_runtime import paired_effects
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_evidence import validate_output


SCHEMA='eatnav_p1_g3_rosfree_crossover_v1'
ASSIGNMENT_SHA='8c910f1204502cdf749c2d265887c3b29026bcdef4647674b8548d49a6277d29'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def motion_series(rows, arrival_step):
    """Keep heading-limited nominal command separate from executed body motion."""
    result=[]
    for row in rows:
        step=row['control_step']
        heading=float(row['follower_diagnostic']['heading_error'])
        cap=(.3*max(0.,1.-abs(heading)/math.pi)
             if abs(heading)<.6 and not row['follower_diagnostic']['goal_reached'] else 0.)
        yaw=float(row['core_evidence']['pose'][2])
        root=row['root_state']
        actual=float(root[7])*math.cos(yaw)+float(root[8])*math.sin(yaw)
        record={'step':step,'logical_sim_time_ns':row['logical_sim_time_ns'],
                'phase':'after_arrival' if arrival_step is not None and step>=arrival_step
                        else 'pre_arrival',
                'heading_error_rad':heading,
                'nominal_heading_forward_cap_mps':cap,
                'follower_forward_mps':float(row['follower_command'][0]),
                'injected_forward_mps':float(row['applied_command'][0]),
                'actual_forward_mps':actual,
                'follower_yaw_rate_radps':float(row['follower_command'][2]),
                'root_angular_z_radps':float(root[12]),
                'linear_recovery':bool(row['follower_diagnostic']['linear_recovery']),
                'angular_recovery':bool(row['follower_diagnostic']['angular_recovery'])}
        if not all(math.isfinite(v) for v in record.values() if isinstance(v,float)):
            raise ValueError('nonfinite follower/body time series')
        result.append(record)
    return result


def analyze(roots):
    if len(roots)!=3:
        raise ValueError('exactly three distinct period roots required')
    roots=[Path(root) for root in roots]
    if len(set(map(str,roots)))!=3:
        raise ValueError('period roots reused')
    trials=[];series=[];periods=[];snapshots=[];shared=[];sim_sources=[];cores=[]
    for period,root in enumerate(roots):
        batch=root/f'batch-first-differential-0-period-{period}'
        manifest=json.loads((root/'manifest.json').read_text())
        assignment=json.loads((root/'g3-assignment.json').read_text())
        if (manifest['schema']!=SCHEMA or manifest['assignment_sha256']!=ASSIGNMENT_SHA or
                sha(root/'g3-assignment.json')!=ASSIGNMENT_SHA or
                manifest['authorized_process_ids']!=[batch.name] or
                manifest.get('authorized_period_index',0)!=period):
            raise ValueError('period manifest/assignment mismatch')
        seal=verify_seal(batch)
        if seal['stage']!='G3_rosfree_crossover_protocol_valid':
            raise ValueError('period seal not protocol-valid')
        status=json.loads((batch/'rosfree-host-status.json').read_text())
        cleanup=json.loads((batch/'cleanup-evidence.json').read_text())
        if (status['failure'] is not None or status['native_exit']!=0 or
                cleanup['owned_residual'] or cleanup['native_exit']!=0):
            raise ValueError('period host/cleanup invalid')
        output=batch/'output'
        assigned=assignment['processes'][period]['trials']
        validation=validate_output(output,manifest,assigned)
        cross=json.loads((output/'cross-period-prepath-gate.json').read_text())
        if not cross['pass'] or (period and
                (not cross['cross_period_equality_assessed'] or
                 cross['comparison']['strict_failures'])):
            raise ValueError('cross-period pre-Path equality invalid')
        snapshots.append(sha(output/'prepath-crossover-snapshot.json'))
        shared.append(json.loads((output/'prepath-shared-dem-gate.json').read_text())['source_frame_sha256'])
        sim_sources.append(manifest['g3_simulator_source_sha256'])
        cores.append(manifest['cpp_core_source_sha256'])
        publication=json.loads((output/'path-activation.json').read_text())['trial_paths']
        summary=json.loads((output/'pilot-summary.json').read_text())
        terrain=json.loads((output/'terrain.json').read_text())
        resources=[json.loads(line) for line in (batch/'resource-series.jsonl').read_text().splitlines()]
        for env in range(4):
            rows=[json.loads(line) for line in (output/f'trace-slot-{env}.jsonl').read_text().splitlines()]
            identity={'anchor_id':'first-differential','state_group':'first-differential',
                      'batch_seed':2026101101,'env_id':env,'execution_index':period,
                      'assignment_period':period,'path_sha256':assigned[env]['path_sha256']}
            trial=_trial(add_display_mode(rows),summary['arrivals'][env],publication[env],
                         terrain['world_tile_origins'][env],identity)
            trials.append(trial)
            series.append({'period':period,'env_id':env,'path_role':trial['path_role'],
                           'rows':motion_series(rows,summary['arrivals'][env]['arrival_step'])})
        periods.append({'period':period,'root':str(root),'manifest_sha256':sha(root/'manifest.json'),
                        'host_seal_sha256':sha(batch/'host-seal.json'),
                        'host_source_sha256':manifest['g3_host_source_sha256'],
                        'validation':validation,'cross_period_prepath':cross,
                        'native_exit':status['native_exit'],
                        'owned_residual_count':len(cleanup['owned_residual']),
                        'wall_s':status['elapsed_s'],
                        'gpu_peak_mib':max(row['gpu_device_total_mib'] for row in resources),
                        'ram_peak_bytes':max(row['container_anon_plus_shmem_bytes'] for row in resources),
                        'storage_peak_bytes':max(row['persistent_root_bytes'] for row in resources)})
    if len(set(snapshots))!=1 or len(set(sim_sources))!=1 or len(set(cores))!=1 or \
            any(item!=shared[0] for item in shared):
        raise ValueError('strict pre-Path/DEM/simulator/follower provenance differs')
    if len(trials)!=12 or any({trial['path_role'] for trial in trials if trial['env_id']==env}!=
                              {'direct','left_20','right_20'} for env in range(4)):
        raise ValueError('incomplete balanced crossover')
    return {'schema':SCHEMA+'_three_period_analysis_v1',
            'analyzer_sha256':sha(__file__),'assignment_sha256':ASSIGNMENT_SHA,
            'independent_state_group_seed_conditions':1,
            'interpretation':'descriptive paired comparison only; no significance or generalization',
            'periods':periods,'prepath_snapshot_sha256':snapshots[0],
            'shared_dem_source_sha256':shared[0],
            'simulator_source_sha256':sim_sources[0],
            'follower_core_sha256':cores[0],
            'trials':trials,'paired':paired_effects(trials),
            'heading_motion_timeseries':series}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--period-root',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.period_root)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(args.output)


if __name__=='__main__':main()
