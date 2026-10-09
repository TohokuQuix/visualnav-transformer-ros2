"""Read-only verification and descriptive analysis of the limited P1 pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from visualnav_transformer.eat_nav.p1_statistical_analysis import analyze, extract_outcome, extract_diagnostic, METRICS
from visualnav_transformer.eat_nav.p1_statistical_freeze import sha
from visualnav_transformer.eat_nav.p1_statistical_runtime import validate_candidate_free_trace


GROUPS=('first-differential','second-differential','first-graded','second-graded')


def verify_candidate_free(root):
    marker=json.loads((root/'output/complete.json').read_text())
    if marker.get('Path_publications')!=0 or marker.get('control_steps')!=120 or len(marker.get('trace_sha256',[]))!=7:
        raise ValueError('candidate-free 120-step contract')
    summaries=[]
    for index,digest in enumerate(marker['trace_sha256']):
        path=root/'output'/f'trace-slot-{index}.jsonl'
        if sha(path)!=digest:
            raise ValueError('candidate-free trace SHA mismatch')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        validate_candidate_free_trace(rows)
        primary=extract_outcome(rows,(.75,0.0),.99)
        summaries.append({'slot':index,'control_steps':len(rows),
                          'primary_window_processed':primary['valid_primary'] or
                              primary['first_terminal_step'] is not None})
    return {'candidate_free':True,'Path_publications':0,'trace_count':7,
            'control_steps_per_trace':120,'analysis_primary_window_accepted':all(
                row['primary_window_processed'] for row in summaries)}


def atomic(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def verify_one(root,group,batch,manifest,base_config):
    seal=json.loads((root/'host-seal.json').read_text())
    marker_path=root/'output/complete.json'
    if seal['schema']!='p1_statistical_pilot_seal_v1' or sha(marker_path)!=seal['complete_sha256']:
        raise ValueError('pilot seal mismatch: '+str(root))
    marker=json.loads(marker_path.read_text())
    freeze=json.loads((root/'input/freeze.json').read_text())
    cfg=json.loads((root/'input/statistical-config.json').read_text())
    if (marker['anchor_id'],marker['batch_index'],marker['Path_publications'],marker['control_steps'])!=(
            group,batch,1,base_config['diagnostic_control_steps']):
        raise ValueError('pilot completion contract mismatch: '+str(root))
    if sha(root/'source.py')!=freeze['pilot_simulator_source_sha256'] or sha(root/'input/statistical-config.json')!=freeze['config_sha256']:
        raise ValueError('source/config SHA drift: '+str(root))
    if sha(root/'input/assignment.json')!=freeze['assignment_sha256'] or sha(root/'input/source-contract.json')!=freeze['source_contract_sha256']:
        raise ValueError('assignment/source contract SHA drift: '+str(root))
    if sha(root/'output/initial-state.json')!=marker['initial_state_sha256'] or sha(root/'output/terrain.json')!=marker['terrain_sha256'] or sha(root/'output/prepath-gate.json')!=marker['prepath_gate_sha256']:
        raise ValueError('initialization SHA mismatch: '+str(root))
    if not json.loads((root/'output/prepath-gate.json').read_text())['pass']:
        raise ValueError('initialization gate failed: '+str(root))
    for field,path in [('path_publication_sha256','path-publication.json'),
                       ('pilot_summary_sha256','pilot-summary.json')]:
        if sha(root/'output'/path)!=marker[field]:
            raise ValueError('pilot artifact SHA mismatch: '+path)
    publication=json.loads((root/'output/path-publication.json').read_text())
    if publication['count']!=1 or not publication['monitor_path_equal_slot_0']:
        raise ValueError('Path publication contract')
    rows=[row for row in manifest['trials'] if row['anchor_id']==group and row['batch_index']==batch]
    paths={row['trial_id']:row['path_sha256'] for row in publication['trial_paths']}
    if paths!={row['trial_id']:row['path_sha256'] for row in rows}:
        raise ValueError('published Path assignment drift')
    outcomes=[]
    for row in rows:
        slot=row['slot'];trace_path=root/'output'/f'trace-slot-{slot}.jsonl'
        if sha(trace_path)!=marker['trace_sha256'][slot]:
            raise ValueError('trace SHA mismatch: '+str(trace_path))
        trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
        if not trace or len(trace)>base_config['diagnostic_control_steps'] or [x['control_step'] for x in trace]!=list(range(len(trace))):
            raise ValueError('trace step continuity')
        if len(trace)<base_config['diagnostic_control_steps'] and trace[-1].get('terminal') is None:
            raise ValueError('short trace without terminal')
        for item in trace:
            if (len(item['actor_input'])!=22 or len(item['actor_pre_noise'])!=22 or
                    len(item['actor_post_noise_pre_delay'])!=22 or
                    len(item['actor_action'])!=6 or len(item['follower_command'])!=3):
                raise ValueError('Actor/command dimension drift')
            if not all(math.isfinite(v) for key in ('actor_input','actor_pre_noise',
                    'actor_post_noise_pre_delay','actor_action','follower_command',
                    'root_state','post_root_state') for v in item[key]):
                raise ValueError('nonfinite pilot trace')
            if not all(math.isfinite(item[key]) for key in ('reward','critic_value')):
                raise ValueError('nonfinite reward/Critic value')
            if item['logical_sim_time_ns']!=publication['stamp_ns']+item['control_step']*50_000_000:
                raise ValueError('trace stamp mismatch')
            if item['policy_action_mode']!='act_inference':
                raise ValueError('policy action mode drift')
        outcome=extract_outcome(trace,tuple(base_config['common_global_goal_base_xy_m']),
                                base_config['reward_gamma_per_step'])
        diagnostic=extract_diagnostic(trace,tuple(base_config['common_global_goal_base_xy_m']))
        outcomes.append({**outcome,'diagnostic':diagnostic,'trial_id':row['trial_id'],'slot':slot,
                         'anchor_id':group,'batch_index':batch,'path_sha256':row['path_sha256'],
                         'path_identity':row['path_identity'],'trace_sha256':sha(trace_path)})
    terrain=json.loads((root/'output/terrain.json').read_text())
    status=json.loads((root/'host-status.json').read_text())
    if status['resource_failure'] is not None or status['native_exit']!=0:
        raise ValueError('resource/native exit mismatch')
    cleanup=json.loads((root/'bridge-cleanup.json').read_text())
    if cleanup['owned_residual']:
        raise ValueError('owned follower residual')
    return outcomes,{'root':str(root),'seal_sha256':sha(root/'host-seal.json'),
                     'freeze_sha256':sha(root/'input/freeze.json'),
                     'terrain_mesh_sha256':terrain['fingerprints'][0]['canonical_mesh_sha256'],
                     'composed_ground_sha256':terrain['collision']['sha256'],
                     'peak_gpu_mib':status['peak_gpu_mib'],
                     'elapsed_s':status['elapsed_s'],
                     'router_before_sha256':sha(root/'router-before.json'),
                     'router_after_sha256':sha(root/'router-after.json')}


def direct_slot_diagnostics(outcomes):
    results=[]
    for group in GROUPS:
        for metric in METRICS:
            direct=[r for r in outcomes if r['anchor_id']==group and r['path_identity']=='direct_reference' and isinstance(r[metric],(int,float))]
            values=[r[metric] for r in direct]
            results.append({'anchor_id':group,'metric':metric,'count':len(values),
                            'mean':statistics.mean(values) if values else None,
                            'sd':statistics.stdev(values) if len(values)>1 else None,
                            'by_batch':[{'batch_index':b,'slots':[r['slot'] for r in direct if r['batch_index']==b],
                                         'values':[r[metric] for r in direct if r['batch_index']==b]}
                                        for b in (0,1)],
                            'slot_effect_estimable':False,
                            'reason':'two randomized blocks and sparse direct repeats'})
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-root',type=Path)
    parser.add_argument('--candidate-free-root',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.candidate_free_root is not None:
        if args.pilot_root is not None or args.output is not None:
            parser.error('candidate-free validation takes only --candidate-free-root')
        print(json.dumps(verify_candidate_free(args.candidate_free_root),sort_keys=True))
        return
    if args.pilot_root is None or args.output is None:
        parser.error('pilot analysis requires --pilot-root and --output')
    if args.output.exists():raise FileExistsError(args.output)
    manifest=json.loads((args.pilot_root/'pilot-first-differential-b0-v1/input/assignment.json').read_text())
    config=json.loads((args.pilot_root/'pilot-first-differential-b0-v1/input/statistical-config.json').read_text())
    if config['primary_control_steps']!=40 or config['diagnostic_control_steps']!=120:
        raise ValueError('protocol-invalid diagnostic window; cannot analyze as a complete P1 pilot')
    args.output.mkdir(parents=True)
    outcomes=[];ledger=[]
    for group in GROUPS:
        fingerprints=[]
        for batch in (0,1):
            root=args.pilot_root/f'pilot-{group}-b{batch}-v1'
            rows,evidence=verify_one(root,group,batch,manifest,config)
            outcomes.extend(rows);ledger.append(evidence)
            fingerprints.append((evidence['terrain_mesh_sha256'],evidence['composed_ground_sha256']))
        if fingerprints[0]!=fingerprints[1]:
            raise ValueError('cross-batch mesh/ground collision mismatch: '+group)
    result=analyze(outcomes,manifest,config)
    result['slot_diagnostics']=direct_slot_diagnostics(outcomes)
    result['remaining_confounders']=[
        'hidden PhysX contact warm start and sensor history were not directly cloned',
        'global observation/simulator RNG streams are not paired per environment',
        'two independent batches cannot support confirmatory confidence intervals or reliable tile-effect estimation',
    ]
    atomic(args.output/'outcomes.json',outcomes)
    atomic(args.output/'analysis.json',result)
    atomic(args.output/'run-ledger.json',ledger)
    seal={'schema':'p1_statistical_analysis_seal_v1',
          'assignment_sha256':sha(args.pilot_root/'pilot-first-differential-b0-v1/input/assignment.json'),
          'analysis_source_sha256':sha(Path(__file__).resolve()),
          'analysis_module_sha256':sha(Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/p1_statistical_analysis.py'),
          'outcomes_sha256':sha(args.output/'outcomes.json'),
          'analysis_sha256':sha(args.output/'analysis.json'),
          'run_ledger_sha256':sha(args.output/'run-ledger.json')}
    atomic(args.output/'seal.json',seal)
    print(json.dumps({'trial_count':len(outcomes),'analysis_root':str(args.output),'seal_sha256':sha(args.output/'seal.json')}))


if __name__=='__main__':main()
