"""Verify frozen passive evidence and publish new, sealed v2 teacher artifacts."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from .fall_stall_targets import (EVENT_NAMES, HORIZONS, SCHEMA, build_targets,
                                 finite_vector, root_yaw)
from .long_horizon_labels import label_trace


REPO = Path(__file__).resolve().parents[3]
SPEC = REPO/'docs/eat-nav/fall-stall-learning-spec-v0.1.md'
FROZEN_SOURCES = {
    'labels': 'src/visualnav_transformer/eat_nav/long_horizon_labels.py',
    'host_runner': 'scripts/eat_nav_motion_14_collection_host.py',
    'sim_runner': 'scripts/eat_nav_motion_14_collection_sim.py',
    'follower': 'src/visualnav_transformer/eat_nav/p1_follower_core.cpp',
    'config': 'config/eat_nav.yaml',
}
FROZEN_IDENTITIES = {
    'actor': 'b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d',
    'actor_env': '14fc2441d928a92b1ba45c06ccd8603eb0cf71aebb5d216efc4005f09696545c',
    'protocol': '1df876cc86e749dbff56b9b81ab1fd10e2ad69790f576bbda08db0c82844d9a7',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')


def require(condition,message):
    if not condition:
        raise ValueError(message)


def verify_ledger(root,ledger_name,seal_name):
    """Verify both membership and bytes; reject symlinks outside source identity."""
    require(not root.is_symlink(),'symlink at sealed evidence root')
    ledger,seal = read(root/ledger_name),read(root/seal_name)
    require(sha(root/ledger_name)==seal['ledger_sha256'],'SHA ledger seal mismatch')
    files={}
    for path in root.rglob('*'):
        require(not path.is_symlink(),'symlink in sealed evidence')
        if path.is_file() and str(path.relative_to(root)) not in (ledger_name,seal_name):
            files[str(path.relative_to(root))]=sha(path)
    require(files==ledger and len(files)==seal['file_count'],'file SHA ledger membership mismatch')
    return files


def verify_derived(root):
    root=Path(root)
    return verify_ledger(root,'sha-ledger.json','derived-seal.json')


def seal_derived(root):
    files={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
    write(root/'sha-ledger.json',files)
    write(root/'derived-seal.json',{'ledger_sha256':sha(root/'sha-ledger.json'),'file_count':len(files)})
    verify_derived(root)


def legacy_initial(saved):
    """Aggregate v1 events at the same initial (0,H] decision as v2.

    The saved per-row v1 future labels omit row 0 and lack a 10 s head;
    they cannot be compared directly to the approved initial-decision problem.
    """
    result={}
    for horizon in HORIZONS:
        result[horizon]={}
        for name in EVENT_NAMES:
            occurred=any(row['events_at_step'][name] for row in saved['labels'][:horizon*20])
            result[horizon][name]=1 if occurred else 0 if saved['window_count']>=horizon*20 else None
    return result


def compare_labels(saved,v2):
    old=legacy_initial(saved)
    event_changes,cycle_changes,horizon_changes=[],[],[]
    for name in EVENT_NAMES:
        before=[row['control_step'] for row in saved['labels'] if row['events_at_step'][name]]
        after=[event['event_step'] for event in v2['events'] if event['event']==name]
        if before!=after:
            onsets=[event['onset_step'] for event in v2['events'] if event['event']==name]
            kind='confirmation_time_shift' if name=='near_fall' and before==onsets else 'measurement_or_event_change'
            event_changes.append({'event':name,'kind':kind,'v1_steps':before,'v2_steps':after})
    for cycle in v2['cycles']:
        for name in ('linear_stall','turn_stall'):
            before=int(any(row['events_at_step'][name] for row in
                           saved['labels'][cycle['start_step']:cycle['end_step']+1]))
            after=cycle[name]['label']
            if before!=after:
                cycle_changes.append({'event':name,'cycle':cycle['cycle'],'v1_label':before,
                                      'v2_label':after,'reason':cycle[name]['reason'],
                                      'evidence':cycle})
    for horizon in HORIZONS:
        for name in EVENT_NAMES:
            after=v2['horizons'][f'{horizon}s']['events'][name]['label']
            if old[horizon][name]!=after:
                horizon_changes.append({'horizon_sec':horizon,'event':name,
                                        'v1_label':old[horizon][name],'v2_label':after,
                                        'reason':v2['horizons'][f'{horizon}s']['events'][name]['reason']})
    return {'event_changes':event_changes,'cycle_changes':cycle_changes,'horizon_changes':horizon_changes}


def compare_cycle_measurements(rows,v2):
    comparisons=[]
    for cycle in v2['cycles']:
        first=rows[cycle['start_step']].get('follower_diagnostic') or {}
        last=rows[cycle['end_step']].get('follower_diagnostic') or {}
        command,actual=last.get('commanded_distance'),last.get('actual_distance')
        remaining=None if 'remaining_path_distance' not in first or 'remaining_path_distance' not in last else \
                  first['remaining_path_distance']-last['remaining_path_distance']
        best=None if 'best_remaining_path_distance' not in first or 'best_remaining_path_distance' not in last else \
             first['best_remaining_path_distance']-last['best_remaining_path_distance']
        valid=cycle['complete'] and all(isinstance(v,(int,float)) and math.isfinite(v)
                                        for v in (command,actual,best))
        coarse=int(command>=.20 and actual/command<=.25 and best<=.05) if valid else None
        comparisons.append({'cycle':cycle['cycle'],'v1_diagnostic_commanded_m':command,
                            'v1_diagnostic_actual_m':actual,'v1_remaining_improvement_m':remaining,
                            'diagnostic_best_improvement_m':best,'diagnostic_best_rule_label':coarse,
                            'v2':cycle})
    return comparisons


def verify_assignment(root,pilot,assignment):
    require(pilot['schema']=='eatnav_passive_paired_long_horizon_pilot_v0.2','unexpected pilot schema')
    for name,digest in FROZEN_IDENTITIES.items():
        require(pilot['source_sha256'].get(name)==digest,f'frozen specification {name} SHA mismatch')
    for key,value in [('control_hz',20),('control_steps',200),('replan_every_steps',40),
                      ('num_envs_per_process',4),('failure_induction',False)]:
        require(pilot.get(key)==value,f'pilot {key} protocol mismatch')
    status=read(root/'status.json')
    require(status.get('state')=='completed' and status['completed_processes']==len(assignment)
            and status['total_processes']==len(assignment)==pilot['process_count'],'pilot is not complete')
    groups=pilot['state_groups'];bank=pilot['candidates']
    require(len({g['terrain_seed'] for g in groups})==len(groups),'duplicate base seed')
    require(len({g['state_group_id'] for g in groups})==len(groups),'duplicate state group')
    require([c['candidate_id'] for c in bank]==['forward_direct','forward_-30','forward_+30','reverse_direct'],
            'candidate bank/order mismatch')
    require(len(assignment)==len(groups)*5,'assignment process count mismatch')
    require(pilot['candidate_process_count']==len(groups)*4 and pilot['candidate_trajectory_count']==len(groups)*16,
            'candidate manifest count mismatch')
    for j,group in enumerate(groups):
        expected=[{'stage':'prepath','candidate_id':None,'direction':1,'angle':0}]
        expected += [{'stage':'candidate',**c} for c in bank]
        for row,slot in zip(assignment[j*5:j*5+5],expected):
            for key in ('state_group_id','terrain_type','terrain_level','terrain_seed'):
                require(row.get(key)==group[key],f'assignment {key} mismatch')
            for key in ('stage','candidate_id','direction','angle'):
                require(row.get(key)==slot[key],f'assignment {key} mismatch')
            require(row.get('failure_induction') is False,'assignment is not passive')
    for name,relative in FROZEN_SOURCES.items():
        require(sha(REPO/relative)==pilot['source_sha256'][name],f'frozen {name} source SHA mismatch')


def verify_attempt(attempt,assignment,pilot):
    outer=verify_ledger(attempt,'outer-sha-ledger.json','outer-seal.json')
    sim=attempt/'sim-output'
    verify_ledger(sim,'sha-ledger.json','host-seal.json')
    host=read(attempt/'manifest.json');info=read(sim/'input-manifest.json');complete=read(sim/'complete.json')
    require(read(attempt/'decision.json').get('pass') is True,'host decision failed')
    require(read(attempt/'verification.json')['simulator_seal_sha256']==sha(sim/'host-seal.json'),
            'verification inner seal identity mismatch')
    require(read(sim/'prepath-pair-gate.json').get('pass') is True,'prepath gate failed')
    stage='candidate' if assignment['stage']=='candidate' else 'prepath_only'
    expected={'seed':assignment['terrain_seed'],'terrain_type':assignment['terrain_type'],
              'terrain_level':assignment['terrain_level'],'direction':assignment['direction'],
              'condition':pilot['condition'],'repeat':pilot['repeat'],'mode':stage,
              'control_steps':200,'replan_every_steps':40}
    for key,value in expected.items():
        require(host.get(key)==value and info.get(key)==value,f'assignment/manifest {key} identity mismatch')
    require(host.get('angle_deg')==info.get('arc_angle_deg')==assignment['angle'],'angle identity mismatch')
    for host_key,sim_key,frozen in [('actor_env_sha256','actor_env_sha256','actor_env'),
                                   ('rl_checkpoint_sha256','checkpoint_sha256','actor'),
                                   ('protocol_sha256','protocol_sha256','protocol'),
                                   ('source_sha256','source_sha256','sim_runner'),
                                   ('eat_config_sha256','eat_config_sha256','config')]:
        require(host.get(host_key)==info.get(sim_key)==pilot['source_sha256'][frozen],f'{frozen} SHA identity mismatch')
    require(host.get('host_source_sha256')==pilot['source_sha256']['host_runner'],'host source identity mismatch')
    require(info.get('core_source_sha256')==pilot['source_sha256']['follower'],'follower source identity mismatch')
    require(complete.get('status')=='complete' and complete.get('env_count')==4 and complete.get('mode')==stage,
            'completion contract mismatch')
    require(complete['prepath_sha256']==sha(sim/'prepath.npz') and complete['raw_dem_sha256']==sha(sim/'raw-dem.npz'),
            'prepath or DEM source SHA mismatch')
    if stage=='prepath_only':
        require(complete['control_steps']==0 and not complete['trace_sha256'] and not (sim/'path-activation.json').exists(),
                'candidate activation in prepath-only attempt')
    else:
        require(complete['requested_control_steps']==200 and complete['replan_every_steps']==40,
                'candidate duration mismatch')
        require(len(complete['trace_lengths'])==len(complete['trace_sha256'])==4
                and len(set(complete['trace_lengths']))==1,'trace length identity mismatch')
        require(not set(complete['terminated_envs'])&set(complete['censored_envs']),'terminal/censor overlap')
    return host,info,complete,outer


def verify_path_activation(activation,body_path,rows,env,initial_root,initial_local_root):
    import numpy as np
    require(np.allclose(initial_root,rows[0]['root_state'],rtol=0,atol=1e-6),'initial trace/prepath root mismatch')
    points=activation['paths_tile_xy_m'][env]
    angle=root_yaw(rows[0]['root_state'])
    require(angle is not None and len(points)==len(body_path)==8 and
            all(finite_vector(p,2) is not None for p in body_path+points),'invalid activation Path frame/shape')
    c,s=math.cos(angle),math.sin(angle)
    for (x,y),point in zip(body_path,points):
        expected=(c*x-s*y,s*x+c*y)
        measured=(point[0]-initial_local_root[0],point[1]-initial_local_root[1])
        require(max(abs(a-b) for a,b in zip(expected,measured))<2e-6,'body/activation Path frame mismatch')


def audit_pilot(source_root,output_root,*,make_figures=False):
    """Verify first; publish into a fresh output directory only after all checks."""
    import numpy as np
    source_root,output_root=Path(source_root).resolve(),Path(output_root).resolve()
    require(source_root!=output_root and source_root not in output_root.parents,
            'derived output must be outside source evidence')
    if output_root.exists():
        raise FileExistsError(f'fresh derived output required: {output_root}')
    pilot=read(source_root/'pilot-manifest.json');assignment=read(source_root/'assignment.json')['rows']
    verify_assignment(source_root,pilot,assignment)
    records,differences,integrity,cycle_evidence=[],[],[],[]
    bases={};source_files={p.name:sha(p) for p in source_root.iterdir() if p.is_file()};initial_cache={}
    old_counts,new_counts=Counter(),Counter()
    for index,a in enumerate(assignment):
        attempt=source_root/f'{index:03d}-{a["state_group_id"]}-{a["candidate_id"] or "prepath"}'
        host,info,complete,outer=verify_attempt(attempt,a,pilot)
        sim=attempt/'sim-output'
        from .fall_stall_evidence import validate_initial
        initial=validate_initial(sim,a['terrain_seed'],REPO,read,sha,require,initial_cache)
        source_files.update({str((attempt/relative).relative_to(source_root)):digest for relative,digest in outer.items()})
        source_files[str((attempt/'outer-seal.json').relative_to(source_root))]=sha(attempt/'outer-seal.json')
        source_files[str((attempt/'outer-sha-ledger.json').relative_to(source_root))]=sha(attempt/'outer-sha-ledger.json')
        integrity.append({'attempt':attempt.name,'outer_seal_sha256':sha(attempt/'outer-seal.json'),
                          'inner_seal_sha256':sha(sim/'host-seal.json'),'pass':True,
                          'dem_reconstruction_max_abs_m':initial['dem_reconstruction_max_abs_m']})
        prepath_hash=sha(sim/'prepath.npz')
        if a['stage']=='prepath':
            bases[a['state_group_id']]=prepath_hash
            continue
        require(host.get('reference_prepath_sha256')==info.get('reference_prepath_sha256')==
                bases[a['state_group_id']]==prepath_hash,'cross-candidate prepath binary identity mismatch')
        activation=read(sim/'path-activation.json')
        require(activation['direction']==a['direction'] and activation['arc_angle_deg']==a['angle'],
                'activation candidate identity mismatch')
        labels=read(sim/'long-horizon-event-labels.json')['trials']
        require(len(labels)==4,'label env count mismatch')
        roots=initial['root']
        for env in range(4):
            trace_path=sim/f'trace-env-{env}.jsonl'
            require(sha(trace_path)==complete['trace_sha256'][env],'trace SHA mismatch')
            rows=[json.loads(line) for line in trace_path.read_text().splitlines()]
            require(len(rows)==complete['trace_lengths'][env],'trace row count mismatch')
            require(label_trace(rows)==labels[env],'frozen v1 label regeneration mismatch')
            verify_path_activation(activation,info['path_body_xy_m'],rows,env,roots[env],initial['root_tile_local'][env])
            v2=build_targets(rows,info['path_body_xy_m'],decision_stamp_ns=activation['activation_stamp_ns'])
            terminal=rows[-1].get('terminal') or {}
            if env in complete['terminated_envs']:
                require(bool(terminal.get('pre_reset_state')),'natural termination missing pre-reset snapshot')
            elif env in complete['censored_envs']:
                require(terminal.get('termination_reason')=='parallel_env_censored_by_other_env','parallel censor mismatch')
            else:
                require(len(rows)==200 and not terminal,'partial trace without termination/censor reason')
            identity={'attempt':attempt.name,'state_group':f'{a["state_group_id"]}/env{env}',
                      'seed':a['terrain_seed'],'env':env,'candidate':a['candidate_id'],
                      'terrain':a['terrain_type'],'level':a['terrain_level'],
                      'trace':str(trace_path.relative_to(source_root)),'trace_sha256':sha(trace_path),
                      'v1_label_sha256':sha(sim/'long-horizon-event-labels.json')}
            records.append({**identity,'targets':v2})
            differences.append({**identity,**compare_labels(labels[env],v2)})
            cycle_evidence.extend({**identity,**cycle} for cycle in compare_cycle_measurements(rows,v2))
            old_counts.update(labels[env]['events']);new_counts.update(v2['event_counts'])
    require(len(records)==pilot['candidate_trajectory_count'],'derived trajectory count mismatch')
    event_support={}
    for horizon in HORIZONS:
        event_support[f'{horizon}s']={name:dict(Counter('unknown' if r['targets']['horizons'][f'{horizon}s']['events'][name]['label'] is None
                                                     else 'positive' if r['targets']['horizons'][f'{horizon}s']['events'][name]['label']==1
                                                     else 'negative' for r in records)) for name in EVENT_NAMES}
        for support in event_support[f'{horizon}s'].values():
            for category in ('positive','negative','unknown'):
                support.setdefault(category,0)
    summary={'schema':SCHEMA,'processes_verified':len(integrity),'trajectory_count':len(records),
             'state_group_count':len({r['state_group'] for r in records}),'base_seed_count':len(pilot['state_groups']),
             'observation_windows':sum(r['targets']['row_count'] for r in records),
             'unique_initial_evidence_groups':len(initial_cache),
             'unique_dem_frames_verified':sum(v['dem_frames_verified'] for v in initial_cache.values()),
             'max_raw_dem_reconstruction_error_m':max(v['dem_reconstruction_max_abs_m'] for v in initial_cache.values()),
             'event_counts_v1':dict(old_counts),'event_counts_v2':dict(new_counts),
             'event_trajectory_counts_v2':{name:sum(r['targets']['event_counts'][name]>0 for r in records) for name in EVENT_NAMES},
             'event_seed_support_v2':{name:sorted({r['seed'] for r in records if r['targets']['event_counts'][name]>0}) for name in EVENT_NAMES},
             'horizon_support':event_support,
             'event_change_records':sum(len(d['event_changes']) for d in differences),
             'cycle_change_records':sum(len(d['cycle_changes']) for d in differences),
             'horizon_change_records':sum(len(d['horizon_changes']) for d in differences),
             'complete_cycle_count':sum(c['v2']['complete'] for c in cycle_evidence),
             'partial_cycle_count':sum(not c['v2']['complete'] for c in cycle_evidence),
             'coarse_best_only_positive_cycles':sum(c['diagnostic_best_rule_label']==1 and c['v2']['linear_stall']['label']==0
                                                   for c in cycle_evidence),
             'recovery_labels':dict(Counter('unknown' if recovery['label'] is None else str(recovery['label'])
                                            for r in records for recovery in r['targets']['recoveries'])),
             'review_type':'numerical_differential_audit; independent_human_annotation_pending',
             'dataset_tensor_generation':False,'training_performed':False}
    # Everything above is read-only and validated before creating the destination.
    output_root.mkdir(parents=True,exist_ok=False)
    with (output_root/'targets.jsonl').open('x') as stream:
        for record in records:
            stream.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    write(output_root/'label-diff.json',{'schema':SCHEMA,'comparison':'initial_decision_(0,H]; v1 events reaggregated, not saved sliding labels',
                                       'trajectories':differences})
    write(output_root/'summary.json',summary)
    write(output_root/'source-integrity.json',integrity)
    write(output_root/'cycle-measurement-comparison.json',cycle_evidence)
    write(output_root/'dataset-manifest.json',{'schema':SCHEMA,'artifact_kind':'teachers_and_differential_audit_only',
          'source_root':str(source_root),'source_manifest_sha256':sha(source_root/'pilot-manifest.json'),
          'assignment_sha256':sha(source_root/'assignment.json'),'frozen_source_sha256':pilot['source_sha256'],
          'source_file_sha256':source_files,'spec_path':str(SPEC.relative_to(REPO)),'spec_sha256':sha(SPEC),
          'implementation_sha256':{str(p.relative_to(REPO)):sha(p) for p in
               [Path(__file__),Path(__file__).with_name('fall_stall_targets.py'),
                Path(__file__).with_name('fall_stall_audit_figures.py'),Path(__file__).with_name('fall_stall_evidence.py'),
                Path(__file__).with_name('config.py'),Path(__file__).with_name('dem.py'),
                REPO/'scripts/eat_nav_fall_stall_targets_v2.py']},
          'python_version':sys.version,'numpy_version':np.__version__,'horizons_sec':list(HORIZONS),
          'control_hz':20,'replan_every_steps':40,'thresholds':{'near_fall_deg':60,'near_fall_samples':5,
               'tip_over_deg':85,'linear_commanded_min_m':.20,'linear_ratio_max':.25,'linear_best_progress_max_m':.05,
               'turn_commanded_min_radps':.01,'turn_response_ratio_max':.5,'recovery_progress_min_m':.10,'recovery_window_sec':5}})
    with (output_root/'review.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['state_group','seed','candidate','event','cycle','v1_label','v2_label','reason','human_review'])
        writer.writeheader()
        for diff in differences:
            for change in diff['cycle_changes']:
                writer.writerow({**{k:diff[k] for k in ('state_group','seed','candidate')},
                                 **{k:change[k] for k in ('event','cycle','v1_label','v2_label','reason')},
                                 'human_review':'pending'})
    if make_figures:
        from .fall_stall_audit_figures import write_figures
        write_figures(output_root,records,differences,source_root)
    seal_derived(output_root)
    return summary
