"""Freeze the bounded five-condition P1 command intervention experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


REPO=Path(__file__).resolve().parents[1]
VIEW=Path('/tmp/eatnav-p1-statistical-v2-recovered-view-20261009')
PILOT_SOURCE=REPO/'scripts/eat_nav_p1_statistical_pilot_sim.py'
PILOT_SHA='60b53d58593f304dabafa5be3ee37b70c4706340ea83c932177631fcc189a2c1'
GROUPS=('first-differential','second-graded')
CONDITIONS=('direct','plus_normal','minus_normal','plus_controlled','minus_controlled')
SEEDS={'first-differential':(2026101001,2026101002,2026101003),
       'second-graded':(2026101011,2026101012,2026101013)}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def once(text,old,new):
    if text.count(old)!=1:raise ValueError(f'expected one source anchor, got {text.count(old)}: {old[:90]}')
    return text.replace(old,new)


def build_source():
    if sha(PILOT_SOURCE)!=PILOT_SHA:raise ValueError('validated pilot simulator source changed')
    source=PILOT_SOURCE.read_text()
    changes={
        "choices=(0, 1)":"choices=(0, 1, 2)",
        "batch_seed=manifest['batch_seeds'][args.batch_index]":"batch_seed=manifest['batch_seeds'][args.anchor_id][args.batch_index]",
        "if len(assigned)!=7 or {row['slot'] for row in assigned}!=set(range(7)):":"if len(assigned)!=5 or {row['slot'] for row in assigned}!=set(range(5)):",
        "if (self.cfg.num_rows,self.cfg.num_cols)!=(1,7):":"if (self.cfg.num_rows,self.cfg.num_cols)!=(1,5):",
        "if tensor.shape[0]==7:":"if tensor.shape[0]==5:",
        "make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,7)":"make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,5)",
        "gen.num_rows,gen.num_cols=1,7":"gen.num_rows,gen.num_cols=1,5",
        "len(fingerprints)!=7":"len(fingerprints)!=5",
        "'scene_num_envs':7":"'scene_num_envs':5",
        "ids=torch.arange(7,device=bare.device)":"ids=torch.arange(5,device=bare.device)",
        "[state['joint_position']]*7":"[state['joint_position']]*5",
        "[state['joint_velocity']]*7":"[state['joint_velocity']]*5",
        "zero=torch.zeros((7,6),device=bare.device)":"zero=torch.zeros((5,6),device=bare.device)",
        "override=CommandOverride(bare.command_manager,7)":"override=CommandOverride(bare.command_manager,5)",
        "*7,\n                                         device=bare.device":"*5,\n                                         device=bare.device",
        "clone_internal_branch_state(bare,0,range(1,7))":"clone_internal_branch_state(bare,0,range(1,5))",
        "'monitor_path_equal_slot_0':paths[7]==paths[0]":"'monitor_path_equal_slot_0':paths[5]==paths[0]",
        "if 0<=index<7:":"if 0<=index<5:",
        "traces=[[] for _ in range(7)]":"traces=[[] for _ in range(5)]",
        "ended=[False]*7":"ended=[False]*5",
        "tuple(actions.shape)!=(7,6)":"tuple(actions.shape)!=(5,6)",
        "validate_bridge_response(startup,logical_t0)":"validate_bridge_response(startup,logical_t0,branches=6)",
        "validate_bridge_response(response,next_stamp)":"validate_bridge_response(response,next_stamp,branches=6)",
    }
    for old,new in changes.items():source=once(source,old,new)
    for old,new in [('(7,22)','(5,22)'),('(7,267)','(5,267)')]:
        if source.count(old)!=2:raise ValueError('observation shape source anchor count')
        source=source.replace(old,new)
    source=source.replace('range(7)','range(5)').replace('[:7]','[:5]').replace('for i in range(7):','for i in range(5):')
    source=once(source,"from visualnav_transformer.eat_nav.p1_statistical_freeze import sha",
        "from visualnav_transformer.eat_nav.p1_statistical_freeze import sha\nfrom visualnav_transformer.eat_nav.p1_causal_intervention import applied_commands")
    source=once(source,"if len(assigned)!=5 or {row['slot'] for row in assigned}!=set(range(5)):",
        "if len(assigned)!=5 or {row['slot'] for row in assigned}!=set(range(5)):")
    source=once(source,"if sha(args.input/'source-contract.json') != freeze['source_contract_sha256']:",
        "if sha(args.input/'reference-forward-schedule.json') != freeze['reference_schedule_sha256']:\n    raise ValueError('reference schedule SHA drift')\nif sha(args.input/'source-contract.json') != freeze['source_contract_sha256']:")
    source=once(source,"assigned=[row for row in manifest['trials'] if row['anchor_id']==args.anchor_id and row['batch_index']==args.batch_index]",
        "assigned=[row for row in manifest['trials'] if row['anchor_id']==args.anchor_id and row['batch_index']==args.batch_index]\nconditions=[row['condition'] for row in sorted(assigned,key=lambda row:row['slot'])]\nreference_schedule=json.loads((args.input/'reference-forward-schedule.json').read_text())['forward_m_s']\nif len(reference_schedule)!=120:\n    raise ValueError('reference schedule length')")
    source=once(source,"commands=validate_bridge_response(startup,logical_t0,branches=6)",
        "commands=validate_bridge_response(startup,logical_t0,branches=6)\n        if max(abs(a-b) for a,b in zip(commands[0],commands[5]))>1e-6:\n            raise ValueError('production follower monitor command mismatch')\n        applied=applied_commands(commands[:5],conditions,reference_schedule[0])")
    source=once(source,"commands=validate_bridge_response(response,next_stamp,branches=6)",
        "commands=validate_bridge_response(response,next_stamp,branches=6)\n                    if max(abs(a-b) for a,b in zip(commands[0],commands[5]))>1e-6:\n                        raise ValueError('production follower monitor command mismatch')\n                    applied=applied_commands(commands[:5],conditions,reference_schedule[step+1])")
    source=source.replace("torch.as_tensor(commands[:5],device=bare.device,dtype=command_tensor.dtype)",
                          "torch.as_tensor(applied,device=bare.device,dtype=command_tensor.dtype)")
    source=once(source,"'follower_command':commands[i],",
        "'follower_command':commands[i], 'applied_command':applied[i],\n                        'condition':conditions[i],\n                        'joint_position':values(robot.data.joint_pos[i]),\n                        'joint_velocity':values(robot.data.joint_vel[i]),\n                        'applied_torque':values(robot.data.applied_torque[i]) if hasattr(robot.data,'applied_torque') else None,\n                        'contact_net_forces_w':values(bare.scene['contact_forces'].data.net_forces_w[i]) if 'contact_forces' in bare.scene.sensors else None,")
    source=once(source,"'trace_sha256':[sha(args.output/f'trace-slot-{i}.jsonl') for i in range(5)]",
        "'trace_sha256':[sha(args.output/f'trace-slot-{i}.jsonl') for i in range(5)],\n               'reference_schedule_sha256':sha(args.input/'reference-forward-schedule.json')")
    if 'range(7)' in source or '(7,22)' in source or 'commands[:7]' in source:
        raise ValueError('unconverted seven-slot simulator expression')
    return source


def build_paths_module():
    src=(REPO/'src/visualnav_transformer/eat_nav/p1_statistical_pilot.py').read_text()
    return (src.replace('len(rows) != 7 or len(poses) != 7','len(rows) != 5 or len(poses) != 5')
            .replace('set(range(7))','set(range(5))').replace('[None] * 7','[None] * 5')
            .replace('seven-slot assignment required','five-slot assignment required')
            .replace('The frozen bridge has eight production followers. Branch eight receives',
                     'The diagnostic bridge has six production followers. Branch six receives'))


def build_protocol_module():
    src=(REPO/'src/visualnav_transformer/eat_nav/p1_statistical_protocol.py').read_text()
    if src.count('len(records) != 7')!=2:
        raise ValueError('initialization gate source changed')
    return src.replace('len(records) != 7','len(records) != 5')


def prepare(study_root):
    if study_root.exists():raise FileExistsError(study_root)
    study_root.mkdir(parents=True)
    base=VIEW/'pilot-first-differential-b0-v1'
    contract=json.loads((base/'input/source-contract.json').read_text())
    path_by_name={('direct_reference' if p['bearing_deg']==0 else f"bearing_{p['bearing_deg']:+d}"):p
                  for p in contract['paths']}
    # Freeze the assignment and all scenario seeds before a Path is exposed.
    trials=[]
    for group in GROUPS:
        for block,seed in enumerate(SEEDS[group]):
            for condition in CONDITIONS:
                path_name='direct_reference' if condition=='direct' else ('bearing_+20' if condition.startswith('plus') else 'bearing_-20')
                slot=(CONDITIONS.index(condition)+block)%5
                trials.append({'trial_id':f'{group}-block{block}-{condition}',
                               'anchor_id':group,'batch_index':block,'batch_seed':seed,
                               'slot':slot,'condition':condition,
                               'path_identity':path_name,'path_sha256':path_by_name[path_name]['path_sha256']})
    manifest={'schema':'p1_causal_assignment_v1','batch_seeds':SEEDS,
              'independent_unit':'StateGroup x block_seed','trials':trials}
    atomic(study_root/'assignment.json',manifest)
    raw_params=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml')
    params=raw_params.read_text()
    if params.count('debug_control_log_period_sec: 0.5')!=1 or params.count('debug_control_log: true')!=1:
        raise ValueError('production follower diagnostic logging parameters changed')
    debug_params=study_root/'follower-debug.yaml'
    debug_params.write_text(params.replace('debug_control_log_period_sec: 0.5','debug_control_log_period_sec: 0.1'))
    source=build_source()
    paths_module=build_paths_module()
    protocol_module=build_protocol_module()
    if 'len(rows) != 5' not in paths_module:raise ValueError('five-slot Path helper not generated')
    (study_root/'generated-source.py').write_text(source)
    (study_root/'generated-p1-statistical-pilot.py').write_text(paths_module)
    (study_root/'generated-p1-statistical-protocol.py').write_text(protocol_module)
    config=json.loads((base/'input/statistical-config.json').read_text())
    config.update({'schema':'p1_causal_diagnostic_v1','follower_params_sha256':sha(debug_params),
                   'reference_speed_m_s':0.3,
                   'intervention_rule':'replace positive final vx only; preserve zero/reverse, vy, wz; hard cap 0.3',
                   'thresholds':{'normal_command_parity_max_abs':1e-6,
                                 'reference_forward_max_abs':0.3,
                                 'substantial_progress_recovery_m':0.025,
                                 'residual_progress_m':0.05}})
    atomic(study_root/'config.json',config)
    atomic(study_root/'design-freeze.json',{'schema':'p1_causal_design_freeze_v1',
           'assignment_sha256':sha(study_root/'assignment.json'),
           'source_sha256':sha(study_root/'generated-source.py'),
           'paths_module_sha256':sha(study_root/'generated-p1-statistical-pilot.py'),
           'protocol_module_sha256':sha(study_root/'generated-p1-statistical-protocol.py'),
           'config_sha256':sha(study_root/'config.json'),
           'debug_params_sha256':sha(debug_params),
           'production_params_sha256':sha(raw_params),
           'follower_binary_sha256':config['follower_sha256'],
           'reference_schedule':'constant 0.3 m/s; validated production hard limit; preserve zero/reverse final commands',
           'staging':'first complete block for each group, then remaining four blocks'})
    return {'study_root':str(study_root),'planned_trials':len(trials),'source_sha256':sha(study_root/'generated-source.py')}


def prepare_run(study_root,group,block):
    if group not in GROUPS or block not in (0,1,2):raise ValueError('unknown diagnostic block')
    root=study_root/f'pilot-{group}-b{block}-v1'
    if root.exists():raise FileExistsError(root)
    reference=VIEW/f'pilot-{group}-b0-v1'
    if not (reference/'host-seal.json').is_file():raise ValueError('sealed source batch missing')
    root.mkdir()
    shutil.copytree(reference/'input',root/'input')
    shutil.copytree(REPO/'src/visualnav_transformer',root/'src/visualnav_transformer')
    shutil.copy2(study_root/'generated-p1-statistical-pilot.py',
                 root/'src/visualnav_transformer/eat_nav/p1_statistical_pilot.py')
    shutil.copy2(study_root/'generated-p1-statistical-protocol.py',
                 root/'src/visualnav_transformer/eat_nav/p1_statistical_protocol.py')
    shutil.copy2(study_root/'generated-source.py',root/'source.py')
    shutil.copy2(study_root/'assignment.json',root/'input/assignment.json')
    shutil.copy2(study_root/'config.json',root/'input/statistical-config.json')
    schedule={'schema':'p1_causal_reference_forward_v1','forward_m_s':[.3]*120,
              'hard_limit_m_s':.3,'preserve_stop_turn_goal_reverse':True}
    atomic(root/'input/reference-forward-schedule.json',schedule)
    freeze=json.loads((root/'input/freeze.json').read_text())
    freeze.update({'schema':'p1_causal_run_freeze_v1',
        'assignment_sha256':sha(root/'input/assignment.json'),
        'config_sha256':sha(root/'input/statistical-config.json'),
        'pilot_simulator_source_sha256':sha(root/'source.py'),
        'reference_schedule_sha256':sha(root/'input/reference-forward-schedule.json'),
        'causal_design_sha256':sha(study_root/'design-freeze.json'),
        'paths_module_sha256':sha(root/'src/visualnav_transformer/eat_nav/p1_statistical_pilot.py')})
    freeze['protocol_module_sha256']=sha(root/'src/visualnav_transformer/eat_nav/p1_statistical_protocol.py')
    atomic(root/'input/freeze.json',freeze)
    atomic(root/'input/causal-provenance.json',{'source_batch':str(reference),
        'source_batch_seal_sha256':sha(reference/'host-seal.json'),
        'group':group,'block':block,'batch_seed':SEEDS[group][block]})
    return {'root':str(root),'freeze_sha256':sha(root/'input/freeze.json')}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--group',choices=GROUPS);p.add_argument('--block',type=int,choices=(0,1,2))
    args=p.parse_args()
    print(json.dumps(prepare_run(args.study_root,args.group,args.block) if args.group is not None
                     else prepare(args.study_root)))
