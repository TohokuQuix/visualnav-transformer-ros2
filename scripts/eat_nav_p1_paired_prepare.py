"""Freeze the two-case, four-condition, three-block paired-noise P1 diagnostic."""
from __future__ import annotations
import hashlib, json, shutil
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
V6=Path('/tmp/eatnav-p1-causal-20261009-v6')
ROOT=Path('/tmp/eatnav-p1-paired-noise-20261010-v7')
GROUPS=('first-differential','second-graded')
SEEDS={'first-differential':[2026101101,2026101102,2026101103],
       'second-graded':[2026101111,2026101112,2026101113]}
PATHS={'first-differential':{'A':'4ec126da6c09e7ebc02f2febba35666ef57a334cec9745301554a13aa1d0155e',
                             'curved':'01c33ca38a4e1dcd883714ac63033f22fae685e6a1d1b24f7f5cf8c42c3c95e2'},
       'second-graded':{'A':'4ec126da6c09e7ebc02f2febba35666ef57a334cec9745301554a13aa1d0155e',
                        'curved':'b0a721766cbc4ac03bf24e41d46cd0cecd5b7bbe5b6578ec8c16348a3648f403'}}
ORDERS=[('A','B','S','T'),('S','A','T','B'),('T','S','B','A')]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temp.replace(path)
def once(s,a,b):
    if s.count(a)!=1:raise ValueError(f'expected one anchor, got {s.count(a)}: {a[:90]}')
    return s.replace(a,b)

def build_source():
    source=(V6/'generated-source.py').read_text()
    source=once(source,'from visualnav_transformer.eat_nav.p1_causal_intervention import applied_commands',
        'from visualnav_transformer.eat_nav.p1_paired_command import intervene\n'
        'from visualnav_transformer.eat_nav.p1_paired_noise import PairedPolicyNoise, key_seed\n'
        'from visualnav_transformer.eat_nav.p1_paired_integrity import assess_step')
    source=once(source,'AppLauncher.add_app_launcher_args(parser)',
                "parser.add_argument('--gate1-probe',action='store_true')\nAppLauncher.add_app_launcher_args(parser)")
    source=once(source,"if sha(args.input/'reference-forward-schedule.json') != freeze['reference_schedule_sha256']:\n    raise ValueError('reference schedule SHA drift')\n",'')
    source=once(source,"reference_schedule=json.loads((args.input/'reference-forward-schedule.json').read_text())['forward_m_s']\nif len(reference_schedule)!=120:\n    raise ValueError('reference schedule length')\n",'')
    source=once(source,"if config_json['policy_action_mode']!='act_inference' or config_json['paired_policy_noise']:",
                "if config_json['policy_action_mode']!='act_inference' or not config_json['paired_policy_noise']:")
    source=once(source,"env=None;override=None;capture=None;bridge=None", "env=None;override=None;capture=None;pairer=None;bridge=None")
    source=once(source,"capture=ObservationCapture(bare.observation_manager)\n        capture.begin_step()",
                "pairer=PairedPolicyNoise(bare.observation_manager,args.anchor_id,batch_seed)\n        capture=ObservationCapture(bare.observation_manager)\n        pairer.set_step(0)\n        capture.begin_step()")
    source=once(source,"raw=bare.observation_manager.compute()\n        obs=bare._apply_policy_observation_latency(raw)",
        "raw=bare.observation_manager.compute()\n        latency_prev=bare._latency_policy_obs_prev.detach().clone() if bare._latency_policy_obs_prev is not None else None\n        latency_reset=bare._latency_policy_obs_reset_mask.detach().clone()\n        obs=bare._apply_policy_observation_latency(raw)")
    source=once(source,"observed=capture.finish_step(obs)\n        if tuple(obs['policy'].shape)",
                "observed=capture.finish_step(obs)\n        noise_current=pairer.verify_step()\n        if tuple(obs['policy'].shape)")
    source=once(source,"path_lookup={p['path_sha256']:p for p in contract['paths']}",
        """baseline_errors={}
        b_slot=next(row['slot'] for row in assigned if row['condition']=='B')
        offset=0
        expected_post=observed['policy_post_noise_pre_delay'][b_slot:b_slot+1].clone()
        noise_names=[name for name,cfg in zip(
            bare.observation_manager._group_obs_term_names['policy'],
            bare.observation_manager._group_obs_term_cfgs['policy'])
            if callable(getattr(getattr(cfg,'noise',None),'func',None))]
        original_noise=dict(zip(noise_names,(func for _,func in pairer.originals)))
        for name,cfg,dim in zip(bare.observation_manager._group_obs_term_names['policy'],
                                bare.observation_manager._group_obs_term_cfgs['policy'],
                                bare.observation_manager._group_obs_term_dim['policy']):
            width=math.prod(dim)
            if name in original_noise:
                raw_term=capture.pre_terms[name][b_slot:b_slot+1]
                devices=[torch.cuda.current_device()] if raw_term.device.type=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(key_seed(args.anchor_id,batch_seed,0,name))
                    expected=original_noise[name](raw_term,cfg.noise)
                if cfg.clip:expected=expected.clamp(min=cfg.clip[0],max=cfg.clip[1])
                if cfg.scale is not None:expected=expected*cfg.scale
                actual=observed['policy_post_noise_pre_delay'][b_slot:b_slot+1,offset:offset+width]
                baseline_errors[name]=float((expected.reshape_as(actual)-actual).abs().max())
                expected_post[:,offset:offset+width]=expected.reshape_as(actual)
            offset+=width
        if latency_prev is None or bool(latency_reset[b_slot]):
            expected_actor=expected_post
        else:
            alpha=(bare._onix_latency_seconds[b_slot]/float(bare.step_dt)).clamp(0.0,1.0)
            expected_actor=expected_post+alpha*(latency_prev[b_slot:b_slot+1]-expected_post)
        baseline_errors['post_delay_actor_input']=float((expected_actor-obs['policy'][b_slot:b_slot+1]).abs().max())
        with torch.inference_mode():
            expected_action=policy.act_inference({'policy':expected_actor,'critic':obs['critic'][b_slot:b_slot+1]})
            observed_action=policy.act_inference({'policy':obs['policy'][b_slot:b_slot+1],
                                                   'critic':obs['critic'][b_slot:b_slot+1]})
        baseline_errors['Actor_action']=float((expected_action-observed_action).abs().max())
        if set(baseline_errors)!={'base_ang_vel','joint_pos','joint_vel','projected_gravity',
                                  'post_delay_actor_input','Actor_action'} or max(baseline_errors.values())>2e-6:
            raise ValueError('instrumented B baseline noise equivalence failed: '+repr(baseline_errors))
        atomic(args.output/'gate1-noise-baseline.json',{'B_slot':b_slot,'per_term_max_abs':baseline_errors,
            'noise_terms':list(noise_current),'pre_delay_shape':list(observed['policy_post_noise_pre_delay'].shape),
            'actor_input_shape':list(observed['policy_actor_input'].shape)})
        if args.gate1_probe:
            atomic(args.output/'gate1-probe-complete.json',{'stage':'pre_Path_instrumentation_probe',
                'Path_publications':0,'baseline_errors':baseline_errors,'noise_terms':list(noise_current)})
            return
        path_lookup={p['path_sha256']:p for p in contract['paths']}""")
    source=once(source,"applied=applied_commands(commands[:5],conditions,reference_schedule[0])",
                "diagnostics=startup['follower_diagnostics']\n        if len(diagnostics)!=5:raise ValueError('follower diagnostic count')\n        interventions=[intervene(commands[i],diagnostics[i],conditions[i],stamp_ns=logical_t0) for i in range(4)]\n        applied=[row[0] for row in interventions]")
    source=once(source,"'condition':conditions[i],", "'condition':conditions[i],\n                        'follower_diagnostic':diagnostics[i],\n                        'intervention_reason':interventions[i][1],\n                        'actor_injected_noise':{name:values(sample[0]) for name,sample in noise_current.items()},")
    source=once(source,"capture.begin_step()\n                with torch.inference_mode():\n                    next_obs,reward,done,_=env.step(action_batch)",
                "pairer.set_step(step+1)\n                capture.begin_step()\n                with torch.inference_mode():\n                    next_obs,reward,done,_=env.step(action_batch)")
    source=once(source,"observed=capture.finish_step(next_obs)\n                for i in range(5):",
                "observed=capture.finish_step(next_obs)\n                noise_current=pairer.verify_step()\n                for i in range(4):")
    source=once(source,"applied=applied_commands(commands[:5],conditions,reference_schedule[step+1])",
                "diagnostics=response['follower_diagnostics']\n                    if len(diagnostics)!=5:raise ValueError('follower diagnostic count')\n                    interventions=[intervene(commands[i],diagnostics[i],conditions[i],stamp_ns=next_stamp) for i in range(4)]\n                    applied=[row[0] for row in interventions]")
    source=once(source,"if capture is not None:capture.close()", "if capture is not None:capture.close()\n        if pairer is not None:pairer.close()")
    source=once(source,"'reference_schedule_sha256':sha(args.input/'reference-forward-schedule.json')", "'noise_contract':'keyed_original_additive_uniform_p1_v1'")
    # Replace only structural five-env/six-follower constants.  The sixth
    # follower was a direct monitor; it becomes the fifth follower.
    for old,new in [('range(5)','range(4)'),('(1,5)','(1,4)'),('(5,22)','(4,22)'),
                    ('(5,267)','(4,267)'),('(5,6)','(4,6)'),('[0]*5','[0]*4'),
                    ('[False]*5','[False]*4'),('[:5]','[:4]'),('[5]','[4]'),
                    (']*5',']*4'),('==5','==4'),('!=5','!=4'),('branches=6','branches=5'),
                    ('commands[5]','commands[4]'),('range(1,5)','range(1,4)'),
                    ('torch.arange(5','torch.arange(4'),('num_cols=1,5','num_cols=1,4'),
                    ("'scene_num_envs':5","'scene_num_envs':4")]:
        source=source.replace(old,new)
    # Explicit leftovers must be reviewed rather than accidentally remaining.
    for bad in ('reference_schedule','applied_commands','commands[:5]','range(5)','branches=6'):
        if bad in source:raise ValueError(f'unconverted simulator source: {bad}')
    source=once(source,'make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,5)',
                'make_collection_cfg(OnixRoughEnvCfg(**toggles),eat_cfg,batch_seed,4)')
    source=once(source,'CommandOverride(bare.command_manager,5)',
                'CommandOverride(bare.command_manager,4)')
    source=once(source,'if 0<=index<5:', 'if 0<=index<4:')
    if source.count("if len(diagnostics)!=4:raise ValueError('follower diagnostic count')")!=2:
        raise ValueError('follower diagnostic count anchor drift')
    source=source.replace("if len(diagnostics)!=4:raise ValueError('follower diagnostic count')",
                          "if len(diagnostics)!=5:raise ValueError('follower diagnostic count')")
    if source.count('50_000_000')!=2 or '40_000_000' in source:
        raise ValueError('20 Hz exact-stamp interval drift')
    source=once(source,'def main():',
        """def accepted_events(response,stamp_ns):
    events=[event for event in response['events'] if event['accepted'] and event['stamp_ns']==stamp_ns]
    if len(events)!=5 or {event['branch'] for event in events}!=set(range(5)):
        raise ValueError('accepted exact-stamp bridge event count')
    if any(event['tf_stamp_ns']!=stamp_ns for event in events):
        raise ValueError('accepted command TF stamp mismatch')
    return {event['branch']:event for event in events}


def main():""")
    source=once(source,"env.reset()\n        robot=bare.scene['robot']",
        """env.reset()
        robot=bare.scene['robot']
        contact_sensor=bare.scene['contact_forces']
        contact_body_names=list(contact_sensor.body_names)
        if len(contact_body_names)!=contact_sensor.data.net_forces_w.shape[1]:
            raise ValueError('contact body identity/count mismatch')
        atomic(args.output/'contact-body-names.json',{'body_names':contact_body_names,
            'count':len(contact_body_names),'force_frame':'world'})""")
    source=once(source,"diagnostics=startup['follower_diagnostics']\n        if len(diagnostics)!=5:raise ValueError('follower diagnostic count')",
        "diagnostics=startup['follower_diagnostics']\n        if len(diagnostics)!=5:raise ValueError('follower diagnostic count')\n        bridge_events=accepted_events(startup,logical_t0)\n        if any(row['linear_recovery'] or row['angular_recovery'] for row in diagnostics[:4]):\n            raise ValueError('unexpected recovery at Path activation')")
    source=once(source,"diagnostics=response['follower_diagnostics']\n                    if len(diagnostics)!=5:raise ValueError('follower diagnostic count')",
        "diagnostics=response['follower_diagnostics']\n                    if len(diagnostics)!=5:raise ValueError('follower diagnostic count')\n                    bridge_events=accepted_events(response,next_stamp)\n                    if any(row['linear_recovery'] or row['angular_recovery'] for row in diagnostics[:4]):\n                        abort_run('unexpected_recovery',step+1)")
    source=once(source,"'follower_diagnostic':diagnostics[i],",
                "'follower_diagnostic':diagnostics[i],\n                        'bridge_acceptance':bridge_events[i],")
    source=once(source,"'actor_input':values(obs['policy'][i]),",
                "'policy_latency_seconds':float(bare._onix_latency_seconds[i]),\n                        'actor_input':values(obs['policy'][i]),")
    source=once(source,"'pilot_summary_sha256':sha(args.output/'pilot-summary.json'),",
                "'pilot_summary_sha256':sha(args.output/'pilot-summary.json'),\n               'contact_body_names_sha256':sha(args.output/'contact-body-names.json'),")
    source=once(source,"traces=[[] for _ in range(4)]\n        ended=[False]*4",
        """traces=[[] for _ in range(4)]
        def abort_run(reason,control_step):
            for index,trace in enumerate(traces):
                with (args.output/f'abort-trace-slot-{index}.jsonl').open('x') as stream:
                    for record in trace:
                        stream.write(json.dumps(record,sort_keys=True,allow_nan=True)+'\\n')
            atomic(args.output/'abort.json',{'reason':reason,'control_step':control_step,
                'Path_publications':1,'trace_lengths':[len(trace) for trace in traces]})
            raise ValueError('paired diagnostic abort: '+reason)
        ended=[False]*4""")
    source=once(source,"obs=next_obs\n                atomic(args.output/'heartbeat.json'",
        """obs=next_obs
                if not bool(torch.isfinite(obs['policy']).all()):
                    abort_run('nonfinite_Actor_observation',step)
                for i in range(4):
                    if traces[i] and traces[i][-1]['control_step']==step:
                        reasons=assess_step(traces[i][-1],
                            roll_pitch_cap_rad=config_json['abort_roll_pitch_rad'],
                            speed_cap_mps=config_json['abort_root_speed_mps'],
                            angular_cap_radps=config_json['angular_command_cap_radps'])
                        if reasons:abort_run(','.join(reasons),step)
                atomic(args.output/'heartbeat.json'""")
    compile(source,'paired-source.py','exec')
    return source

def build_paths():
    source=(V6/'generated-p1-statistical-pilot.py').read_text()
    for old,new in [('len(rows) != 5','len(rows) != 4'),('len(poses) != 5','len(poses) != 4'),
                    ('set(range(5))','set(range(4))'),("'five-slot assignment required'","'four-slot assignment required'"),
                    ('[None] * 5','[None] * 4'),('six production followers','five production followers'),
                    ('Branch six','Branch five')]:
        source=source.replace(old,new)
    compile(source,'paired-paths.py','exec')
    return source

def build_protocol():
    source=(V6/'generated-p1-statistical-protocol.py').read_text()
    for old,new in [('len(records) != 5','len(records) != 4')]:
        source=source.replace(old,new)
    if source.count('len(records) != 4')!=2:
        raise ValueError('four-env gate contract drift')
    compile(source,'paired-protocol.py','exec')
    return source

def prepare():
    if ROOT.exists():raise FileExistsError(ROOT)
    source=build_source();paths=build_paths();protocol=build_protocol()
    ROOT.mkdir()
    (ROOT/'source.py').write_text(source)
    (ROOT/'paths.py').write_text(paths)
    (ROOT/'protocol.py').write_text(protocol)
    assignment={'schema':'p1_paired_noise_assignment_v1','batch_seeds':SEEDS,
                'independent_unit':'StateGroup x block_seed','condition_order_by_block':ORDERS,'trials':[]}
    for group in GROUPS:
        for block,order in enumerate(ORDERS):
            for slot,condition in enumerate(order):
                identity=PATHS[group]['A' if condition=='A' else 'curved']
                assignment['trials'].append({'trial_id':f'{group}-block{block}-{condition}',
                    'anchor_id':group,'batch_index':block,'batch_seed':SEEDS[group][block],
                    'slot':slot,'condition':condition,'path_sha256':identity})
    atomic(ROOT/'assignment.json',assignment)
    config=json.loads((V6/'config.json').read_text())
    config.update({'schema':'p1_paired_noise_v1','paired_policy_noise':True,
                   'bridge_sha256':sha(REPO/'scripts/eat_nav_p1_paired_bridge.py'),
                   'intervention_rule':'S removes soft scale; T additionally removes ordinary turn stop; goal/recovery preserved',
                   'noise_method':'sha256(StateGroup,block_seed,control_step,term); original IsaacLab additive uniform function; broadcast per condition',
                   'conditions':['A','B','S','T'],'blocks_per_group':3,
                   'nominal_forward_mps':0.3,'turn_threshold_rad':0.6,
                   'abort_roll_pitch_rad':1.0,'abort_root_speed_mps':5.0,
                   'angular_command_cap_radps':1.6})
    atomic(ROOT/'config.json',config)
    freeze={'schema':'p1_paired_noise_design_freeze_v1','source_sha256':sha(ROOT/'source.py'),
            'paths_sha256':sha(ROOT/'paths.py'),'protocol_sha256':sha(ROOT/'protocol.py'),
            'assignment_sha256':sha(ROOT/'assignment.json'),
            'config_sha256':sha(ROOT/'config.json'),'command_module_sha256':sha(REPO/'src/visualnav_transformer/eat_nav/p1_paired_command.py'),
            'noise_module_sha256':sha(REPO/'src/visualnav_transformer/eat_nav/p1_paired_noise.py'),
            'diagnostic_module_sha256':sha(REPO/'src/visualnav_transformer/eat_nav/p1_follower_diagnostic.py'),
            'integrity_module_sha256':sha(REPO/'src/visualnav_transformer/eat_nav/p1_paired_integrity.py'),
            'bridge_source_sha256':sha(REPO/'scripts/eat_nav_p1_paired_bridge.py'),
            'v6_source_sha256':sha(V6/'generated-source.py'),
            'pilot_order':['first-differential block0','second-graded block0'],
            'remaining_order':['first-differential block1','second-graded block1','first-differential block2','second-graded block2']}
    atomic(ROOT/'design-freeze.json',freeze)
    return {'root':str(ROOT),'trials':len(assignment['trials']),'freeze_sha256':sha(ROOT/'design-freeze.json')}

def prepare_run(group,block):
    if group not in GROUPS or block not in (0,1,2):raise ValueError('unknown run')
    run=ROOT/f'pilot-{group}-b{block}-v1'
    if run.exists():raise FileExistsError(run)
    reference=V6/f'pilot-{group}-b0-v1'
    if not (reference/'host-seal.json').is_file():raise ValueError('source batch not sealed')
    run.mkdir();shutil.copytree(reference/'input',run/'input')
    shutil.copytree(REPO/'src/visualnav_transformer',run/'src/visualnav_transformer')
    shutil.copy2(ROOT/'paths.py',run/'src/visualnav_transformer/eat_nav/p1_statistical_pilot.py')
    shutil.copy2(ROOT/'protocol.py',run/'src/visualnav_transformer/eat_nav/p1_statistical_protocol.py')
    shutil.copy2(ROOT/'source.py',run/'source.py')
    shutil.copy2(ROOT/'assignment.json',run/'input/assignment.json')
    shutil.copy2(ROOT/'config.json',run/'input/statistical-config.json')
    freeze=json.loads((run/'input/freeze.json').read_text())
    freeze.update({'schema':'p1_paired_noise_run_freeze_v1',
        'assignment_sha256':sha(run/'input/assignment.json'),
        'config_sha256':sha(run/'input/statistical-config.json'),
        'pilot_simulator_source_sha256':sha(run/'source.py'),
        'paired_design_sha256':sha(ROOT/'design-freeze.json'),
        'paths_module_sha256':sha(run/'src/visualnav_transformer/eat_nav/p1_statistical_pilot.py'),
        'protocol_module_sha256':sha(run/'src/visualnav_transformer/eat_nav/p1_statistical_protocol.py')})
    freeze.pop('reference_schedule_sha256',None)
    atomic(run/'input/freeze.json',freeze)
    atomic(run/'input/paired-provenance.json',{'source_batch':str(reference),
        'source_batch_seal_sha256':sha(reference/'host-seal.json'),
        'group':group,'block':block,'seed':SEEDS[group][block]})
    return {'run':str(run),'freeze_sha256':sha(run/'input/freeze.json')}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--group',choices=GROUPS);p.add_argument('--block',type=int,choices=(0,1,2))
    a=p.parse_args()
    print(json.dumps(prepare_run(a.group,a.block) if a.group else prepare()))
