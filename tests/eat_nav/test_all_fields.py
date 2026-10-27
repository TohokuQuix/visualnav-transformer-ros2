import numpy as np
from eat_nav_all_fields import extend_path, field_bounds, Route, compare_anchor, geometry_choice

def test_reverse_extension_uses_body_heading():
    from eat_nav_risk_navigation import candidate_paths
    p=candidate_paths()[3].astype(float)
    q,h=extend_path(p,0.,3)
    assert np.all(np.diff(q[:,0])<0)
    assert h==0 and len(q)==15

def test_turn_extension_uses_analytic_heading():
    from eat_nav_risk_navigation import candidate_paths
    p=candidate_paths()[2].astype(float)
    q,h=extend_path(p,np.pi/6,2)
    np.testing.assert_allclose(q[8]-p[-1],candidate_paths()[2][1]@np.array([[np.cos(np.pi/6),np.sin(np.pi/6)],[-np.sin(np.pi/6),np.cos(np.pi/6)]]),atol=1e-7)
    assert abs(h-np.pi/3)<1e-7

def test_tile_center_not_spawn_patch():
    b=field_bounds('k_rail_field',[16.,16.],4)
    np.testing.assert_allclose(b['tile_center'],[32.,0.])
    assert b['root_bounds'][0]>24 and b['root_bounds'][2]<40

def test_waypoint_continues_physics_and_rejects_fall():
    r=Route([0.,0.],[0.,0.]);p=r.goal.copy()
    assert not r.observe(p,83.56,False,False)['normal']
    assert r.index==0
    assert not r.observe(p,0,False,True)['normal']
    assert r.observe(p,0,False,False)['normal'] and r.index==1

def test_anchor_mismatch_excludes_group():
    assert compare_anchor({'x':np.array([1.])},{'x':np.array([1.+1e-7])})==[]
    assert compare_anchor({'x':np.array([1])},{'x':np.array([2])})==['x']

def test_geometry_choice_tie_order():
    assert geometry_choice([10.,0.])==0

def test_censored_boundary_not_negative():
    from eat_nav_all_fields_learning import teacher_rows
    rows=[dict(control_step=0,logical_sim_time_ns=1_000_000_000,candidate_cycle=0,candidate_replan=False,
        root_state=[0,0,0,1,0,0,0,0,0,0,0,0,0],post_root_state=[0,0,0,1,0,0,0,0,0,0,0,0,0],
        terminal=None,body_angular_velocity_radps=[0,0,0],follower_command=[.3,0,0],boundary_censor=True)]
    assert teacher_rows(rows)==[]

def test_weighted_sources_exclude_missing_term():
    import torch
    from eat_nav_all_fields_learning import mix_terms
    a={'event':torch.tensor(3.),'progress':torch.tensor(2.),'pair':torch.tensor(1.)}
    b={'event':torch.tensor(6.),'progress':torch.tensor(5.),'pair':torch.tensor(4.)}
    assert abs(float(mix_terms(a,b,{'event':False,'progress':True,'pair':False})['event'])-6)<1e-6
    assert abs(float(mix_terms(a,b,{'event':False,'progress':True,'pair':False})['progress'])-3)<1e-6

def test_delay_snapshot_covers_ring_pointer_and_lag():
    from types import SimpleNamespace as S
    from eat_nav_all_fields import delay_snapshot
    ring=S(_buffer=np.arange(6).reshape(3,1,2),_num_pushes=np.array([9]),_pointer=2)
    env=S(_action_latency_buffers={'main':S(_circular_buffer=ring,_time_lags=np.array([2]))},
        _latency_substeps=np.array([2]),_latency_policy_obs_prev=np.array([[1.]]),
        _latency_policy_obs_reset_mask=np.array([False]),_onix_latency_seconds=np.array([[.01]]))
    a=delay_snapshot(env)
    assert int(a['delay/main/pointer'])==2
    assert a['delay/main/buffer'].shape==(3,1,2)
    assert np.array_equal(a['delay/main/lags'],[2])
    assert 'latency/_latency_policy_obs_reset_mask' in a

def test_dataset_excludes_entire_mismatched_group(tmp_path):
    import json
    from eat_nav_all_fields_learning import field_dataset
    from eat_nav_risk_navigation import pack_inputs
    from visualnav_transformer.eat_nav.fall_stall_audit import write,seal_derived
    root=[0,0,0,1,0,0,0,0,0,0,0,0,0]
    items=[]
    for group in range(2):
        for candidate in range(4):
            i=len(items);items.append(dict(field='k_rail_field',seed=100+group,anchor=0,role='train',candidate=candidate))
            p=tmp_path/f'collect-{i:03d}'/'sim-output';p.mkdir(parents=True)
            write(p/'complete.json',dict(anchor_available=True))
            np.savez_compressed(p/'anchor-state.npz',root=np.asarray(root,dtype=float)+(1 if group==1 and candidate==3 else 0))
            arrays=pack_inputs(np.stack([np.stack([np.zeros((20,20)),np.ones((20,20))])]*4),np.zeros(22),np.zeros(9))
            np.savez_compressed(p/'initial-input.npz',**arrays)
            write(p/'anchor.json',dict(root=root))
            rows=[dict(control_step=j,logical_sim_time_ns=1_000_000_000+j*50_000_000,candidate_cycle=0,candidate_replan=False,
                root_state=root,post_root_state=root,terminal=None,body_angular_velocity_radps=[0,0,0],follower_command=[.3,0,0],boundary_censor=False) for j in range(20)]
            (p/'trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n');seal_derived(p)
    write(tmp_path/'manifest.json',dict(collect=items))
    arrays,index,roles=field_dataset(tmp_path,'k_rail_field')
    assert len(index)==1 and roles['train']==[0]
    assert arrays['candidate_mask'].all()
    assert not arrays['event_mask'][:,:,3,:].any()
    audit=json.loads((tmp_path/'datasets/k_rail_field/anchor-audit.json').read_text())
    assert audit[1]['status']=='anchor_mismatch'

def test_manifest_keeps_all_seeds_and_roles_disjoint():
    from eat_nav_all_fields import make_manifest
    m=make_manifest({'groups':[{'terrain_seed':100}], 'known_exclusions':[]},[200])
    assert len(m['collect'])==448 and len(m['evaluate'])==84
    assert len({g['seed'] for g in m['groups']})==77
    assert all(sum(r['field']==f for r in m['collect'])==64 for f in m['fields'])

def test_invalid_replay_or_pair_cannot_be_comparison():
    from eat_nav_all_fields_audit import comparison_eligible
    assert comparison_eligible({'eligible':True,'replay_errors':[]},{'matched':True})
    assert not comparison_eligible({'eligible':True,'replay_errors':[{'step':0}]},{'matched':True})
    assert not comparison_eligible({'eligible':True,'replay_errors':[]},{'matched':False})

def test_boundary_crossing_interval_not_in_stall_window():
    from eat_nav_all_fields_audit import observed_windows
    rows=[dict(boundary_censor=j==39) for j in range(40)]
    assert len(observed_windows(rows)[0])==39

def test_evaluation_order_rotates_old_model_too(tmp_path):
    import json
    from eat_nav_all_fields import make_manifest
    from eat_nav_all_fields_pipeline import plan
    m=make_manifest({'groups':[],'known_exclusions':[]},[])
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    p=plan(tmp_path);jobs=[j for j in p['evaluation'] if j['field']=='k_rail_field']
    assert len(jobs)==57
    assert [jobs[i]['condition'] for i in (0,19,38)]==['R0','R1','R2']
    assert len(p['evaluation'])==129
