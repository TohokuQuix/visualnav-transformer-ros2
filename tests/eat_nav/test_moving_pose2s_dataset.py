"""Dataset packing invariants; no simulator or private recordings required."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

P = Path(__file__).resolve().parents[2] / 'scripts/eat_nav_moving_pose2s_dataset.py'
spec = importlib.util.spec_from_file_location('moving_dataset', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_old_reverse_slot_is_canonical_seven():
    assert m.local_slot(7, 4) == 3
    assert m.local_slot(7, 14) == 7
    with pytest.raises(ValueError):
        m.local_slot(4, 4)


def test_dem_future_and_t0_mismatch_rejected():
    with pytest.raises(ValueError, match='dem'):
        m.check_dem_stamps([250, 500, 750, 1001], 1000, period_ns=250)
    with pytest.raises(ValueError, match='root'):
        m.check_root(np.zeros(13), np.ones(13))
    m.check_dem_stamps([250, 500, 750, 1000], 1000, period_ns=250)


def test_exact_duplicate_and_parent_role_collision():
    a = dict(field='x', seed=1, original_role='train', source_trace_sha256='t', t0_ns=1,
             candidate_id='forward_direct', input_sha256='i', initial_state_signature='s')
    rows, removed = m.deduplicate([a, dict(a)])
    assert len(rows) == removed == 1
    with pytest.raises(ValueError, match='parent'):
        m.assign_roles([a, dict(a, original_role='validation')])


def test_initial_signature_merges_validation_and_train_parents():
    a = dict(field='x', seed=1, original_role='train', initial_state_signature='same')
    b = dict(field='x', seed=2, original_role='validation', initial_state_signature='same')
    result = m.assign_roles([a, b])
    assert result == ['train', 'train']


def test_split_seed_parent_kept_across_levels_and_starts():
    rows = [dict(field='x', seed=s, original_role='validation', initial_state_signature=str(s))
            for s in range(4)]
    roles = m.assign_roles(rows)
    assert roles.count('validation') == roles.count('audit_eval') == 2
    assert m.assign_roles(rows + [dict(rows[0], level=4)])[4] == roles[0]


def test_teacher_missing_is_not_candidate_missing():
    arrays = m.stack_samples([dict(dem=np.zeros((4,2,20,20)),actor_state=np.zeros(22),
        body_state=np.zeros(9), paths=np.zeros((1,8,2)), directions=np.ones((1,1)),
        candidate_mask=np.ones(1,dtype=bool), position_targets=np.zeros((1,3)),
        rotation_targets=np.zeros((1,6)), event_targets=np.zeros((1,4)),
        pose_mask=np.zeros(1,dtype=bool), event_mask=np.ones((1,4),dtype=bool))])
    assert arrays['candidate_mask'][0,0] and not arrays['pose_mask'][0,0]
    assert arrays['event_mask'].all()


def test_common_base_parent_all_levels_stays_in_one_role():
    rows=[dict(field='x',seed=s,original_role='validation',split_parent_id='batch/x/slot0',initial_state_signature=str(s))
          for s in (11,42)]
    assert m.assign_roles(rows)==['validation','validation']


def test_actual_packing_and_immutable_resume(tmp_path):
    import json
    raw=tmp_path/'raw';raw.mkdir()
    root=[0,0,0,1,0,0,0,0,0,0,0,0,0]
    rows=[dict(control_step=i,logical_sim_time_ns=1_000_000_000+i*50_000_000,
               root_state=root,post_root_state=root,candidate_id='reverse_direct',candidate_cycle=0,
               candidate_replan=i==0,follower_command=[-.2,0,0],body_angular_velocity_radps=[0,0,0]) for i in range(40)]
    (raw/'trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    dem=np.zeros((1,4,2,20,20),np.float32);dem[:,:,1]=1
    actor=np.zeros((1,22),np.float32);actor[:,5]=-1
    body=np.zeros((1,9),np.float32);body[:,2]=-1
    np.savez(raw/'input-0000.npz',dem=dem,actor_state=actor,body_state=body,
        paths=m.BANK[[0,1,2,7]][None],directions=np.array([1,1,1,-1],np.float32)[None,:,None],candidate_mask=np.ones((1,4),bool))
    np.savez(raw/'raw-dem-0000.npz',root_state_w=np.array([root],np.float32))
    m.write_json(raw/'selection-0000.json',dict(control_step=0,relative_step=0,input_file='input-0000.npz',
        pose=[0,0,0],candidate=3,prefix=False,dem_stamps_ns=[250000000,500000000,750000000,1000000000]))
    desc=dict(kind='r4',source=str(raw),trace_path=str(raw/'trace.jsonl'),field='x',level=1,seed=1,
        original_role='train',sample_id='x/1/anchor0',anchor=0,env=0)
    output=tmp_path/'dataset'
    result=m.build_dataset(dict(records=[desc]),output)
    assert result['counts']['post_dedup']==1
    assert m.read_json(output/'sample-index.json')[0]['candidate_slot']==7
    with np.load(output/'samples.npz') as arrays:
        assert arrays['paths'].shape==(1,1,8,2)
        assert arrays['trajectory_position'].shape==(1,1,40,3)
        assert arrays['pose_mask'].all()
    assert m.build_dataset(dict(records=[]),output)==result
    with (output/'samples.npz').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):
        m.build_dataset(dict(records=[]),output)


def test_fresh_production_only_frozen_role_and_exact_snapshot(tmp_path):
    root=tmp_path/'collection';group=root/'production/job/sim-output/group-000';branch=group/'candidate-00';branch.mkdir(parents=True)
    parent=dict(group_id='moving/seed7',seed=7,field='x',level=1,role='audit-eval')
    m.write_json(root/'plan.json',dict(parents=[parent]))
    m.write_json(group/'group.json',dict(parent,anchor_available=True,state='moving',counterfactual_complete=True))
    state=[0,0,0,1,0,0,0,0,0,0,0,0,0]
    rows=[dict(control_step=i,logical_sim_time_ns=1_000_000_000+i*50_000_000,root_state=state,post_root_state=state,
         candidate_id='forward_direct',candidate_cycle=0,candidate_replan=i==0,follower_command=[.2,0,0],body_angular_velocity_radps=[0,0,0]) for i in range(40)]
    (branch/'trace.jsonl').write_text(''.join(__import__('json').dumps(r)+'\n' for r in rows));m.write_json(branch/'complete.json',dict(state='completed'))
    dem=np.zeros((1,4,2,20,20),np.float32);dem[:,:,1]=1
    actor=np.zeros((1,22),np.float32);actor[:,5]=-1
    body=np.zeros((1,9),np.float32);body[:,2]=-1
    np.savez(branch/'input.npz',dem=dem,actor_state=actor,body_state=body,paths=m.BANK[None],
        directions=np.array([c.direction for c in m.bank()],np.float32)[None,:,None],candidate_mask=np.ones((1,14),bool),
        root_state_w=np.array([state],np.float32),t0_ns=np.array(1_000_000_000),dem_stamps_ns=np.arange(1,5)*250000000)
    source=m.discover_fresh_sources(root);assert len(source['records'])==1
    out=tmp_path/'fresh';m.build_dataset(source,out)
    index=m.read_json(out/'sample-index.json');assert index[0]['role']=='audit_eval' and not index[0]['is_initial']
    assert index[0]['moving']
    merged=tmp_path/'merged';result=m.merge_datasets([out,out],merged)
    assert result['counts']['post_dedup']==1 and result['counts']['duplicates_removed']==1
    m.verify_dataset(merged)
    m.write_json(group/'group.json',dict(parent,role='train',anchor_available=True,state='moving'))
    with pytest.raises(ValueError,match='role differs'):
        m.discover_fresh_sources(root)
