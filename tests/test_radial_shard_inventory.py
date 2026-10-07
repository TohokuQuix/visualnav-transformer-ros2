import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.radial_shard_inventory import (
    build_inventory, validate_inventory, sensor_paths_for_env,
    verify_shard_artifacts, numerical_exceedance_stop, select_shard_rows,
    settled_exceedance_reasons,
)


def test_inventory_is_112_unique_23_interleaved_shards():
    inventory = build_inventory(20261007)
    assert len(inventory['poses']) == 112
    assert len(inventory['shards']) == 23
    assert [len(s['poses']) for s in inventory['shards']] == [5]*22 + [2]
    assert len({p['pose_id'] for p in inventory['poses']}) == 112
    assert len({p['pose_id'] for s in inventory['shards'] for p in s['poses']}) == 112
    assert inventory['shards'][0]['poses'][0]['pose_id'] == 'b20261007-boxes-a00'
    assert len({p['terrain'] for p in inventory['shards'][0]['poses']}) == 5
    validate_inventory(inventory)


def test_inventory_rejects_duplicate_changed_geometry_and_six_view_shard():
    inventory = build_inventory(20261007)
    inventory['shards'][0]['poses'][1] = inventory['shards'][0]['poses'][0]
    with pytest.raises(ValueError): validate_inventory(inventory)
    inventory = build_inventory(20261007)
    inventory['shards'][0]['poses'][0]['theta_rad'] = 0.3
    with pytest.raises(ValueError): validate_inventory(inventory)
    inventory = build_inventory(20261007)
    inventory['shards'][0]['poses'].append(inventory['shards'][1]['poses'][0])
    with pytest.raises(ValueError): validate_inventory(inventory)


def test_sensor_paths_expand_only_env_identity():
    paths = ['/World/envs/env_0/onix_chassis/chassis_body',
             '/World/envs/env_0/onix_chassis/chassis_l01_sprocket']
    assert sensor_paths_for_env(paths, 78) == [
        '/World/envs/env_78/onix_chassis/chassis_body',
        '/World/envs/env_78/onix_chassis/chassis_l01_sprocket']
    with pytest.raises(ValueError):
        sensor_paths_for_env(['/World/unrelated/body'], 78)


def test_selection_accepts_last_two_pose_shard_and_rejects_geometry_drift():
    inventory = build_inventory()
    schedule = inventory['poses']
    selected = select_shard_rows(inventory['shards'][-1]['poses'], schedule)
    assert len(selected) == 2
    changed = [dict(x) for x in inventory['shards'][-1]['poses']]
    changed[0]['tile_local_xy_centered_m'] = [0.0, 0.0]
    with pytest.raises(ValueError): select_shard_rows(changed, schedule)


def test_shard_artifacts_require_every_step_dem_actor_and_sha(tmp_path):
    pose = 'b20261007-boxes-a00'
    run = tmp_path/'run';(run/'trace').mkdir(parents=True)
    (run/'trace'/f'{pose}.jsonl').write_text(''.join(json.dumps({'physics_step':i,'contact':{'bounds_valid':True,'buffer_full_or_truncation_suspected':False,'finite':True,'buffer_capacity':4096}})+'\n' for i in range(1,201)))
    for i in range(4):
        p=run/'dem'/pose/f'frame-{i:02d}.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'physics_step':(i+1)*50,'shape':[2,20,20],'finite':True,'valid_fraction':.9}))
    for phase in ('prephysics','settled_step_200'):
        p=run/'actor'/pose/f'{phase}.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'shape':[22],'finite':True,'field_order':['a']*22}))
    assert verify_shard_artifacts(run,[pose])['pose_count'] == 1
    (run/'dem'/pose/'frame-03.json').unlink()
    with pytest.raises(ValueError): verify_shard_artifacts(run,[pose])


def test_terminated_pose_requires_exact_pre_reset_evidence_and_no_post_reset_trace(tmp_path):
    pose = 'b20261007-k_rail_field-a00'
    run = tmp_path/'run'; (run/'trace').mkdir(parents=True)
    stream = run/'trace'/f'{pose}.jsonl'
    stream.write_text(''.join(json.dumps({'physics_step':i,'contact':{
        'bounds_valid':True,'buffer_full_or_truncation_suspected':False,
        'finite':True,'buffer_capacity':4096}})+'\n' for i in range(1,141)))
    for i in range(2):
        p=run/'dem'/pose/f'frame-{i:02d}.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'physics_step':(i+1)*50,'shape':[2,20,20],'finite':True}))
    p=run/'actor'/pose/'prephysics.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'shape':[22],'finite':True,'field_order':['a']*22}))
    import hashlib
    marker={'pose_id':pose,'env_id':5,'termination_step':140,'last_valid_step':140,
            'termination_reason':['tilt_over'],'terminated':True,'truncated':False,
            'done':True,'post_reset_state':False,'episode_id':0,'reset_counter_before':0,
            'reset_counter_after':1,'pre_reset_state':{'root_state_w':[0]*13},
            'pre_reset_actor':[0]*22,'pre_reset_contact':{'bounds_valid':True,'finite':True,
            'buffer_capacity':4096,'buffer_full_or_truncation_suspected':False},
            'term_flags':{'tilt_over':True,'time_out':False},
            'logical_timestamp_ns':700000000,
            'dem_frame_ids':['frame-00','frame-01'],
            'trace_sha256':hashlib.sha256(stream.read_bytes()).hexdigest()}
    p=run/'termination'/f'{pose}.json';p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(marker))
    (run/'control-events.jsonl').write_text(''.join(json.dumps({
        'physics_step':i*10,'selected':{pose:{'terminated':i==14,'truncated':False,
        'done':i==14,'actor_observation':[0.0]*22,'post_reset_state':False}} if i<=14 else {}})+'\n'
        for i in range(1,21)))
    assert verify_shard_artifacts(run,[pose],termination_aware=True)['terminated_pose_count']==1
    stream.write_text(stream.read_text()+json.dumps({'physics_step':141})+'\n')
    with pytest.raises(ValueError): verify_shard_artifacts(run,[pose],termination_aware=True)


def test_numeric_exceedance_stop_is_prospective_majority():
    assert not numerical_exceedance_stop([False, True, True, False, False])
    assert numerical_exceedance_stop([False, True, True, True, False])


def test_settled_numeric_limits_use_all_final_40_steps():
    rows = [{'state':{'xy_drift_m':.01,'h1_signed_rad':.01,'linear_speed_mps':.01,'angular_speed_radps':.01}} for _ in range(200)]
    rows[160]['state']['linear_speed_mps'] = .11
    assert settled_exceedance_reasons(rows,[.9]*4) == ['linear_speed_mps']
    rows[160]['state']['linear_speed_mps'] = .01
    assert settled_exceedance_reasons(rows,[.9,.8,.69,.9]) == ['dem_valid_fraction']
