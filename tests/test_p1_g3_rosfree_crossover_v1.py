"""Frozen ROS-free crossover preparation, without an Isaac runtime."""

import json
from pathlib import Path

import pytest

import eat_nav_p1_g3_rosfree_crossover_prepare as prepare
import eat_nav_p1_g3_rosfree_crossover_authorize as authorize
import eat_nav_p1_g3_rosfree_crossover_host as host
from eat_nav_p1_g3_rosfree_crossover_analyze import validate_trial_assignment
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_contract import (
    validate_crossover_prepath,
)
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_path_contract import (
    validate_simulator_paths,
)


REPO = Path(__file__).resolve().parents[1]


def test_three_period_assignment_preserves_frozen_roles_and_sha():
    base = json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = prepare.build_assignment(base)
    assert len(assignment['processes']) == 3
    assert [[row['path_role'] for row in process['trials']]
            for process in assignment['processes']] == [
                ['direct', 'direct', 'left_20', 'right_20'],
                ['left_20', 'right_20', 'direct', 'direct'],
                ['right_20', 'left_20', 'right_20', 'left_20'],
            ]
    for env in range(4):
        assert {process['trials'][env]['path_role'] for process in assignment['processes']} == \
               {'direct', 'left_20', 'right_20'}
    assert assignment['processes'][0]['trials'] == base['processes'][0]['trials']


def test_period_zero_requires_no_prior_and_later_periods_require_exact_prepath():
    reference = {'state_group': 'first-differential', 'seed': 2026101101,
                 'terrain_mesh_sha256': 'a'*64, 'terrain_material_sha256': 'b'*64,
                 'robot_material_sha256': 'c'*64,
                 'envs': [{'env_id': i, 'root_state_w': [1.], 'joint_position': [1.],
                           'joint_velocity': [1.], 'applied_torque': [1.],
                           'actor_pre_noise': [1.], 'actor_post_noise_pre_delay': [1.],
                           'actor_input': [1.], 'actor_action': [1.],
                           'noise': {'sample': [1.]}, 'command': [1.],
                           'shared_dem_frame_sha256': ['d'*64]*4,
                           'contact_current': [1.], 'contact_history': [1.],
                           'critic_observation': [1.], 'critic_value': 1.,
                           'episode_length': 20, 'reset_count': 0}
                          for i in range(4)],
                 'follower_pre_activation': {'instance_count': 0, 'path_activations': 0}}
    assert validate_crossover_prepath(reference, None, 0)['pass']
    assert validate_crossover_prepath(reference, reference, 1)['pass']
    with pytest.raises(ValueError, match='sealed period-0'):
        validate_crossover_prepath(reference, None, 1)
    changed = json.loads(json.dumps(reference))
    changed['follower_pre_activation']['instance_count'] = 1
    with pytest.raises(ValueError, match='follower'):
        validate_crossover_prepath(changed, reference, 2)
    for field,value in [('root_state_w',[2.]),('actor_input',[2.]),
                        ('shared_dem_frame_sha256',['e'*64]*4),
                        ('noise',{'sample':[2.]})]:
        changed = json.loads(json.dumps(reference))
        changed['envs'][0][field] = value
        with pytest.raises(ValueError, match='strict'):
            validate_crossover_prepath(changed, reference, 1)


def test_period_zero_attempt_scope_is_one_process(tmp_path):
    base = json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = prepare.build_assignment(base)
    manifest = prepare.build_manifest(REPO, prepare.REFERENCE, 'a'*64, assignment)
    assert manifest['runtime_authorized'] is False
    assert manifest['process_count'] == 3
    assert manifest['authorized_process_ids'] == []
    assert manifest['external_router_required'] is False
    preparation = tmp_path/'preparation'
    prepare.stage_one(REPO, prepare.REFERENCE, preparation, assignment, manifest)
    bind = tmp_path/'host-bind'
    bind.mkdir()
    attempt = bind/'.eatnav-p1-g3-rosfree-crossover-v1-attempt-0'
    authorize.create_attempt(preparation, attempt, host_bind_source=str(bind),
                             container_bind_destination=str(tmp_path/'container-bind'))
    class Args:
        anchor_id='first-differential'
        batch_index=0
        execution_index=0
    checked, process, assigned = host.validate_staged(attempt/prepare.PROCESS_ID, Args())
    assert checked['authorized_process_ids'] == [prepare.PROCESS_ID]
    assert [r['path_role'] for r in assigned] == ['direct','direct','left_20','right_20']


def test_wrong_role_or_sha_cannot_be_published():
    base = json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    frozen = prepare.build_assignment(base)['processes'][0]['trials']
    observed = [{'trial_id': row['trial_id'], 'path_role': row['path_role'],
                 'path_sha256': row['path_sha256']} for row in frozen]
    validate_frozen_assignment(frozen, observed, stage='post_path')
    for key, bad in [('path_role','direct'),('path_sha256','0'*64)]:
        corrupted = json.loads(json.dumps(observed))
        corrupted[2][key] = bad
        with pytest.raises(ValueError):
            validate_frozen_assignment(frozen, corrupted, stage='post_path')


def test_period_zero_analysis_requires_four_frozen_roles():
    rows=[{'env_id':i,'path_role':role} for i,role in enumerate(
        ['direct','direct','left_20','right_20'])]
    validate_trial_assignment(rows)
    rows[3]['path_role']='left_20'
    with pytest.raises(ValueError):
        validate_trial_assignment(rows)


def test_new_runtime_has_no_ros_control_imports_or_old_protocol_references():
    import ast
    for relative in ('scripts/eat_nav_p1_g3_rosfree_crossover_sim.py',
                     'scripts/eat_nav_p1_g3_rosfree_crossover_host.py'):
        text=(REPO/relative).read_text()
        imports=[]
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or '')
        assert not any(name.startswith(('rclpy','tf2_ros','rmw_zenoh',
                                        'geometry_msgs','nav_msgs')) for name in imports)
        assert 'eatnav_p1_g3_rosfree_direct_core_v3' not in text


def test_crossover_attempt_path_requires_authorization_and_matching_mount(tmp_path):
    host=tmp_path/'host'
    host.mkdir()
    attempt='.eatnav-p1-g3-rosfree-crossover-v1-attempt-0'
    root=tmp_path/'container'/attempt
    batch=root/prepare.PROCESS_ID
    (batch/'input').mkdir(parents=True)
    (batch/'source.py').write_text('source')
    manifest={'schema':'eatnav_p1_g3_rosfree_crossover_v1',
              'runtime_authorized':True,
              'authorized_process_ids':[prepare.PROCESS_ID],
              'automatic_continuation':False,'attempt_id':attempt,
              'host_bind_source':str(host),
              'container_bind_destination':str(tmp_path/'container'),
              'attempt_host_root':str(host/attempt),
              'attempt_container_root':str(root)}
    mount={str(tmp_path/'container')}
    validate_simulator_paths(manifest,batch/'input',batch/'output',
                             batch/'source.py',mount,
                             expected_bind=(str(host),str(tmp_path/'container')))
    with pytest.raises(ValueError):
        validate_simulator_paths(manifest,batch/'input',batch/'output',
                                 batch/'source.py',set(),
                                 expected_bind=(str(host),str(tmp_path/'container')))
    manifest['runtime_authorized']=False
    with pytest.raises(ValueError):
        validate_simulator_paths(manifest,batch/'input',batch/'output',
                                 batch/'source.py',mount,
                                 expected_bind=(str(host),str(tmp_path/'container')))
