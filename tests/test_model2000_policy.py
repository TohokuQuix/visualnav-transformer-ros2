from pathlib import Path
import pytest
from visualnav_transformer.eat_nav import model2000_policy as policy

ROOT=Path(__file__).resolve().parents[1]

def test_verified_policy_bundle_and_legacy_limits():
    if not (ROOT/'checkpoints/eat_nav/rl_actor_model2000/model_2000.pt').exists():
        pytest.skip('retrieve pinned Actor with eat_nav_model2000_fetch_policy.py')
    env,agent=policy.load_bundle(ROOT/'checkpoints/eat_nav/rl_actor_model2000/env.yaml')
    assert agent['empirical_normalization'] is False
    assert env['actions']['right_track_vel']['scale']==14.2
    assert env['actions']['flipper_pos']['joint_names']==['.*rr.*flipper.*','.*rl.*flipper.*','.*fr.*flipper.*','.*fl.*flipper.*']
    a=env['scene']['robot']['actuators']['r_maintrack_sync']
    assert a['velocity_limit']==4.0 and a['velocity_limit_sim'] is None

def test_mixed_policy_bundle_is_rejected(tmp_path):
    if not (ROOT/'checkpoints/eat_nav/rl_actor_model2000/model_2000.pt').exists():
        pytest.skip('retrieve pinned Actor with eat_nav_model2000_fetch_policy.py')
    for name in ['env.yaml','agent.yaml','model_2000.pt']:
        (tmp_path/name).write_bytes((ROOT/'checkpoints/eat_nav/rl_actor_model2000'/name).read_bytes())
    (tmp_path/'agent.yaml').write_text('empirical_normalization: true\n')
    with pytest.raises(ValueError,match='identity'):
        policy.load_bundle(tmp_path/'env.yaml')

def test_paired_legacy_actuator_values_replace_none_without_migration():
    from types import SimpleNamespace
    target=SimpleNamespace(effort_limit=None,effort_limit_sim=40.0,velocity_limit=None,velocity_limit_sim=14.2)
    values={'effort_limit':25.0,'effort_limit_sim':None,'velocity_limit':4.0,'velocity_limit_sim':None}
    policy.apply_actuator_values(target,values)
    assert vars(target)==values
