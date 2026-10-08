import copy
import importlib.util
from pathlib import Path

import pytest


_path = Path(__file__).resolve().parents[1] / 'scripts/eat_nav_radial_anchor_contract.py'
_spec = importlib.util.spec_from_file_location('radial_anchor_contract', _path)
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def fixture():
    anchor = {'pose_id':'p', 'terrain':'boxes', 'terrain_seed':4, 'source_family':'seed:boxes',
              'requested_xy_yaw':[1.,2.,.5], 'actor_config_identity':'cfg'}
    freeze = {'batch_seeds':[3], 'planned_physics_steps':200, 'candidate_free':True}
    pose = {**{key:anchor[key] for key in ('pose_id','terrain','terrain_seed','source_family','requested_xy_yaw')},
            'batch_seed':3,'env_index':7}
    actor = {'shape':[22],'finite':True,'field_order':[f'f{i}' for i in range(22)],
             'values':[0.] * 22,'config_identity':'cfg'}
    frames = [{'frame_index':i,'physics_step':50*(i+1),'shape':[2,20,20],
               'resolution_m':.1,'finite':True,'source_mask':[[True]*20 for _ in range(20)],
               'transformed':[[[0.]*20 for _ in range(20)] for _ in range(2)]} for i in range(4)]
    record={'pose_id':'p','env_index':7,'physics_step_count':200,'control_step_count':20,
            'candidate_free':True,'action':[0.]*6}
    return anchor, freeze, pose, actor, frames, record, contract.field_hash(actor['field_order'])


def verify(parts):
    contract.verify_radial_source(*parts)


def test_radial_cannot_enter_episode_prefix():
    with pytest.raises(ValueError, match='incompatible'):
        contract.reject_episode_prefix({'pose_id':'radial', 'source_shard':'shard-A'})


def test_source_rejections():
    for index,key,value,reason in [
        (0,'episode_key','invented','synthetic'),(0,'terrain_seed',99,'terrain_seed'),
        (1,'action_prefix','invented','synthetic'),
        (0,'source_family','wrong','source_family'),(2,'env_index',9,'environment'),
        (1,'planned_physics_steps',199,'settling'),(3,'field_order',['wrong']*22,'field_order'),
        (5,'action',[1.]*6,'candidate_free')]:
        parts=list(copy.deepcopy(fixture()));parts[index][key]=value
        with pytest.raises(ValueError,match=reason):verify(parts)
    for key,value,reason in [('frame_index',2,'order'),('shape',[2,19,20],'schema'),
                             ('source_mask',[[True]*19 for _ in range(20)],'mask')]:
        parts=list(copy.deepcopy(fixture()));parts[4][0][key]=value
        with pytest.raises(ValueError,match=reason):verify(parts)


def test_clean_source_and_output_only_overlay():
    verify(fixture())
    manifest={'rows':[{'physical_run_id':f'r{i}','order':i,'anchor_id':'a',
                       'candidate_path_sha256':'path','random_seed':123,'candidate_retry_count':0,
                       'navigation_global_goal':{'base_xy_m':[.75,0.]},'output_directory':'/old'}
                      for i in range(28)]}
    over,diff=contract.output_overlay(manifest,'/new')
    assert manifest['rows'][0]['output_directory']=='/old'
    assert all(set(d['changed'])=={'output_directory'} for d in diff)
    assert over['rows'][0]['physical_run_id']=='r0'
    contract.verify_output_overlay(manifest,over)
    for field,value in [('physical_run_id','tampered'),('order',9),('anchor_id','other'),
                        ('candidate_path_sha256','wrong')]:
        tampered=copy.deepcopy(over);tampered['rows'][0][field]=value
        with pytest.raises(ValueError,match='logical_identity'):
            contract.verify_output_overlay(manifest,tampered)


def test_completion_needs_marker_and_no_termination():
    with pytest.raises(ValueError,match='marker'):
        contract.require_completion(0,None,[{'physics_step':200,'termination_phase':'pre_evaluation','reset_counter':0,'episode_id':0,'post_reset_state':False}])
    with pytest.raises(ValueError,match='termination'):
        contract.require_completion(0,{'physics_steps':200},[{'physics_step':200,'termination_phase':'post_reset','reset_counter':1,'episode_id':1,'post_reset_state':True}])


def test_pre_evaluation_trace_is_valid_without_reset():
    contract.require_completion(0,{'physics_steps':200},[{'physics_step':200,'termination_phase':'pre_evaluation',
                                                          'reset_counter':0,'episode_id':0,'post_reset_state':False}])


def test_fixture_output_directory_must_not_preexist(tmp_path):
    script=(Path(__file__).resolve().parents[1]/'scripts/eat_nav_radial_anchor_fixture.py').read_text()
    assert "docker('exec', CONTAINER, 'mkdir', '-p', target/'run'" not in script
