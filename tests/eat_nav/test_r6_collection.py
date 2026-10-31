"""R6 collection contracts: independent masks, group identity and real geometry."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

PATH = Path(__file__).parents[2] / 'scripts/eat_nav_r6_collection.py'

def load():
    assert PATH.exists(), 'R6 vector collection contract is missing'
    spec = importlib.util.spec_from_file_location('r6_collection', PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_pre_reset_is_used_without_censoring_other_env():
    m = load()
    pre = np.zeros((2, 13)); after = pre.copy(); after[0, 0] = 999
    terminal = {0: {'root_state': [7.] + [0.] * 12}}
    rows = m.step_records(pre, after, [[1, 0, 0]] * 2, [True, False], terminal,
                          [True, True], 3, 0, 1000000000)
    assert rows[0]['post_root_state'][0] == 7
    assert rows[0]['stop_reason'] == 'natural_termination'
    assert rows[1]['valid_mask'] and rows[1]['stop_reason'] is None
    assert rows[0]['terminal']['pre_reset_state']['root_state'][0] == 7
    assert rows[0]['contact_observation']['available'] is False


def test_anchor_verification_rejects_geometry_or_state_change():
    m = load()
    a = {'root': np.zeros((2, 13)), 'dem': np.ones((2, 4, 2, 20, 20))}
    assert m.verify_anchor(a, a)['passed']
    b = {k: v.copy() for k, v in a.items()}; b['root'][1, 0] = .01
    assert not m.verify_anchor(a, b)['passed']


def test_manifest_keeps_every_candidate_in_assigned_group_split():
    m = load()
    groups = m.validate_groups([{'group_id': 'a', 'role': 'validation', 'seed': 123,
                                 'env_index': 0}], 1)
    assert groups[0]['role'] == 'validation'
    with pytest.raises(ValueError):
        m.validate_groups([{'group_id': 'a', 'role': 'test', 'seed': 123}], 2)


def test_fixture_geometry_distinguishes_solid_wall_and_narrow_passage():
    m = load()
    wall = m.fixture_boxes('wall'); narrow = m.fixture_boxes('narrow')
    assert len(wall) == 1 and len(narrow) == 2
    assert wall[0]['size_m'][2] > .5
    assert narrow[0]['center_m'][1] * narrow[1]['center_m'][1] < 0


def test_fixture_distance_is_geometry_proxy_and_never_contact_truth():
    m = load()
    near = m.fixture_proximity([[1.0, 0., .4]], [0, 0, 0], 'wall')
    far = m.fixture_proximity([[-2., 0., .4]], [0, 0, 0], 'wall')
    assert near['minimum_body_origin_distance_m'] < far['minimum_body_origin_distance_m']
    assert near['true_contact_available'] is False
    assert near['definition'] == 'body-frame origins to fixture oriented boxes; not occupied-body clearance'


def test_tile_indices_validate_fixed_parent_terrain_grid():
    m = load()
    assert m.selected_tiles(2, 10, '3,7') == [3, 7]
    with pytest.raises(ValueError):m.selected_tiles(2, 10, '3,3')
    with pytest.raises(ValueError):m.selected_tiles(2, 2, '3,7')


def test_actual_contact_point_on_fixture_wall_excludes_floor_support():
    m = load()
    assert m.fixture_contact_index([1.,0.,.4], [0,0,0], 'wall') == 0
    assert m.fixture_contact_index([0.,0.,0.], [0,0,0], 'wall') is None
    assert m.fixture_contact_index([1.1,0.,0.], [0,0,0], 'wall') is None


def test_actual_physx_mass_inertia_audit_flags_invalid_properties():
    m=load()
    good=m.audit_physics_properties([[1.,2.]],np.tile(np.eye(3).reshape(1,1,9),(1,2,1)))
    assert good['valid']
    bad=m.audit_physics_properties([[1.,-1.]],np.zeros((1,2,9)))
    assert not bad['valid'] and bad['nonpositive_mass_count']==1


def test_one_failed_settling_environment_does_not_drop_other_anchor():
    m=load()
    valid=m.update_anchor_valid([True,True],[True,False])
    assert valid.tolist()==[False,True]
    assert m.update_anchor_valid(valid,[False,False]).tolist()==[False,True]


def test_exact_handoff_wall_case_is_distinct_and_metres():
    m=load();box=m.fixture_boxes('wall_60x100')[0]
    assert box['size_m']==[.2,.6,1.]
    assert box['center_m'][2]==.5
    assert m.fixture_contact_index([1.,0.,.5],[0,0,0],'wall_60x100')==0


def test_one_body_filtered_contact_view_preserves_environment_identity():
    m=load();forces=np.array([[2.],[3.]]);points=np.array([[1.,0.,.4],[1.,16.,.4]])
    rows=m.decode_fixture_contacts(forces,points,np.ones((2,3)),np.zeros((2,1)),np.array([[1],[1]]),np.array([[0],[1]]),[[0,0,0],[0,16,0]],'wall','front_roller',99,.005)
    assert [r['env_index'] for r in rows]==[0,1]
    assert rows[1]['body_name']=='front_roller' and rows[1]['normal_force_n']==3


def test_filtered_primitive_contact_is_truth_even_with_deep_penetration():
    m=load()
    rows=m.decode_fixture_contacts(np.array([[2.]]),np.array([[1.1,0.,.4]]),np.ones((1,3)),np.array([[-.1]]),np.array([[1]]),np.array([[0]]),[[0,0,0]],'wall','body',1,.005)
    assert len(rows)==1 and rows[0]['fixture_index']==0
    assert rows[0]['contact_identity_source']=='per-environment exact primitive filter column'


def test_fixture_dem_audit_detects_hidden_wall_as_flat():
    m=load();hits=np.array([[[1.1,0.,.8],[0,0,0]]])
    assert m.audit_fixture_rays(hits,[[0,0,0]],'wall')['passed']
    hits[0,0,2]=0
    assert not m.audit_fixture_rays(hits,[[0,0,0]],'wall')['passed']


def test_parent_difficulty_is_seed_local_and_stays_in_selected_level_band():
    m=load()
    a=m.parent_difficulty(123,4,(0.,1.))
    assert .8<=a<1.
    assert a==m.parent_difficulty(123,4,(0.,1.))
    assert a!=m.parent_difficulty(456,4,(0.,1.))
