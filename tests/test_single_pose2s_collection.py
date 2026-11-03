import numpy as np
import pytest
from visualnav_transformer.eat_nav.single_pose2s_collection import window_endpoints, acquisition_steps, inpaint_frame, causal_dem_context


def test_exact_endpoints_and_early_reset_mask():
    stamps=np.arange(201)*50000000
    alive=np.ones(201,bool);alive[67:]=False
    ends=window_endpoints(stamps,alive)
    assert ends==[(0,40,True),(40,80,False),(80,120,False),(120,160,False),(160,200,False)]
    stamps[40]+=1
    with pytest.raises(ValueError):window_endpoints(stamps,alive)


def test_dem_phase_is_per_parent_not_vector_batch():
    assert acquisition_steps(3)==[0]+list(range(3,201,5))
    assert acquisition_steps(0)==list(range(0,201,5))


def test_inpaint_does_not_label_missing_as_observed():
    h=np.full((40,40),2.);raw=np.ones((40,40),bool);raw[10:20,10:20]=False
    dem=inpaint_frame(h,raw,1.)
    assert dem.shape==(3,40,40)
    assert dem[1].all() and dem[2,15,15]==0 and dem[0,15,15]==1
    with pytest.raises(ValueError):inpaint_frame(h,np.zeros((40,40),bool),1.)
    c=causal_dem_context(np.array([0,0,0,1,0,0,0]),np.array([.1,0,0,1,0,0,0]),0,100000000)
    np.testing.assert_allclose(c,[.1,1,-.1,0,0,1])


def test_terminal_exactly_at_two_seconds_keeps_pre_reset_endpoint():
    alive=np.ones(201,bool);alive[40:]=False
    result=window_endpoints(np.arange(201)*50000000,alive)
    assert result[0][2] is True and result[1][2] is False


def test_spawn_clearance_uses_actual_raised_surface_not_world_zero():
    from visualnav_transformer.eat_nav.single_pose2s_collection import terrain_clearance_z
    vertices=np.array([[-2,-2,.4],[2,-2,.4],[2,2,.4],[-2,2,.4]])
    faces=np.array([[0,1,2],[0,2,3]])
    assert terrain_clearance_z(vertices,faces,[0,0],.35)==pytest.approx(.75)


def test_thirty_env_batches_preserve_parent_splits_and_skip_completed():
    from visualnav_transformer.eat_nav.single_pose2s_collection import pending_parent_batches
    from visualnav_transformer.eat_nav.single_pose2s_contract import make_manifest
    m=make_manifest(123);completed={p['parent_id'] for p in m['parents'][:10]}
    batches=pending_parent_batches(m,completed,30)
    ids=[p['parent_id'] for batch in batches for p in batch]
    assert len(ids)==1320 and len(set(ids))==1320
    assert not completed.intersection(ids)
    assert any(len(batch)==30 for batch in batches)
    assert all(1<=len(batch)<=30 for batch in batches)
    assert all(len({(p['field'],p['level']) for p in batch})==1 for batch in batches)
    original={p['parent_id']:p for p in m['parents']}
    assert all(p==original[p['parent_id']] for batch in batches for p in batch)
    with pytest.raises(ValueError):pending_parent_batches(m,set(),0)
    with pytest.raises(ValueError):pending_parent_batches(m,{'unknown-parent'},30)
