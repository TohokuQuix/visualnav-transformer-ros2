import math
import numpy as np
import pytest
from visualnav_transformer.eat_nav.single_pose2s_contract import make_manifest, validate_input, INPUT_SCHEMA
from visualnav_transformer.eat_nav.single_pose2s_geometry import path_errors, endpoint_target
from visualnav_transformer.eat_nav.single_pose2s_follower import NominalFollower
from visualnav_transformer.eat_nav.bidirectional_bank import bank


def valid():
    state=np.zeros(27);state[2]=-1;state[25]=1
    return dict(schema=INPUT_SCHEMA,dem=np.stack([np.zeros((40,40)),np.ones((40,40)),np.ones((40,40))]),state=state,state_mask=np.ones(27),state_age=np.zeros(27),dem_context=np.array([0.,1.,0.,0.,0.,1.]),t0_ns=250000000,dem_stamp_ns=250000000,dem_received_ns=250000000,paths=np.stack([x.waypoints for x in bank()]),directions=np.array([[x.direction] for x in bank()]))


def test_plan_counts_and_independent_parents():
    m=make_manifest(123)
    assert len(m['parents'])==1330
    assert len({p['parent_id'] for p in m['parents']})==1330
    assert len({p['seed'] for p in m['parents']})==1330
    for role,n in [('train',350),('validation',490),('final_test',490)]:
        assert sum(p['role']==role for p in m['parents'])==n
    for f in m['fields']:
        for l in range(5):
            rows=[p for p in m['parents'] if p['field']==f and p['level']==l and p['role']=='final_test']
            for cid in range(14):
                found=[(p['parent_id'],j) for p in rows for j,c in enumerate(p['candidates']) if c==cid]
                assert len(found)==5 and len({p for p,j in found})==5 and len({j for p,j in found})==5


def test_inputs_single_frame_and_future_rejection():
    a=valid();a['dem'][1]=1;validate_input(a)
    a['dem']=np.zeros((4,3,40,40))
    with pytest.raises(ValueError):validate_input(a)
    a=valid();a['dem_received_ns']=a['t0_ns']+1
    with pytest.raises(ValueError):validate_input(a)
    a=valid();a['dem_stamp_ns']=a['t0_ns']-500000001
    with pytest.raises(ValueError):validate_input(a)
    a=valid();a['state_age'][0]=.201
    with pytest.raises(ValueError):validate_input(a)


def test_endpoint_rotation_and_reverse_path_basis():
    a=np.array([0.,0,0,1,0,0,0]);b=np.array([.2,.1,.05,1,0,0,0])
    t=endpoint_target(a,b)
    np.testing.assert_allclose(t['position'],[.2,.1,.05])
    e=path_errors([-.3,.1,0],[-.2,0,0],[1,0,0,0],7)
    assert e['longitudinal_m']==pytest.approx(.1)
    assert e['lateral_m']==pytest.approx(-.1)
    e=path_errors([.2,.1,.03],[.2,0,0],[1,0,0,0],0)
    assert e['longitudinal_m']==pytest.approx(0.)
    assert e['lateral_m']==pytest.approx(.1)


def test_nominal_follower_does_not_recover_when_stationary():
    f=NominalFollower(bank()[0].waypoints)
    for i in range(500):
        v,vy,w=f.step(0.,0.,0.)
        assert v==pytest.approx(.3) and vy==0 and w==pytest.approx(0)
    r=NominalFollower(bank()[7].waypoints)
    assert r.step(0.,0.,0.)[0]==pytest.approx(-.3)


def test_bad_gravity_mode_and_nonbank_path_rejected():
    a=valid();a['state'][:3]=0
    with pytest.raises(ValueError):validate_input(a)
    a=valid();a['state'][24:]=0
    with pytest.raises(ValueError):validate_input(a)
    a=valid();a['paths'][0,-1,0]=42
    with pytest.raises(ValueError):validate_input(a)


def test_nominal_curve_endpoint_extension_uses_legacy_end_tangent():
    f=NominalFollower(bank()[5].waypoints)
    cmd=f.step(.19514863273861344,-.1532974059593647,-2.030677995822562)
    np.testing.assert_allclose(cmd,[0.,0.,.5739419673092182],atol=1e-9)


def test_cpp_nominal_stateful_direction_parity(tmp_path):
    from visualnav_transformer.eat_nav.p1_follower_core import PureFollower,compile_core
    library=compile_core(tmp_path)
    for c in bank():
        old=PureFollower(c.waypoints.tolist(),library=library);new=NominalFollower(c.waypoints)
        try:
            for step,yaw in enumerate([0.,np.pi,1.5,1.3,1.8]):
                np.testing.assert_allclose(old.step(0.,0.,yaw,step*50000000),new.step(0.,0.,yaw),atol=1e-9)
        finally:old.close()
