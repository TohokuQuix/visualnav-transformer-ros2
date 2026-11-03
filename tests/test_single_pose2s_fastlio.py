import numpy as np
import pytest


def test_pose_velocity_is_body_frame_causal_and_not_zero_twist():
    from visualnav_transformer.eat_nav.single_pose2s_fastlio import odom_velocity_at
    # body points toward world+y; world motion+y must be forward body+x
    q=[np.sqrt(.5),0,0,np.sqrt(.5)]
    poses=np.array([[0,0,0,*q],[0,.02,0,*q],[0,99,0,*q]])
    v,w,age,ids=odom_velocity_at(100000000,[0,100000000,200000000],[0,100000000,200000000],poses)
    np.testing.assert_allclose(v,[.2,0,0],atol=1e-8)
    np.testing.assert_allclose(w,0,atol=1e-8)
    assert ids==[0,1] and age==0
    # Not yet received: no silently substituted truth/wheel velocity.
    with pytest.raises(ValueError,match='two causal'):
        odom_velocity_at(100000000,[0,100000000],[0,100000001],poses[:2])
    with pytest.raises(ValueError,match='stale'):
        odom_velocity_at(400000001,[0,100000000],[0,100000000],poses[:2])


def test_imu_specific_force_keeps_stationary_gravity_and_no_future():
    from visualnav_transformer.eat_nav.single_pose2s_fastlio import imu_specific_force
    root=np.zeros(13);root[3]=1
    np.testing.assert_allclose(imu_specific_force(root,root,.05),[0,0,9.81])
    current=root.copy();current[7]=.1
    np.testing.assert_allclose(imu_specific_force(root,current,.05),[2,0,9.81])
