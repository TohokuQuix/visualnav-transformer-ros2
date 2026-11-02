import math
import numpy as np
import pytest
from visualnav_transformer.eat_nav.moving_pose2s_geometry import (
    quat_to_matrix, matrix_to_quat, matrix_to_sixd, sixd_to_matrix,
    matrix_to_rpy, rotation_geodesic, relative_endpoint)


def root(p=(0,0,0), q=(1,0,0,0)):
    return list(p)+list(q)+[0.]*6


def test_full_body_relative_endpoint_and_first_two_columns():
    q=[math.cos(math.pi/8),0,math.sin(math.pi/8),0]
    R=quat_to_matrix(q)
    result=relative_endpoint(root(q=q),root(R @ [1,2,3],q))
    np.testing.assert_allclose(result['delta_position_body_m'],[1,2,3],atol=1e-12)
    np.testing.assert_allclose(result['delta_rotation_6d'],[1,0,0,0,1,0],atol=1e-12)
    np.testing.assert_allclose(matrix_to_sixd(R),np.r_[R[:,0],R[:,1]])
    restored,valid=sixd_to_matrix(matrix_to_sixd(R))
    assert valid
    np.testing.assert_allclose(restored,R,atol=1e-12)


def test_quaternion_sign_normalization_and_yaw_wrap():
    q=[0,0,0,-2.]
    np.testing.assert_allclose(matrix_to_quat(quat_to_matrix(q)),[0,0,0,1],atol=1e-12)
    a=quat_to_matrix([math.cos(1.56),0,0,math.sin(1.56)])
    b=quat_to_matrix([math.cos(-1.56),0,0,math.sin(-1.56)])
    assert rotation_geodesic(a,b)<.05
    np.testing.assert_allclose(a.T @ a,np.eye(3),atol=1e-12)


@pytest.mark.parametrize('q',[[0,0,0,0],[float('nan'),0,0,1],[1e-10,0,0,0]])
def test_invalid_quaternion_rejected(q):
    with pytest.raises(ValueError): quat_to_matrix(q)


def test_degenerate_sixd_invalid_and_gimbal_lock_explicit():
    R,valid=sixd_to_matrix([1,0,0,2,0,0])
    assert not valid
    np.testing.assert_equal(R,np.eye(3))
    R=quat_to_matrix([math.sqrt(.5),0,math.sqrt(.5),0])
    rpy,lock=matrix_to_rpy(R)
    assert lock
    assert rpy[1]==pytest.approx(math.pi/2)
    np.testing.assert_allclose(R.T @ [0,0,-1],[1,0,0],atol=1e-12)
