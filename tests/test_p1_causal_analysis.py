import math
import pytest

from scripts.eat_nav_p1_causal_analyze import world_to_body, predicted_follower_forward


def test_world_velocity_uses_full_quaternion_rotation():
    q=(math.sqrt(.5),0.,0.,math.sqrt(.5))
    assert world_to_body(q,(1.,0.,0.))==pytest.approx((0.,-1.,0.))


def test_follower_heading_error_speed_scale_and_turn_stop():
    assert predicted_follower_forward(.2)==pytest.approx(.3*(1-.4/math.pi))
    assert predicted_follower_forward(.4)==0.
