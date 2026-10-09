import pytest

from visualnav_transformer.eat_nav.p1_causal_intervention import applied_commands


def test_only_forward_command_changes_for_controlled_paths():
    normal=[[.20,0.,.4],[.18,0.,-.5],[.25,0.,0.],[0.,0.,.7],[.22,0.,-.4]]
    applied=applied_commands(normal,['plus_controlled','minus_controlled','direct','plus_controlled','minus_controlled'],.30)
    assert applied==[[.30,0.,.4],[.30,0.,-.5],[.25,0.,0.],[0.,0.,.7],[.30,0.,-.4]]


def test_intervention_rejects_out_of_limit_schedule():
    with pytest.raises(ValueError):
        applied_commands([[.2,0.,0.]],['plus_controlled'],.31)
