import numpy as np

from visualnav_transformer.eat_nav.anchor_placement_geometry import footprint_diagnostic


def test_collision_proxy_reports_known_and_outside_without_safety_verdict():
    mask=np.ones((20,20),dtype=bool)
    mask[9,9]=False
    points={'main_track_sprocket':[(-.05,-.05),(.05,.05)],
            'initial_flipper_support':[(.2,0.0)],
            'articulated_reach':[(1.2,0.0)],
            'full_collision_union':[(-.05,-.05),(.05,.05),(.2,0.0),(1.2,0.0)]}
    result=footprint_diagnostic(mask,points,radius_m=.6,theta_rad=0.0)
    assert result['full_collision_union']['outside_dem_fraction']>0
    assert result['full_collision_union']['known_fraction']<1
    assert result['articulated_reach_m']==1.2
    assert result['tile_boundary_margin_m']>0
    assert 'safe' not in result
