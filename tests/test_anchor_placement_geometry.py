import numpy as np
import ast
from pathlib import Path

from visualnav_transformer.eat_nav.anchor_placement_geometry import footprint_diagnostic


def test_scan_does_not_relabel_geometry_errors_as_insufficient_dem():
    source = Path(__file__).resolve().parents[1] / 'scripts/eat_nav_anchor_placement_scan_sim.py'
    tree = ast.parse(source.read_text())
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'scan_radius')
    catches = [node for node in ast.walk(fn) if isinstance(node, ast.Try) and any(
        isinstance(handler.type, ast.Name) and handler.type.id == 'ValueError' for handler in node.handlers)]
    assert len(catches) == 1
    caught_calls = [node.func.id for node in ast.walk(ast.Module(body=catches[0].body, type_ignores=[]))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert 'robot_dem' in caught_calls
    assert 'footprint_diagnostic' not in caught_calls
    assert 'classify_dem' not in caught_calls


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
