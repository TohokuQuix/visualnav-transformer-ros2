import copy
import ast
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.anchor_placement_parity import compare_dem_frame


def frame():
    return {'pose_id':'p','env_id':5,'terrain':'boxes','terrain_seed':123,
            'frame_id':'map','axis':'robot-relative-grid-xy','resolution_m':0.1,
            'shape':[2,20,20],'physics_step':200,'source_origin_xy':[1.0,2.0],
            'source_height':np.ones((20,20)).tolist(),
            'source_mask':np.ones((20,20),dtype=int).tolist(),
            'transformed':np.stack((np.zeros((20,20)),np.ones((20,20)))).tolist(),
            'yaw_rad':0.4,'env_origin_xyz':[3.0,4.0,0.0]}


def test_identical_geometry_and_dem_pass():
    result=compare_dem_frame(frame(),frame())
    assert result['pass']
    assert result['transformed_height_rms_m']==0
    assert result['transformed_mask_iou']==1


def test_height_error_and_mask_error_fail_independently():
    expected=frame();actual=copy.deepcopy(expected)
    actual['transformed'][0][0][0]=0.001
    assert not compare_dem_frame(expected,actual)['pass']
    actual=copy.deepcopy(expected);actual['source_mask'][0][0]=0
    assert not compare_dem_frame(expected,actual)['pass']


def test_identity_and_grid_contract_fail_exactly():
    expected=frame();actual=copy.deepcopy(expected)
    actual['terrain_seed']=124
    assert 'terrain_seed' in compare_dem_frame(expected,actual)['identity_mismatches']
    actual=copy.deepcopy(expected);actual['resolution_m']=0.10001
    assert 'resolution_m' in compare_dem_frame(expected,actual)['identity_mismatches']


def test_terrain_generator_methods_match_frozen_inventory_source():
    root=Path(__file__).resolve().parents[1]
    def methods(path):
        tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='RadialTerrainGenerator')
        return {x.name:ast.dump(x,include_attributes=False) for x in cls.body if isinstance(x,ast.FunctionDef)}
    assert methods(root/'scripts/eat_nav_anchor_placement_parity_sim.py')==methods(root/'scripts/eat_nav_radial_five_pose_sim.py')
    assert methods(root/'scripts/eat_nav_anchor_placement_scan_sim.py')==methods(root/'scripts/eat_nav_radial_five_pose_sim.py')
