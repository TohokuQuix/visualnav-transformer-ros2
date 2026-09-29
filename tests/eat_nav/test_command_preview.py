import math
import numpy as np
import pytest
from visualnav_transformer.eat_nav.command_preview import pose_from_root_and_map_path,unicycle_step,map_to_body,body_to_map,commands_to_waypoints,command_error,macro_mean

def test_pose_and_frames():
 q=[math.cos(.35),0,0,math.sin(.35)]
 pose=pose_from_root_and_map_path([0,0,0,*q],[[2.,3.],[3.,3.]])
 assert np.allclose(pose,[2,3,.7])
 points=np.array([[2,3],[3,3],[3,4]])
 assert np.allclose(body_to_map(map_to_body(points,pose),pose),points,atol=1e-12)
 assert map_to_body([[2,4]],pose)[0,0]>0
 assert map_to_body([[2,4]],pose)[0,1]>0

def test_unicycle_and_trajectory():
 p=unicycle_step([0,0,0],[1,0,1],.05)
 assert p[0]>0 and p[1]>0 and p[2]==pytest.approx(.05)
 cmd=np.tile([.3,0.,0.],(40,1));w=commands_to_waypoints(cmd,dt=.05,stride=5,initial_yaw=.0)
 assert w.shape==(8,2) and np.allclose(w[:,0],np.arange(1,9)*.075)
 assert np.allclose(w[:,1],0)

def test_error_and_state_macro():
 a=np.zeros((40,3));b=np.zeros((40,3));b[5,2]=.1
 r=command_error(a,b,linear_threshold=.03,angular_threshold=.05)
 assert r['first_threshold_step']==5 and r['same_cluster'] is True
 assert macro_mean({'a':[1,1,1],'b':[0]})==.5

def test_future_inputs_rejected():
 import inspect
 from visualnav_transformer.eat_nav.command_preview import preview_candidate
 sig=inspect.signature(preview_candidate)
 assert set(sig.parameters)=={'client','map_path','initial_pose','steps','dt'}

def test_production_follower_first_command_fixtures():
 import json
 from pathlib import Path
 fixture=json.loads((Path(__file__).parent/'fixtures/command_preview_production.json').read_text())
 assert len(fixture['fixtures'])==7
 for row in fixture['fixtures']:
  assert np.max(np.abs(np.asarray(row['preview_command'])-np.asarray(row['production_command'])))<=row['tolerance']
