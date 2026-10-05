import hashlib
import json

import numpy as np
import pytest

from eat_nav_expanded_feasibility_run import (
    execution_entry, freeze_map_paths, router_identity, validate_anchor_limit_contract,
)


def test_execution_entry_keeps_frozen_order_and_role():
    row={'order':0,'process_id':'expanded-train-000-anchor_only-00',
         'role':'anchor_only','path_id':'direct_reference','state_index':0,
         'state_group_id':'expanded-train-000','episode_key':[0,36,0],
         'robot_local_waypoint_sha256':'local','base_path_file_sha256':'file','seed':123}
    state={'step':20,'episode_key':[0,36,0],'state_id':'expanded-train-000'}
    entry=execution_entry(row,state,map_digest='pending_anchor_only')
    assert entry['kind']=='anchor'
    assert entry['candidate_id']==row['process_id']
    assert entry['logical_t0_ns']==2_300_000_000_000
    assert entry['map_path_sha256']=='pending_anchor_only'


def test_freeze_map_paths_uses_anchor_pose_and_preserves_direct(tmp_path):
    root=tmp_path
    folder=root/'inputs/state-000';folder.mkdir(parents=True)
    base=[[0.,0.]]+[[i*.1,0.] for i in range(1,9)]
    lateral=[[x,y+.05*(i/8)] for i,(x,y) in enumerate(base)]
    (folder/'direct_reference-base-path.json').write_text(json.dumps(base))
    (folder/'left_1-base-path.json').write_text(json.dumps(lateral))
    anchor_map=[[1+x,2+y] for x,y in base]
    rows=[{'process_id':'anchor','path_id':'direct_reference','state_index':0,'role':'anchor_only',
           'base_path_file_sha256':'x','robot_local_waypoint_sha256':'x','state_group_id':'expanded-train-000'},
          {'process_id':'direct','path_id':'direct_reference','state_index':0,'role':'nominal_duplicate',
           'base_path_file_sha256':'x','robot_local_waypoint_sha256':'x','state_group_id':'expanded-train-000'},
          {'process_id':'left','path_id':'left_1','state_index':0,'role':'candidate_outcome',
           'base_path_file_sha256':'x','robot_local_waypoint_sha256':'x','state_group_id':'expanded-train-000'}]
    for row in rows:
        path=folder/f"{row['path_id']}-base-path.json"
        row['base_path_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        row['robot_local_waypoint_sha256']=hashlib.sha256(
            np.asarray(json.loads(path.read_text())[1:],dtype=np.float32).tobytes()).hexdigest()
    result=freeze_map_paths(root,0,rows,anchor_map,
        {'root_state':[11.,12.,0.,1.,0.,0.,0.]},
        logged_goal=[.8,0.],amplitudes={'direct_reference':0.,'left_1':.05})
    assert result['status']=='pass'
    assert json.loads((folder/'direct-map-path.json').read_text())==anchor_map
    assert json.loads((folder/'left-map-path.json').read_text())[-1][1]>2.
    assert json.loads((folder/'left-goal-metadata.json').read_text())['state_id']=='expanded-train-000'
    with pytest.raises(FileExistsError):
        freeze_map_paths(root,0,rows,anchor_map,{'root_state':[11.,12.,0.,1.,0.,0.,0.]},
            logged_goal=[.8,0.],amplitudes={'direct_reference':0.,'left_1':.05})


def test_router_identity_comes_from_new_root_not_prior_pid(tmp_path):
    root=tmp_path/'root';root.mkdir()
    proc_root=tmp_path/'proc';proc=(proc_root/'456');proc.mkdir(parents=True)
    (proc/'stat').write_text('456 (rmw_zenohd) S '+' '.join(['1']*18+['789']+['1']*30))
    (proc/'exe').symlink_to('/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd')
    (root/'external-router-provenance.json').write_text(json.dumps({
        'router_identity':{'pid':456,'start_ticks':789,
            'binary':'/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd'}}))
    assert router_identity(root,proc_root=proc_root)['pid']==456


def test_anchor_config_contract_fails_offline_if_original_e1_keys_missing(tmp_path):
    state=tmp_path/'inputs/state-000';state.mkdir(parents=True)
    (state/'config-120.yaml').write_text('anchor_limits:\n  position_m: 0.02\n  orientation_rad: 0.03\n  dem_rms_m: 0.03\n  mask_iou_min: 0.95\n')
    reference={'position_m':.02,'orientation_rad':.03,'dem_rms_m':.03,
        'mask_iou_min':.95,'velocity_max_abs':1e-4,'command_max_abs':1e-5}
    with pytest.raises(ValueError,match='missing.*command_max_abs.*velocity_max_abs'):
        validate_anchor_limit_contract(tmp_path,1,reference)
    (state/'config-120.yaml').write_text('anchor_limits:\n  position_m: 0.02\n  orientation_rad: 0.03\n  dem_rms_m: 0.03\n  mask_iou_min: 0.95\n  velocity_max_abs: 0.0001\n  command_max_abs: 0.00001\n')
    validate_anchor_limit_contract(tmp_path,1,reference)
    (state/'config-120.yaml').write_text('anchor_limits:\n  position_m: 0.03\n  orientation_rad: 0.03\n  dem_rms_m: 0.03\n  mask_iou_min: 0.95\n  velocity_max_abs: 0.0001\n  command_max_abs: 0.00001\n')
    with pytest.raises(ValueError,match="different=\\['position_m'\\]"):
        validate_anchor_limit_contract(tmp_path,1,reference)
