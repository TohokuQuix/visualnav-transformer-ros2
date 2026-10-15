"""Torch CPU process with validated once-per-process checkpoint load."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from visualnav_transformer.eat_nav.arc_bank import arc_waypoints
from visualnav_transformer.eat_nav.demo_arc_candidates import DEMO_ARC_ROLES, DEMO_ARC_ANGLES
from visualnav_transformer.eat_nav.demo_motion_predictor import load_demo_motion_predictor
from visualnav_transformer.eat_nav.motion_14_predictor import Motion14Predictor
from visualnav_transformer.eat_nav.heading_predictor import load_heading_model
import visualnav_transformer.eat_nav.motion_14_predictor as model_source
from .predictor_client import (validate_checkpoint_metadata,
                               CANDIDATE_IDS, PATHS)


def _emit(value):
    print(json.dumps(value,separators=(',',':')),flush=True)


def _serve():
    first=sys.stdin.readline(1_000_001)
    init=json.loads(first)
    if init.get('op')!='init': raise ValueError('worker init required')
    mode=init['mode']; checkpoint=Path(init['checkpoint'])
    if mode=='legacy6_demo':
        predictor=load_demo_motion_predictor(checkpoint)
        paths=np.asarray([arc_waypoints(DEMO_ARC_ANGLES[role])
                          for role in DEMO_ARC_ROLES],np.float32)
        digest=predictor.checkpoint_sha256
    elif mode=='learned14_adopted':
        results=json.loads(Path(init['results']).read_text())
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        validate_checkpoint_metadata(payload,results,expected_source=results['source'])
        if payload['normalization'] != init['normalization'] or \
                payload['source']['model_sha256'] != hashlib.sha256(
                    Path(model_source.__file__).read_bytes()).hexdigest():
            raise ValueError('adopted source/normalization mismatch')
        predictor=Motion14Predictor('full')
        predictor.load_state_dict(payload['model_state'],strict=True)
        predictor.eval()
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if init.get('heading_checkpoint'):
            predictor=load_heading_model(predictor,payload['normalization'],
                                         digest,init['heading_checkpoint'])
    else:
        raise ValueError('worker mode unsupported')
    _emit({'status':'ready','checkpoint_sha256':digest})
    for line in sys.stdin:
        if len(line)>1_000_000: raise ValueError('worker request too large')
        request=json.loads(line)
        if request.get('op')=='stop': break
        if request.get('op')!='predict': raise ValueError('unknown worker operation')
        if mode=='legacy6_demo':
            motion,scores=predictor.predict(np.asarray(request['dem'],np.float32),
                np.asarray(request['state'],np.float32),paths,
                request['robot_pose_map'],request['goal_map'])
            _emit({'motion_xy':motion.tolist(),'scores_m':scores.tolist()})
        else:
            arrays=[np.asarray(value,np.float32) for value in request['arrays']]
            if [v.shape for v in arrays] != [(1,4,2,20,20),(1,22),(1,14,8,2),
                                             (1,14,1),(1,14)] or \
                    not np.array_equal(arrays[2][0],PATHS) or \
                    not np.array_equal(arrays[3][0,:,0],[1]*7+[-1]*7) or \
                    not np.array_equal(arrays[4],np.ones((1,14),np.float32)):
                raise ValueError('adopted model tensor/bank contract')
            arrays[-1]=arrays[-1].astype(bool)
            with torch.no_grad():
                output=predictor(*(torch.from_numpy(a) for a in arrays))[0].numpy()
            if not np.isfinite(output).all(): raise ValueError('model nonfinite')
            if output.shape == (14,3):
                _emit({'scaled_xy':output[:,:2].tolist(),
                       'yaw_rad':output[:,2].tolist()})
            else:
                _emit({'scaled_xy':output.tolist()})


if __name__=='__main__':
    try: _serve()
    except Exception as error:
        _emit({'error':str(error)})
        sys.exit(1)
