"""One-shot CPU PyTorch worker for adopted 14-slot checkpoints."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from visualnav_transformer.eat_nav.motion_14_predictor import Motion14Predictor
import visualnav_transformer.eat_nav.motion_14_predictor as model_source
from .predictor_client import validate_checkpoint_metadata, BANK_SHA, WIRE_SHA


def main():
    line = sys.stdin.readline(1_000_001)
    if len(line) >= 1_000_000:
        raise ValueError('worker input bound')
    req = json.loads(line)
    if req['schema'] != 'eatnav_ros2_predictor_request_v1':
        raise ValueError('worker schema')
    if req['mode'] == 'legacy6_demo':
        from visualnav_transformer.eat_nav.demo_motion_predictor import load_demo_motion_predictor
        from visualnav_transformer.eat_nav.demo_arc_candidates import DEMO_ARC_ROLES, DEMO_ARC_ANGLES
        from visualnav_transformer.eat_nav.arc_bank import arc_waypoints
        paths=np.asarray(req['paths'],dtype=np.float32)
        expected=np.asarray([arc_waypoints(DEMO_ARC_ANGLES[role])
                             for role in DEMO_ARC_ROLES],dtype=np.float32)
        if paths.shape!=(7,8,2) or not np.array_equal(paths,expected):
            raise ValueError('legacy seven-forward bank mismatch')
        predictor=load_demo_motion_predictor(req['checkpoint'])
        motion,scores=predictor.predict(np.asarray(req['dem'],dtype=np.float32),
            np.asarray(req['state'],dtype=np.float32),paths,
            req['robot_pose_map'],req['goal_map'])
        print(json.dumps({'motion_xy':motion.tolist(), 'scores_m':scores.tolist(),
                          'checkpoint_sha256':predictor.checkpoint_sha256}))
        return
    if req['schema'] != 'eatnav_ros2_predictor_request_v1' or \
            req['mode'] != 'learned14_adopted' or \
            req['bank_sha256'] != BANK_SHA or req['wire_sha256'] != WIRE_SHA:
        raise ValueError('worker mode/bank mismatch')
    checkpoint = Path(req['checkpoint'])
    results = json.loads(Path(req['results']).read_text())
    payload = torch.load(checkpoint,map_location='cpu',weights_only=True)
    validate_checkpoint_metadata(payload,results,
                                 expected_source=req['expected_source'])
    if req['normalization'] != payload['normalization']:
        raise ValueError('model normalization provenance mismatch')
    if payload['source']['model_sha256'] != hashlib.sha256(
            Path(model_source.__file__).read_bytes()).hexdigest():
        raise ValueError('model source SHA differs from trained checkpoint')
    arrays = [np.asarray(value,dtype=np.float32) for value in req['arrays']]
    if [v.shape for v in arrays] != [(1,4,2,20,20),(1,22),(1,14,8,2),
                                     (1,14,1),(1,14)]:
        raise ValueError('worker tensor shapes')
    arrays[-1] = arrays[-1].astype(bool)
    model = Motion14Predictor('full')
    model.load_state_dict(payload['model_state'],strict=True)
    model.eval()
    with torch.no_grad():
        scaled = model(*(torch.from_numpy(a) for a in arrays))[0].numpy()
    if not np.isfinite(scaled).all():
        raise ValueError('model output nonfinite')
    print(json.dumps({'scaled_xy':scaled.tolist(),
                      'normalization':payload['normalization'],
                      'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
