import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from visualnav_transformer.eat_nav.motion_14_predictor import Motion14Predictor
import visualnav_transformer.eat_nav.motion_14_predictor as model_source
from visualnav_transformer.eat_nav.demo_motion_predictor import load_demo_motion_predictor
from visualnav_transformer.eat_nav.demo_arc_candidates import DEMO_ARC_ROLES, DEMO_ARC_ANGLES
from visualnav_transformer.eat_nav.arc_bank import arc_waypoints
from visualnav_transformer.eat_nav.ros2_phase2.predictor_client import (
    CANDIDATE_IDS, PATHS, legacy_worker_request, prepare_model_input,
    worker_request)


PYTHON='/home/kojima/anaconda3/envs/tinyvla/bin/python'
LEGACY=Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01/training/full-seed456/best.pt')


class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.dem=np.zeros((4,2,20,20),np.float32); self.dem[:,1]=1
        self.state=np.zeros(22,np.float32)

    def test_legacy_seven_matches_existing_loader(self):
        if not LEGACY.exists(): self.skipTest('legacy checkpoint unavailable')
        motion,scores,sha=legacy_worker_request(PYTHON,LEGACY,self.dem,self.state,
                                                (0,0,0),(2,0))
        paths=np.asarray([arc_waypoints(DEMO_ARC_ANGLES[role])
                          for role in DEMO_ARC_ROLES],np.float32)
        original_motion,original_score=load_demo_motion_predictor(LEGACY).predict(
            self.dem,self.state,paths,(0,0,0),(2,0))
        np.testing.assert_allclose(motion,original_motion,atol=1e-6,rtol=0)
        np.testing.assert_allclose(scores,original_score,atol=1e-6,rtol=0)
        self.assertEqual(sha,hashlib.sha256(LEGACY.read_bytes()).hexdigest())

    def test_adopted_worker_requires_adoption_and_normalization(self):
        norm=dict(fitted_from='train_only',dem_height_mean=0.,dem_height_std=1.,
                  state_mean=[0.]*22,state_std=[1.]*22,
                  target_mean=[0.,0.],target_std=[1.,1.])
        source={key:'a'*64 for key in ('model_sha256','runner_sha256',
                                      'contract_sha256','dataset_ledger_sha256')}
        source['model_sha256']=hashlib.sha256(Path(model_source.__file__).read_bytes()).hexdigest()
        model=Motion14Predictor('full')
        with tempfile.TemporaryDirectory(prefix='eatnav-worker-',dir='/tmp') as root:
            root=Path(root)
            ckpt=root/'best.pt'; results=root/'results.json'
            torch.save(dict(model_state=model.state_dict(),variant='full',
                            candidate_count=14,candidate_order=CANDIDATE_IDS,
                            normalization=norm,source=source),ckpt)
            results.write_text(json.dumps({'demo_adoption':{'decision':'pass'},
                                           'source':source}))
            values=prepare_model_input(self.dem,self.state,PATHS,
                                       np.asarray([1]*7+[-1]*7,np.float32),norm)
            motion,digest=worker_request(PYTHON,ckpt,results,source,values,norm)
            self.assertEqual(motion.shape,(14,2))
            self.assertTrue(np.isfinite(motion).all())
            self.assertEqual(digest,hashlib.sha256(ckpt.read_bytes()).hexdigest())
            results.write_text(json.dumps({'demo_adoption':{'decision':'fail'},
                                           'source':source}))
            with self.assertRaises(RuntimeError):
                worker_request(PYTHON,ckpt,results,source,values,norm)


if __name__=='__main__': unittest.main()
