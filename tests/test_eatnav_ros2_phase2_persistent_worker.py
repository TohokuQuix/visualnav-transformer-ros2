import unittest
from pathlib import Path
import json
import hashlib
import tempfile
import subprocess

import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.persistent_worker import ModelWorkerClient
from visualnav_transformer.eat_nav.ros2_phase2.predictor_client import CANDIDATE_IDS, PATHS


PYTHON='/home/kojima/anaconda3/envs/tinyvla/bin/python'
LEGACY=Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01/training/full-seed456/best.pt')


class PersistentWorkerTest(unittest.TestCase):
    def test_two_legacy_requests_same_process(self):
        if not LEGACY.exists(): self.skipTest('legacy checkpoint unavailable')
        dem=np.zeros((4,2,20,20),np.float32); dem[:,1]=1
        with ModelWorkerClient(PYTHON,'legacy6_demo',LEGACY) as client:
            pid=client.process.pid
            first=client.predict_legacy(dem,np.zeros(22,np.float32),(0,0,0),(2,0))
            second=client.predict_legacy(dem,np.zeros(22,np.float32),(0,0,0),(2,0))
            self.assertEqual(client.process.pid,pid)
            np.testing.assert_array_equal(first[0],second[0])
            np.testing.assert_array_equal(first[1],second[1])
            self.assertEqual(first[0].shape,(7,2))
        self.assertIsNotNone(client.process.poll())

    def test_synthetic_adopted_worker_requires_sealed_provenance(self):
        repo=Path(__file__).resolve().parents[1]
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp); (root/'training/full-seed1').mkdir(parents=True)
            norm={'fitted_from':'train_only','dem_height_mean':0.,'dem_height_std':1.,
                  'state_mean':[0.]*22,'state_std':[1.]*22,
                  'target_mean':[0.,0.],'target_std':[1.,1.]}
            (root/'normalization.json').write_text(json.dumps(norm))
            (root/'contract.json').write_text(json.dumps(
                {'candidate_count':14,'candidate_order':CANDIDATE_IDS}))
            for name in ('metadata.json','audit.json','provenance.json'):
                (root/name).write_text('{}')
            (root/'test-predictions.npz').write_bytes(b'test')
            from numpy import savez
            savez(root/'grouped.npz',paths=PATHS[None],
                  direction=np.asarray([1]*7+[-1]*7,np.float32).reshape(1,14,1),
                  mask=np.ones((1,14),bool))
            names=('grouped.npz','metadata.json','audit.json','contract.json','provenance.json')
            (root/'dataset-ledger.json').write_text(json.dumps({n:sha(root/n) for n in names}))
            source={'model_sha256':sha(repo/'src/visualnav_transformer/eat_nav/motion_14_predictor.py'),
                    'runner_sha256':sha(repo/'scripts/eat_nav_motion_14_train.py'),
                    'contract_sha256':sha(root/'contract.json'),
                    'dataset_ledger_sha256':sha(root/'dataset-ledger.json')}
            (root/'training/source.json').write_text(json.dumps(source))
            ckpt=root/'training/full-seed1/best.pt'
            code=('import torch; from visualnav_transformer.eat_nav.motion_14_predictor '
                  'import Motion14Predictor; import sys,json; '
                  'torch.save(dict(model_state=Motion14Predictor("full").state_dict(),'
                  'variant="full",candidate_count=14,candidate_order=tuple(json.loads(sys.argv[2])),'
                  'normalization=json.loads(sys.argv[3]),source=json.loads(sys.argv[4])),sys.argv[1])')
            subprocess.run([PYTHON,'-c',code,str(ckpt),json.dumps(CANDIDATE_IDS),
                            json.dumps(norm),json.dumps(source)],check=True)
            results={'demo_adoption':{'decision':'pass','contract_sha256':sha(
                repo/'docs/eat-nav/motion-predictor-vnext/motion_14_training_contract.json')},
                'source':source,'selected_full_seed':1,
                'variants':{'full':{'1':{'checkpoint_sha256':sha(ckpt)}}}}
            (root/'results.json').write_text(json.dumps(results))
            (root/'results-ledger.json').write_text(json.dumps({n:sha(root/n) for n in
                ('results.json','normalization.json','test-predictions.npz')}))
            dem=np.zeros((4,2,20,20),np.float32); dem[:,1]=1
            with ModelWorkerClient(PYTHON,'learned14_adopted',ckpt,
                                   results=root/'results.json',
                                   normalization=root/'normalization.json') as client:
                first=client.predict_adopted(dem,np.zeros(22,np.float32))[0]
                second=client.predict_adopted(dem,np.zeros(22,np.float32))[0]
                self.assertEqual(first.shape,(14,2))
                np.testing.assert_array_equal(first,second)


if __name__=='__main__': unittest.main()
