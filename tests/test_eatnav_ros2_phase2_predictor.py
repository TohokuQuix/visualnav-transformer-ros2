import unittest
import json
import tempfile
from pathlib import Path
import hashlib
import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.selection import score_global_goal
from visualnav_transformer.eat_nav.ros2_phase2.predictor_client import (
    validate_mode, validate_checkpoint_metadata, prepare_model_input,
    mock14_predict, inverse_target, validate_adoption_artifacts,
)
from visualnav_transformer.eat_nav.ros2_phase2.wire import CANDIDATE_IDS, PATHS


class PredictorContractTest(unittest.TestCase):
    def setUp(self):
        self.dem = np.zeros((4, 2, 20, 20), np.float32)
        self.dem[:, 1] = 1
        self.dem[0, 0, 0, 0] = -.4
        self.state = np.arange(22, dtype=np.float32)
        self.norm = dict(fitted_from='train_only', dem_height_mean=.1,
                         dem_height_std=.2, state_mean=[0.]*22,
                         state_std=[2.]*22, target_mean=[.2, -.1],
                         target_std=[.5, .25])

    def test_modes_do_not_cross_candidate_contracts(self):
        validate_mode('mock14_transport', 14, None)
        with self.assertRaises(ValueError):
            validate_mode('legacy6_demo', 14, None)
        with self.assertRaises(ValueError):
            validate_mode('learned14_adopted', 14, None)
        with self.assertRaises(ValueError):
            validate_mode('mock14_transport', 6, None)

    def test_normalization_and_inverse(self):
        values = prepare_model_input(self.dem, self.state, PATHS,
                                     np.array([1]*7+[-1]*7, np.float32), self.norm)
        self.assertEqual([x.shape for x in values],
                         [(1,4,2,20,20),(1,22),(1,14,8,2),(1,14,1),(1,14)])
        self.assertAlmostEqual(float(values[0][0,0,0,0,0]), -2.5)
        self.assertEqual(float(values[1][0,2]), 1.)
        np.testing.assert_array_equal(inverse_target(np.zeros((14,2),np.float32),self.norm),
                                      np.tile(np.asarray([.2,-.1],np.float32),(14,1)))

    def test_checkpoint_requires_adoption_and_frozen_order(self):
        meta = dict(variant='full', candidate_count=14,
                    candidate_order=CANDIDATE_IDS, normalization=self.norm,
                    source={k:'a'*64 for k in ('model_sha256','runner_sha256',
                                              'contract_sha256','dataset_ledger_sha256')})
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(meta, {'demo_adoption':{'decision':'fail'}},
                                         expected_source=meta['source'])
        validate_checkpoint_metadata(meta, {'demo_adoption':{'decision':'pass'}},
                                     expected_source=meta['source'])
        meta['candidate_order'] = tuple(reversed(CANDIDATE_IDS))
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(meta, {'demo_adoption':{'decision':'pass'}},
                                         expected_source=meta['source'])

    def test_mock_selection_goal_and_tie(self):
        motion = mock14_predict(PATHS)
        self.assertEqual(motion.shape,(14,2))
        selection = score_global_goal(motion, (0.,0.,0.), (2.,0.), np.ones(14,bool))
        self.assertEqual(selection['index'], 0)
        tied = score_global_goal(np.zeros((14,2),np.float32), (0.,0.,0.), (2.,0.), np.ones(14,bool))
        self.assertEqual(tied['index'], 0)
        with self.assertRaises(ValueError):
            score_global_goal(motion,(0.,0.,0.),(2.,0.),np.zeros(14,bool))

    def test_adoption_requires_sealed_results_and_selected_checkpoint(self):
        repo=Path(__file__).resolve().parents[1]
        sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            root=Path(tmp)
            (root/'training/full-seed1').mkdir(parents=True)
            np.savez(root/'grouped.npz',paths=PATHS[None],
                     direction=np.asarray([1]*7+[-1]*7,np.float32).reshape(1,14,1),
                     mask=np.ones((1,14),bool))
            (root/'contract.json').write_text(json.dumps(
                {'candidate_count':14,'candidate_order':CANDIDATE_IDS}))
            for name in ('metadata.json','audit.json','provenance.json'):
                (root/name).write_text('{}')
            (root/'test-predictions.npz').write_bytes(b'predictions')
            ckpt=root/'training/full-seed1/best.pt'; ckpt.write_bytes(b'checkpoint')
            norm=root/'normalization.json'; norm.write_text(json.dumps(self.norm))
            source=dict(model_sha256=sha(repo/'src/visualnav_transformer/eat_nav/motion_14_predictor.py'),
                        runner_sha256=sha(repo/'scripts/eat_nav_motion_14_train.py'),
                        contract_sha256=sha(root/'contract.json'),dataset_ledger_sha256='')
            dataset={name:sha(root/name) for name in ('grouped.npz','metadata.json',
                     'audit.json','contract.json','provenance.json')}
            (root/'dataset-ledger.json').write_text(json.dumps(dataset))
            source['dataset_ledger_sha256']=sha(root/'dataset-ledger.json')
            (root/'training/source.json').write_text(json.dumps(source))
            result={'demo_adoption':{'decision':'pass','contract_sha256':sha(
                repo/'docs/eat-nav/motion-predictor-vnext/motion_14_training_contract.json')},
                'source':source,'selected_full_seed':1,
                'variants':{'full':{'1':{'checkpoint_sha256':sha(ckpt)}}}}
            (root/'results.json').write_text(json.dumps(result))
            (root/'results-ledger.json').write_text(json.dumps(
                {name:sha(root/name) for name in ('results.json','normalization.json',
                                                 'test-predictions.npz')}))
            checked,_=validate_adoption_artifacts(root/'results.json',ckpt)
            self.assertEqual(checked['selected_full_seed'],1)
            with self.assertRaises(ValueError):
                validate_adoption_artifacts(root/'results.json',root/'wrong.pt')
            (root/'results.json').write_text(json.dumps({**result,'selected_full_seed':2}))
            with self.assertRaises(ValueError):
                validate_adoption_artifacts(root/'results.json',ckpt)


if __name__ == '__main__':
    unittest.main()
