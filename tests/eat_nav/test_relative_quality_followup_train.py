"""Scratch, validation-selected and resumable follow-up Probe training."""
import tempfile
import unittest
from pathlib import Path
import torch
from visualnav_transformer.eat_nav.relative_quality_followup_train import train_followup_probe, predict_followup


def data(n=4):
    return {'dem':torch.zeros((n,4,2,20,20)),'goal':torch.zeros((n,2)),
            'action':torch.zeros((n,8,2)),'mask':torch.ones((n,8),dtype=torch.bool)}


class FollowupTrainTest(unittest.TestCase):
    def test_validation_selected_checkpoint_and_fingerprint_guard(self):
        train=data();val=data();q=torch.ones((4,2));settings={
            'probe_learning_rate':.0003,'probe_max_epochs':2,'probe_min_delta':.0001,
            'probe_batch_size':2,'probe_patience':4}
        with tempfile.TemporaryDirectory() as temp:
            result=train_followup_probe(train,val,q,q,kind='probe_delta_drawdown',seed=11,
                                        settings=settings,output=Path(temp),fingerprint={'data':'a'})
            self.assertTrue(Path(result['checkpoint']).exists())
            self.assertIn(result['best_epoch'],(1,2))
            saved=torch.load(result['checkpoint'],map_location='cpu',weights_only=True)
            self.assertEqual(saved['epoch'],result['best_epoch'])
            self.assertEqual(saved['fingerprint'],{'data':'a'})
            self.assertEqual(tuple(predict_followup(result['checkpoint'],val,q,2).shape),(4,8,2))
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                train_followup_probe(train,val,q,q,kind='probe_delta_drawdown',seed=11,
                                     settings=settings,output=Path(temp),fingerprint={'data':'b'})

if __name__=='__main__':unittest.main()
