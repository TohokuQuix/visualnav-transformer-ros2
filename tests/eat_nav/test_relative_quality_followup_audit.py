"""Frozen label and feature alignment must fail on split/window mismatch."""
import unittest
import numpy as np
from visualnav_transformer.eat_nav.relative_quality_followup_audit import validate_aligned_record


class FollowupAuditTest(unittest.TestCase):
    def test_aligned_record_matches_index_episode_and_labels(self):
        row={'split':'train','window_index':2,'episode_key':[0,1,3],'primary_available':True,
             'quality_abs':.5,'v0':2.,'delta_v':3.,'advantage_h':4.,'drawdown':-1.,
             'recovery':4.,'return_h':6.,'relative_z':{'v0':0.,'delta_v':1.,
                'advantage_h':2.,'drawdown':-1.,'recovery':3.,'return_h':4.},
             'reward_steps':40,'start_timestamp_ns':50_000_000,'end_timestamp_ns':2_050_000_000}
        feature={'index':np.array([2]),'episode':np.array([[0,1,3]]),
                 'quality_abs':np.array([.5]),'v0':np.array([2.]),'delta_v':np.array([3.]),
                 'advantage_h':np.array([4.]),'drawdown':np.array([-1.]),
                 'recovery':np.array([4.]),'return_h':np.array([6.]),
                 'z_v0':np.array([0.]),'z_delta_v':np.array([1.]),
                 'z_advantage_h':np.array([2.]),'z_drawdown':np.array([-1.]),
                 'z_recovery':np.array([3.]),'z_return_h':np.array([4.])}
        validate_aligned_record(row,feature,0,'train')
        row['episode_key']=[0,1,4]
        with self.assertRaisesRegex(ValueError,'episode'):
            validate_aligned_record(row,feature,0,'train')
        row['episode_key']=[0,1,3];row['relative_z']['z_delta_v']=99.
        row['delta_v']=9.
        with self.assertRaisesRegex(ValueError,'delta_v'):
            validate_aligned_record(row,feature,0,'train')

if __name__=='__main__':unittest.main()
