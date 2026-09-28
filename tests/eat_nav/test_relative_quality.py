"""Hand-computed RL transition and relative-label contracts."""
from __future__ import annotations

import math
import unittest
import numpy as np

from visualnav_transformer.eat_nav.relative_quality import (
    aligned_endpoint, compute_relative_window, fit_train_scalers, normalize_labels,
)


def rows(*, terminal=None):
    values=[10.,8.,6.,7.,9.]
    rewards=[1.,2.,-1.,3.,0.]
    result=[]
    for i,(v,r) in enumerate(zip(values,rewards)):
        result.append({'timestamp_ns':(i+1)*50_000_000,'step':i,
                       'critic_value':v,'reward':r,'terminated':terminal=='tilt' and i==3,
                       'truncated':terminal=='timeout' and i==3,
                       'termination_reason':('tilt_over',) if terminal=='tilt' and i==3 else
                                            ('time_out',) if terminal=='timeout' and i==3 else (),
                       'episode_key':(0,0,0)})
    return result[:4] if terminal else result


class TestRelativeQuality(unittest.TestCase):
    def test_timestamp_endpoint_and_pre_action_value_alignment(self):
        source=rows()
        j=aligned_endpoint(source,0,0.15,0.002)
        self.assertEqual(j,3)
        label=compute_relative_window(source,0,j,gamma=.9,step_sec=.05)
        self.assertEqual(label['reward_steps'],3)
        self.assertEqual((label['start_timestamp_ns'],label['end_timestamp_ns']),
                         (50_000_000,200_000_000))
        self.assertEqual(label['v0'],8.)
        self.assertEqual(label['v_end'],9.)
        self.assertEqual(label['v_min'],6.)
        self.assertEqual(label['delta_v'],1.)
        self.assertEqual(label['drawdown'],-2.)
        self.assertEqual(label['recovery'],3.)
        expected=2-.9+(.9**2)*3+(.9**3)*9
        self.assertAlmostEqual(label['return_h'],expected)
        self.assertAlmostEqual(label['advantage_h'],expected-8)
        self.assertTrue(np.isfinite(label['critic_slope_per_sec']))

    def test_true_terminal_has_zero_bootstrap_and_no_endpoint_value(self):
        label=compute_relative_window(rows(terminal='tilt'),0,3,gamma=.9,step_sec=.05)
        self.assertFalse(label['primary_available'])
        self.assertEqual(label['bootstrap_kind'],'true_terminal_zero')
        self.assertIsNone(label['v_end'])
        self.assertIsNone(label['drawdown'])
        self.assertAlmostEqual(label['return_h'],2-.9+(.9**2)*3)

    def test_timeout_matches_ppo_preterminal_value_correction(self):
        label=compute_relative_window(rows(terminal='timeout'),0,3,gamma=.9,step_sec=.05)
        self.assertFalse(label['primary_available'])
        self.assertEqual(label['bootstrap_kind'],'ppo_timeout_preterminal')
        self.assertAlmostEqual(label['return_h'],2-.9+(.9**2)*3+(.9**3)*7)

    def test_reset_and_gap_cannot_be_used_for_bootstrap(self):
        source=rows()
        source[4]['episode_key']=(0,0,1)
        with self.assertRaisesRegex(ValueError,'episode boundary'):
            compute_relative_window(source,0,3,gamma=.9,step_sec=.05)
        source=rows();source[2]['timestamp_ns']+=1_000_000
        with self.assertRaisesRegex(ValueError,'cadence'):
            compute_relative_window(source,0,3,gamma=.9,step_sec=.05)

    def test_train_scaler_ignores_validation_outlier(self):
        train=[{'advantage_h':1.,'drawdown':-2.},{'advantage_h':3.,'drawdown':0.}]
        scalers=fit_train_scalers(train,['advantage_h','drawdown'])
        out=normalize_labels({'advantage_h':1000.,'drawdown':-1.},scalers)
        self.assertEqual(scalers['advantage_h']['mean'],2.)
        self.assertEqual(scalers['advantage_h']['std'],1.)
        self.assertEqual(out['drawdown'],0.)
        self.assertEqual(out['advantage_h'],998.)


if __name__=='__main__':unittest.main()
