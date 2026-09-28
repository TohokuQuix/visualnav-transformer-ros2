"""Probe parity, vector-token capacity, and frozen input contracts."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from visualnav_transformer.eat_nav.quality_gate_probe import DeterministicProbe
from visualnav_transformer.eat_nav.relative_quality_probe import RelativeProbe,quality_tensor,permutation_comparison


class TestRelativeProbe(unittest.TestCase):
    def test_scalar_graph_has_exact_gate_a_parameter_count(self):
        torch.manual_seed(11);old=DeterministicProbe()
        torch.manual_seed(11);new=RelativeProbe(quality_dim=1)
        self.assertEqual(sum(p.numel() for p in old.parameters()),sum(p.numel() for p in new.parameters()))
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for k in old.state_dict():self.assertTrue(torch.equal(old.state_dict()[k],new.state_dict()[k]),k)

    def test_multi_adds_only_192_token_input_weights(self):
        scalar=RelativeProbe(quality_dim=1);multi=RelativeProbe(quality_dim=4)
        self.assertEqual(sum(p.numel() for p in multi.parameters())-sum(p.numel() for p in scalar.parameters()),192)
        dem=torch.zeros(2,4,2,20,20);goal=torch.zeros(2,2)
        self.assertEqual(tuple(multi(dem,goal,torch.zeros(2,4)).shape),(2,8,2))
        with self.assertRaisesRegex(ValueError,'quality shape'):
            multi(dem,goal,torch.zeros(2,1))

    def test_permutation_gain_is_positive_when_original_quality_is_better(self):
        result=permutation_comparison(.2,[.3,.4])
        self.assertTrue(result['observed_outperforms_all_shuffles'])
        self.assertAlmostEqual(result['mean_observed_minus_shuffle_gain'],.15)
        self.assertFalse(permutation_comparison(.2,[.1,.3])['observed_outperforms_all_shuffles'])

    def test_quality_inputs_are_fixed_or_from_train_normalized_columns(self):
        data={'quality_abs':np.array([.1,.9]),'z_advantage_h':np.array([-2.,3.]),
              'z_drawdown':np.array([-.5,.5]),'z_v0':np.array([0.,1.]),'z_recovery':np.array([2.,-2.])}
        self.assertEqual(quality_tensor(data,'probe_0','cpu').tolist(),[[.5],[.5]])
        self.assertEqual(quality_tensor(data,'probe_adv','cpu').tolist(),[[-2.],[3.]])
        self.assertEqual(tuple(quality_tensor(data,'probe_multi','cpu').shape),(2,4))

if __name__=='__main__':unittest.main()
