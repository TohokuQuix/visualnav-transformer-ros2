"""Input routing and architecture parity for frozen relative-Quality ablations."""
import unittest
import numpy as np
import torch
from visualnav_transformer.eat_nav.relative_quality import normalize_labels
from visualnav_transformer.eat_nav.relative_quality_followup_probe import FollowupProbe, followup_quality_tensor
from visualnav_transformer.eat_nav.relative_quality_probe import RelativeProbe


class FollowupProbeTest(unittest.TestCase):
    def setUp(self):
        self.row={
            'quality_abs':np.asarray([.2,.8],dtype=np.float32),
            'z_v0':np.asarray([1.,2.],dtype=np.float32),
            'z_delta_v':np.asarray([3.,4.],dtype=np.float32),
            'z_advantage_h':np.asarray([5.,6.],dtype=np.float32),
            'z_drawdown':np.asarray([7.,8.],dtype=np.float32),
            'z_recovery':np.asarray([9.,10.],dtype=np.float32),
            'z_return_h':np.asarray([11.,12.],dtype=np.float32)}

    def test_scalar_and_multi_inputs_use_train_normalized_fields_in_fixed_order(self):
        expected={
            'probe_delta':[[3.],[4.]],'probe_recovery':[[9.],[10.]],
            'probe_return':[[11.],[12.]],'probe_delta_drawdown':[[3.,7.],[4.,8.]],
            'probe_relative_triplet':[[3.,7.,9.],[4.,8.,10.]],
            'probe_multi_reward_free':[[1.,3.,7.,9.],[2.,4.,8.,10.]],
            'probe_multi_no_v0':[[5.,7.,9.],[6.,8.,10.]]}
        for name,want in expected.items():
            with self.subTest(name=name):
                self.assertEqual(followup_quality_tensor(self.row,name,'cpu').tolist(),want)

    def test_fixed_probe_zero_keeps_quality_path_at_half(self):
        q=followup_quality_tensor(self.row,'probe_0','cpu')
        self.assertEqual(q.tolist(),[[.5],[.5]])
        self.assertEqual(FollowupProbe(1).quality_mlp[0].in_features,1)

    def test_multi_is_near_parity_and_reloads_previous_four_input_weights(self):
        base=RelativeProbe(4)
        follow=FollowupProbe(4)
        self.assertEqual(set(base.state_dict()),set(follow.state_dict()))
        follow.load_state_dict(base.state_dict())
        self.assertEqual(sum(p.numel() for p in base.parameters()),sum(p.numel() for p in follow.parameters()))
        scalar=sum(p.numel() for p in FollowupProbe(1).parameters())
        self.assertEqual(sum(p.numel() for p in follow.parameters())-scalar,192)
        self.assertEqual(sum(p.numel() for p in FollowupProbe(3).parameters())-scalar,128)
        self.assertEqual(sum(p.numel() for p in FollowupProbe(2).parameters())-scalar,64)
        base.eval();follow.eval()
        dem=torch.randn((2,4,2,20,20));goal=torch.randn((2,2));quality=torch.randn((2,4))
        with torch.no_grad():
            self.assertTrue(torch.allclose(base(dem,goal,quality),follow(dem,goal,quality),atol=1e-6))

    def test_bad_or_nonfinite_quality_is_rejected(self):
        with self.assertRaises(ValueError):followup_quality_tensor(self.row,'probe_unknown','cpu')
        self.row['z_delta_v'][0]=np.nan
        with self.assertRaises(ValueError):followup_quality_tensor(self.row,'probe_delta','cpu')

if __name__=='__main__':unittest.main()
