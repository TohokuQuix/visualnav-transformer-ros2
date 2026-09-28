"""Casebook contracts that can be checked without the Isaac runtime."""
import ast
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import eat_nav_quality_casebook_worker as cb


class TestCasebook(unittest.TestCase):
    def test_fixed_q_and_diffusion_seeds(self):
        settings=cb.spec()
        self.assertEqual(settings['q_values'],[.1,.3,.5,.7,.9,.95])
        self.assertEqual(settings['diffusion_seeds'][:3],[11,29,47])
        self.assertEqual(len(settings['diffusion_seeds']),8)

    def test_stratified_selection_reproducible_and_episode_capped(self):
        rows=[(i,(i//5,)) for i in range(100)]
        a=cb.stratified_select(rows,40,641,3,lambda row:row[0]%4)
        b=cb.stratified_select(rows,40,641,3,lambda row:row[0]%4)
        self.assertEqual(a,b)
        self.assertEqual(len(a),len(set(a)))
        self.assertLessEqual(max(sum(i//5==e for i in a) for e in range(20)),3)

    def test_cohort_selection_does_not_read_model_outputs(self):
        source=ast.get_source_segment(Path(cb.__file__).read_text(),next(n for n in ast.parse(Path(cb.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='cohorts'))
        self.assertNotIn('predictions.npz',source)
        self.assertNotIn('load_checkpoint',source)

    def test_paired_noise_checkpoint_inference(self):
        import numpy as np
        import torch
        from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
        from visualnav_transformer.eat_nav.config import load_eat_config
        z=np.load(cb.ROOT/'selected.npz')
        quality=load_eat_config(cb.QUALITY)
        baseline=load_eat_config(cb.BASELINE)
        qm,_,_=load_checkpoint(cb.CHECK_Q,quality)
        bm,_,_=load_checkpoint(cb.CHECK_0,baseline)
        qm.eval();bm.eval()
        dem=torch.from_numpy(z['dem'][:1]).float()
        goal=torch.from_numpy(z['goal'][:1]).float()
        dem_before=dem.clone();goal_before=goal.clone()
        seed=11
        gen=torch.Generator().manual_seed(seed)
        noise=torch.randn((1,quality.len_traj_pred,2),generator=gen)
        with torch.no_grad():
            low=qm.sample(dem,goal,torch.tensor([.1]),seed=seed,initial_noise=noise)
            repeat=qm.sample(dem,goal,torch.tensor([.1]),seed=seed,initial_noise=noise)
            high=qm.sample(dem,goal,torch.tensor([.95]),seed=seed,initial_noise=noise)
            base_none=bm.sample(dem,goal,None,seed=seed,initial_noise=noise)
            base_dummy=bm.sample(dem,goal,torch.tensor([.95]),seed=seed,initial_noise=noise)
        self.assertFalse(qm.training)
        self.assertFalse(bm.training)
        self.assertTrue(torch.equal(low,repeat))
        self.assertTrue(torch.equal(base_none,base_dummy))
        self.assertTrue(torch.equal(dem,dem_before))
        self.assertTrue(torch.equal(goal,goal_before))
        self.assertTrue(torch.isfinite(high).all())
        self.assertFalse(low.requires_grad)

    def test_category_thresholds_frozen(self):
        threshold=cb.spec()['category_thresholds']
        self.assertGreater(threshold['sampling_dominance_ratio_min'],1)
        self.assertGreater(threshold['goal_progress_correlation_min'],0)

    def test_frozen_pair_index_mapping_checks_step_and_episode(self):
        rows=[{'episode_key':(0,1,1),'step':10},{'episode_key':(0,2,1),'step':20}]
        pair={'low_index':3,'high_index':7,'low_episode':[0,1,1],
              'high_episode':[0,2,1],'low_step':10,'high_step':20}
        mapped=cb.map_frozen_pair(pair,{3:0,7:1},rows)
        self.assertEqual((mapped['low_index'],mapped['high_index']),(0,1))
        self.assertEqual((mapped['low_original_index'],mapped['high_original_index']),(3,7))
        with self.assertRaisesRegex(ValueError,'step identity'):
            cb.map_frozen_pair(dict(pair,high_step=21),{3:0,7:1},rows)

if __name__=='__main__':unittest.main()
