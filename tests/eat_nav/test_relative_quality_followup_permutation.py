"""Terrain-stratified permutations must never exchange labels across terrain."""
import unittest
import numpy as np
from visualnav_transformer.eat_nav.relative_quality_followup_permutation import terrain_permutation


class PermutationTest(unittest.TestCase):
    def test_terrain_membership_and_fixed_seed(self):
        terrain=np.array(['a','a','b','b','b','c'])
        a=terrain_permutation(terrain,np.random.default_rng(7))
        b=terrain_permutation(terrain,np.random.default_rng(7))
        self.assertEqual(a.tolist(),b.tolist())
        self.assertEqual(terrain[a].tolist(),terrain.tolist())
        self.assertEqual(sorted(a.tolist()),list(range(6)))
        self.assertEqual(a[5],5)

if __name__=='__main__':unittest.main()
