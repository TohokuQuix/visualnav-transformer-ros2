"""Frozen relative-label pair selection invariants."""
from __future__ import annotations
import unittest
import numpy as np
import json
from visualnav_transformer.eat_nav.relative_quality_matching import goal_direction_delta_deg, select_disjoint_pairs, label_thresholds, episode_id_list


class TestRelativeMatching(unittest.TestCase):
    def test_goal_direction_handles_two_zero_goals_without_fabricating_angle(self):
        self.assertEqual(goal_direction_delta_deg([0.,0.],[0.,0.]),0.)
        self.assertEqual(goal_direction_delta_deg([0.,0.],[1.,0.]),180.)
        self.assertAlmostEqual(goal_direction_delta_deg([1.,0.],[0.,1.]),90.)

    def test_pair_selection_never_reuses_window_or_episode_more_than_cap(self):
        pairs=[(0,1,0.9),(0,2,0.8),(3,4,0.7),(5,6,0.6)]
        episodes=['a','b','c','a','d','e','f']
        selected=select_disjoint_pairs(pairs,episodes,max_pairs=3,episode_cap=1,descending=True)
        self.assertEqual([(i,j) for i,j,_ in selected],[(0,1),(5,6)])

    def test_episode_id_is_json_serializable(self):
        self.assertEqual(json.dumps(episode_id_list(np.array([0,2,3],dtype=np.int64))),'[0, 2, 3]')

    def test_thresholds_depend_only_on_train_candidates(self):
        values=[0.,1.,2.,3.,4.]
        t=label_thresholds(values,.25,.75)
        self.assertEqual(t['small'],1.)
        self.assertEqual(t['large'],3.)
        self.assertEqual(t['train_candidates'],5)

if __name__=='__main__':unittest.main()
