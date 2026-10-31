"""Exact command-policy equivalence on a fully executed recorded trajectory.

Reuse is valid only for the same stateless R6 model, follower and physical
cohort, with every recorded decision input and selected index checked. This is
paired policy-equivalent reuse, not an independently simulated replication.
No event/pose teacher is fabricated for an unexecuted different candidate.
"""
import numpy as np
from visualnav_transformer.eat_nav.r6_policy import Footprint,select_candidate,select_from_features


def choice_equivalence(decisions,dems,profile,mode,support):
 if not decisions or len(decisions)!=len(dems):return dict(equivalent=False,reason='missing_decisions')
 checks=[]
 for d,dem in zip(decisions,dems):
  if 'predicted_trajectory' not in d and 'policy_features' not in d:return dict(equivalent=False,reason='missing_new_model_prediction')
  fp=Footprint(**d['footprint']['dimensions'])
  if abs(fp.stopping_margin_m-float(profile.get('stopping_margin_m',.06)))>1e-12:return dict(equivalent=False,reason='different_footprint_contract')
  choice=select_from_features(d['policy_features'],profile,mode=mode,event_support=support) if 'policy_features' in d else select_candidate(d['predicted_trajectory'],d['event_probabilities'],dem,d['goal_body'],profile,footprint=fp,mode=mode,event_support=support)
  checks.append(dict(step=d['decision_step'],actual=d['selected_index'],counterfactual=choice['selected_index'],reasons=choice['reasons']))
  if choice['selected_index']!=d['selected_index']:return dict(equivalent=False,first_divergence_step=d['decision_step'],checks=checks)
 return dict(equivalent=True,decision_count=len(checks),checks=checks,execution='verified_policy_equivalent_reuse',assumption='same deterministic stateless model and follower, fixed physical cohort; all control choices match throughout actually observed trace')
