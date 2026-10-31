import numpy as np
from eat_nav_r6_shadow import choice_equivalence
from visualnav_transformer.eat_nav.r6_policy import POLICY_SCHEMA

def test_reuse_requires_every_decision_and_identical_choice():
 t=np.zeros((14,20,9));t[...,4]=t[...,6]=t[...,8]=1;t[:,-1,0]=.1
 d=np.zeros((2,20,20));d[1]=1
 p=dict(schema=POLICY_SCHEMA,id='p',pitch_deg=20,roll_deg=20,step_cm=10)
 dec=dict(decision_step=0,selected_index=0,predicted_trajectory=t,event_probabilities=np.zeros((14,4)),goal_body=[2,0],footprint=dict(dimensions=dict(front_m=.2,rear_m=.2,half_width_m=.2,stopping_margin_m=.06)))
 assert choice_equivalence([dec],[d],p,'pose',[True]*4)['equivalent']
 dec['selected_index']=1
 r=choice_equivalence([dec],[d],p,'pose',[True]*4)
 assert not r['equivalent'] and r['first_divergence_step']==0
 assert not choice_equivalence([],[],p,'pose',[True]*4)['equivalent']


def test_cached_features_preserve_full_policy_decision():
 from visualnav_transformer.eat_nav.r6_policy import select_candidate,select_from_features,Footprint
 t=np.zeros((14,20,9));t[...,4]=t[...,6]=t[...,8]=1;t[:,:,0]=np.arange(1,21)/100
 d=np.zeros((2,20,20));d[1]=1;p=dict(schema=POLICY_SCHEMA,id='x',pitch_deg=15,roll_deg=15,step_cm=10)
 fp=Footprint(.3,.3,.2)
 for mode in ('pose','event','hybrid'):
  original=select_candidate(t,np.zeros((14,4)),d,[2,0],p,footprint=fp,mode=mode,event_support=[True]*4)
  cached=select_from_features(original['policy_features'],p,mode=mode,event_support=[True]*4)
  assert cached['selected_index']==original['selected_index'] and cached['reasons']==original['reasons']
