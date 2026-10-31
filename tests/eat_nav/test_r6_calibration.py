import numpy as np
from eat_nav_r6_calibrate_profiles import pose_underprediction

def test_censored_fall_pose_enters_margin_with_matching_observed_times():
 a=np.zeros((1,1,20,9));p=a.copy();a[...,6]=a[...,8]=p[...,6]=p[...,8]=1
 angle=np.radians(85);a[0,0,4,5:7]=[np.sin(angle),np.cos(angle)]
 p[0,0,10,5:7]=[1,0] # unobserved future cannot erase an observed miss
 m=np.zeros((1,1,20),bool);m[...,:5]=True
 e=pose_underprediction(p,a,m,np.ones((1,1),bool))
 assert abs(e[0]-85)<1e-6


def test_profile_margins_use_predeclared_validation_quantiles():
 from eat_nav_r6_calibrate_profiles import calibrated_margin
 errors=np.r_[np.zeros(60),np.linspace(1,10,40)]
 tight=calibrated_margin(dict(quantiles=dict(pitch=.5,roll=.5)),errors)
 wide=calibrated_margin(dict(quantiles=dict(pitch=.95,roll=.95)),errors)
 assert tight['quantile']==.5 and tight['margin_deg']==0
 assert wide['quantile']==.95 and wide['margin_deg']==np.quantile(errors,.95)
