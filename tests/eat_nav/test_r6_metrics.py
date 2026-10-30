import numpy as np
from visualnav_transformer.eat_nav.r6_metrics import event_metrics,trajectory_metrics

def test_event_metrics_unknowns_do_not_improve_ap_or_recall():
 p=np.array([[.9,.2,.1,.1],[.8,.2,.1,.1],[.1,.2,.1,.1]])
 y=np.zeros_like(p);y[0,0]=1;m=np.ones_like(p,bool);m[1,0]=False
 d=event_metrics(p,y,m)
 assert d[0]['average_precision']==1 and d[0]['recall']==1
 assert d[0]['count']==2 and d[0]['false_positive']==0
 assert d[1]['average_precision'] is None

def test_future_metrics_respect_mask_and_rotation_wrap():
 y=np.zeros((1,14,20,9));y[...,4]=1;y[...,6]=1;y[...,8]=1
 p=y.copy();p[...,0]=1;m=np.ones((1,14,20),bool)
 p[0,0,10:]=np.nan;m[0,0,10:]=False
 d=trajectory_metrics(p,y,m)
 assert abs(d['ade_3d_m']-1)<1e-8 and abs(d['fde_3d_m']-1)<1e-8
 assert d['rotation_geodesic_deg']<1e-6
 assert d['complete_endpoint_count']==13
