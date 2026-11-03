import numpy as np
from visualnav_transformer.eat_nav.single_pose2s_selector_summary import summarize_rollout

def test_filter_stop_not_counted_as_stall_and_terminal_censored():
 root=np.zeros((201,13));root[:,3]=1
 raw=dict(root=root,command=np.zeros((200,3)),state=np.zeros((201,27)),alive=np.ones(201,bool))
 meta=dict(parent=dict(parent_id='B-1',field='boxes',level=0,candidates=[-1]*5),terminal=None)
 decisions=[dict(step=j*40,parent_id='B-1',candidate_id=None,all_filtered=True) for j in range(5)]
 r=summarize_rollout(raw,meta,decisions,[3,0],-1)
 assert r['filter_stop_seconds']==10 and r['linear_stall_windows']==0 and r['turn_stall_windows']==0
 meta['terminal']=dict(step=21,reasons=['tilt_over'])
 r=summarize_rollout(raw,meta,decisions,[3,0],-1)
 assert r['filter_stop_seconds']==1.05 and r['tip_over_observed'] and r['complete_2s_windows']==0

def test_goal_progress_and_commanded_stall_are_distinct():
 root=np.zeros((201,13));root[:,3]=1;state=np.zeros((201,27));cmd=np.zeros((200,3));cmd[:,0]=.2
 meta=dict(parent=dict(parent_id='B-1',field='boxes',level=0,candidates=[0]*5),terminal=None)
 raw=dict(root=root,command=cmd,state=state,alive=np.ones(201,bool))
 r=summarize_rollout(raw,meta,[],[3,0],-1)
 assert r['linear_stall_windows']==5 and r['linear_stall_positive_seconds']==10 and r['filter_stop_seconds']==0
 root[:,0]=np.linspace(0,2,201)
 r=summarize_rollout(raw,meta,[],[3,0],-1)
 assert abs(r['route_progress_m']-2)<1e-8 and r['linear_stall_windows']==0

def test_arrival_on_last_tick_and_post_goal_terminal_not_misclassified():
 root=np.zeros((201,13));root[:,3]=1;root[:,0]=np.linspace(0,3,201)
 raw=dict(root=root,command=np.zeros((200,3)),state=np.zeros((201,27)),alive=np.ones(201,bool))
 meta=dict(parent=dict(parent_id='B-1',field='boxes',level=0,candidates=[0]*5),terminal=None)
 # Goal at3.2m: only final step reaches .2m tolerance, including float tolerance.
 r=summarize_rollout(raw,meta,[],[3.2,0],-1)
 assert r['goal_reached']
 # A termination later than already-reached goal must not turn success into tip-over.
 meta['terminal']=dict(step=200,reasons=['tilt_over'])
 r=summarize_rollout(raw,meta,[],[2,0],120)
 assert r['goal_reached'] and not r['tip_over_observed'] and r['duration_seconds']==6
