"""Frozen-signal comparison for old and fresh bridge trajectory replays."""
from __future__ import annotations
import math

FIELDS=(
    'command','follower_command','pre_noise','post_noise','actor_input',
    'actor_latent','actor_action','processed_action','critic_observation',
    'critic_value','reward','root_state','post_root_state','joint_position',
    'joint_velocity','contact_force','raycast_hits','dem','terminal',
    'logical_sim_time_ns','clock_stamp_ns','tf_stamp_ns',
    'follower_command_stamp_ns',
)
EXACT={'command','follower_command','terminal','logical_sim_time_ns',
       'clock_stamp_ns','tf_stamp_ns','follower_command_stamp_ns'}


def difference(a,b):
    if type(a) is not type(b):
        if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)) and not isinstance(b,bool):
            return abs(float(a)-float(b))
        return math.inf
    if isinstance(a,dict):
        if a.keys()!=b.keys():return math.inf
        return max((difference(a[k],b[k]) for k in a),default=0.0)
    if isinstance(a,list):
        if len(a)!=len(b):return math.inf
        return max((difference(x,y) for x,y in zip(a,b)),default=0.0)
    if isinstance(a,(float,int)) and not isinstance(a,bool):
        return abs(float(a)-float(b)) if math.isfinite(float(a)) and math.isfinite(float(b)) else math.inf
    return 0.0 if a==b else math.inf


def compare(old,new,*,xy_rms_limit_m:float,numeric_limit:float):
    if not old or len(old)!=len(new):
        return {'pass':False,'failures':['length'],'steps_old':len(old),'steps_new':len(new)}
    failures=[];max_abs={}
    for field in FIELDS:
        diffs=[difference(a.get(field),b.get(field)) for a,b in zip(old,new)]
        maximum=max(diffs)
        max_abs[field]=maximum
        if maximum>(0.0 if field in EXACT else numeric_limit):failures.append(field)
    ref=old[0]['root_state'][:2];test=new[0]['root_state'][:2]
    xy=[]
    for a,b in zip(old,new):
        xy.extend([(a['post_root_state'][j]-ref[j])-(b['post_root_state'][j]-test[j]) for j in (0,1)])
    xy_rms=math.sqrt(sum(v*v for v in xy)/len(xy))
    if xy_rms>xy_rms_limit_m:failures.append('xy_rms_m')
    return {'pass':not failures,'failures':failures,'max_abs':max_abs,
            'xy_rms_m':xy_rms,'xy_rms_limit_m':xy_rms_limit_m,'step_count':len(old)}
