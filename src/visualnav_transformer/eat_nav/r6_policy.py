"""Validation-derived R6 policies. No legacy progress or simulator-only inputs.

Geometry is a conservative finite DEM screen, not a contact/dynamics solver.
Vehicle dimensions must be supplied from measured collision/flipper geometry.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import time
import numpy as np

POLICY_SCHEMA='eatnav_r6_two_second_policy_v1'
METRICS=('mean_progress_m','linear_windows','turn_windows','linear_stall_sec','turn_stall_sec')


def circular_error(a,b):
    return np.abs(np.arctan2(np.sin(np.asarray(a)-b),np.cos(np.asarray(a)-b)))


def wilson(successes,total,z=1.959963984540054):
    if not isinstance(total,(int,np.integer)) or not 0<=successes<=total:
        raise ValueError('valid binomial counts required')
    if total==0:return [0.,1.]
    p=successes/total;den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    rad=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,mid-rad),min(1.,mid+rad)]


@dataclass(frozen=True)
class Footprint:
    front_m:float
    rear_m:float
    half_width_m:float
    stopping_margin_m:float=.06
    def __post_init__(self):
        if any(not math.isfinite(x) or x<=0 for x in (self.front_m,self.rear_m,self.half_width_m)) or not math.isfinite(self.stopping_margin_m) or self.stopping_margin_m<0:
            raise ValueError('positive measured body+flipper dimensions required')


def longitudinal_geometry(distance,height,min_tread_m=.2,flatness_m=.04):
    """Stable treads separate risers; a repeated stair is not one summed wall.

    One native-cell longitudinal profile, with elevations in metres. A single
    transition is conservatively classified wall; two stable-tread-separated
    transitions classify stairs. A DEM cannot establish hidden footholds.
    """
    x=np.asarray(distance,float);z=np.asarray(height,float)
    if x.ndim!=1 or z.shape!=x.shape or len(x)<2 or not np.isfinite(x).all() or not np.isfinite(z).all() or not (np.diff(x)>0).all():
        raise ValueError('finite increasing native-cell profile required')
    spacing=float(np.median(np.diff(x)))
    runs=[];start=0
    while start<len(z):
        end=start+1;low=high=z[start]
        while end<len(z) and max(high,z[end])-min(low,z[end])<=flatness_m+1e-10:
            low=min(low,z[end]);high=max(high,z[end]);end+=1
        if x[end-1]-x[start]+spacing>=min_tread_m-1e-10:
            runs.append((start,end-1,float(np.median(z[start:end]))))
        start=end
    transitions=[]
    for a,b in zip(runs[:-1],runs[1:]):
        between=np.abs(np.diff(z[a[1]:b[0]+1]))
        # At least one change steeper than the tolerance per native cell.
        if len(between) and between.max()>flatness_m+1e-9:
            transitions.append(abs(b[2]-a[2]))
    # Missing stable landing cannot erase a rising/falling edge.
    unresolved=[float(abs(z[i+1]-z[i])) for i in range(len(z)-1)
                if abs(z[i+1]-z[i])>flatness_m+1e-9]
    maximum=max(transitions+unresolved+[0.])
    kind='stairs' if len(transitions)>=2 else ('wall' if maximum>0 else 'flat_or_ramp')
    return dict(kind=kind,max_step_m=maximum,stable_tread_count=len(runs),riser_count=len(transitions),
                tread_depths_m=[float(x[b]-x[a]+spacing) for a,b,_ in runs],
                height_range_m=float(np.ptp(z)))


def _dense_pose(trajectory,spacing=.04):
    yaw=np.unwrap(np.r_[0.,np.arctan2(trajectory[:,3],trajectory[:,4])])
    xy=np.vstack([np.zeros(2),trajectory[:,:2]])
    dense=[]
    for i in range(len(xy)-1):
        n=max(1,int(math.ceil(np.linalg.norm(xy[i+1]-xy[i])/spacing)),int(math.ceil(abs(yaw[i+1]-yaw[i])/.05)))
        for t in np.linspace(0,1,n,endpoint=False):dense.append(np.r_[xy[i]*(1-t)+xy[i+1]*t,yaw[i]*(1-t)+yaw[i+1]*t])
    return np.vstack([dense,np.r_[xy[-1],yaw[-1]]])


def swept_geometry(dem,trajectory,footprint,min_tread_m=.2):
    dem=np.asarray(dem,float);height,valid=dem
    path=_dense_pose(trajectory)
    # Extend both ends for reversals and stopping uncertainty. Dimensions already
    # include flippers. Native raster cell areas are [-1,1) metres on both axes.
    x=np.linspace(-footprint.rear_m-footprint.stopping_margin_m,footprint.front_m+footprint.stopping_margin_m,
                  int(math.ceil((footprint.front_m+footprint.rear_m+2*footprint.stopping_margin_m)/.04))+1)
    y=np.linspace(-footprint.half_width_m,footprint.half_width_m,int(math.ceil(2*footprint.half_width_m/.04))+1)
    xx,yy=np.meshgrid(x,y);local=np.stack([xx,yy],-1)
    c=np.cos(path[:,2])[:,None,None];s=np.sin(path[:,2])[:,None,None]
    world=np.empty((len(path),len(y),len(x),2));world[...,0]=path[:,None,None,0]+c*local[...,0]-s*local[...,1];world[...,1]=path[:,None,None,1]+s*local[...,0]+c*local[...,1]
    rows=np.floor((world[...,1]+1.)/.1).astype(int);cols=np.floor((world[...,0]+1.)/.1).astype(int)
    inside=(rows>=0)&(rows<20)&(cols>=0)&(cols<20)
    ri=np.clip(rows,0,19);ci=np.clip(cols,0,19)
    known=inside&(valid[ri,ci]==1)&np.isfinite(height[ri,ci])
    # Unique cells prevent long dwell times from diluting an unknown corridor.
    unique=np.unique(np.stack([rows.ravel(),cols.ravel()],1),axis=0)
    cell_known=[]
    for row,col in unique:
        cell_known.append(0<=row<20 and 0<=col<20 and valid[np.clip(row,0,19),np.clip(col,0,19)]==1 and np.isfinite(height[np.clip(row,0,19),np.clip(col,0,19)]))
    result=dict(coverage=float(np.mean(cell_known)),unknown=not bool(known.all()),max_step_m=0.,geometry_kind='unknown',
                clearance_m=None,stable_tread_count=0,height_range_m=None)
    if not known.all():return result
    # Use native cell transitions on longitudinal footprint rays, avoiding
    # oversampling making a .20m tread appear shorter. Classify across all rays.
    geometries=[]
    for n in range(len(path)):
        for j in range(len(y)):
            keys=np.stack([rows[n,j],cols[n,j]],1)
            keep=np.r_[True,np.any(np.diff(keys,axis=0)!=0,axis=1)]
            z=height[ri[n,j][keep],ci[n,j][keep]]
            if len(z)>=2:
                # Euclidean native-cell pitch; .1m repeated cells along this ray.
                positions=np.cumsum(np.r_[0.,np.linalg.norm(np.diff(keys[keep],axis=0),axis=1)*.1])
                geometries.append(longitudinal_geometry(positions,z,min_tread_m=min_tread_m))
    if geometries:
        result['max_step_m']=max(g['max_step_m'] for g in geometries)
        result['geometry_kind']='stairs' if any(g['kind']=='stairs' for g in geometries) else ('wall' if any(g['kind']=='wall' for g in geometries) else 'flat_or_ramp')
        result['stable_tread_count']=max(g['stable_tread_count'] for g in geometries)
        result['height_range_m']=max(g['height_range_m'] for g in geometries)
    # Distance from swept footprint to cells with abrupt height discontinuities.
    # This diagnostic does not assume that a visible high cell is climbable.
    edge=np.zeros((20,20),bool)
    diffx=np.abs(np.diff(height,axis=1))>.04;diffy=np.abs(np.diff(height,axis=0))>.04
    edge[:,1:]|=diffx;edge[:,:-1]|=diffx;edge[1:,:]|=diffy;edge[:-1,:]|=diffy
    rr,cc=np.where(edge&(valid==1))
    if len(rr):
        obstacles=np.stack([(cc-9.5)*.1,(rr-9.5)*.1],1)
        points=world.reshape(-1,2)
        result['clearance_m']=float(min(np.linalg.norm(points-k,axis=1).min() for k in obstacles)-math.sqrt(2)*.05)
    return result


def _validate_predictions(trajectory,event_probs,dem,goal_body):
    t=np.asarray(trajectory,float);p=np.asarray(event_probs,float);d=np.asarray(dem,float);g=np.asarray(goal_body,float)
    if t.shape!=(14,20,9) or p.shape!=(14,4) or d.shape!=(2,20,20) or g.shape!=(2,):raise ValueError('R6 fixed trajectory/event/DEM/goal shape mismatch')
    if not np.isfinite(t).all() or not np.isfinite(p).all() or not np.isfinite(g).all() or ((p<0)|(p>1)).any():raise ValueError('nonfinite prediction/goal or invalid probability')
    if not np.isin(d[1],[0,1]).all() or not np.isfinite(d[0][d[1]==1]).all():raise ValueError('invalid DEM mask or known heights')
    pairs=t[...,3:].reshape(14,20,3,2)
    if not np.allclose(np.linalg.norm(pairs,axis=-1),1.,atol=.02,rtol=0):raise ValueError('R6 sin/cos pairs must be normalized')
    return t,p,d,g


def select_candidate(trajectory,event_probs,dem,goal_body,profile,*,footprint,mode='pose',event_support=None,
                     candidate_mask=None,event_horizon_sec=2,trajectory_times_sec=None,schema=POLICY_SCHEMA):
    start=time.perf_counter()
    if schema!=POLICY_SCHEMA or event_horizon_sec!=2 or mode not in ('pose','event','hybrid'):raise ValueError('R6 policy schema, horizon or mode mismatch')
    if trajectory_times_sec is not None and not np.allclose(trajectory_times_sec,np.arange(1,21)/10):raise ValueError('trajectory times must be .1..2.0 seconds')
    if not isinstance(footprint,Footprint):raise ValueError('measured Footprint required')
    t,p,d,goal=_validate_predictions(trajectory,event_probs,dem,goal_body)
    for key in ('pitch_deg','roll_deg','step_cm'):
        if key not in profile or not math.isfinite(profile[key]) or profile[key]<=0:raise ValueError('positive explicit '+key+' threshold required')
    for kind,limit in profile.get('conditional_step_cm',{}).items():
        if kind not in ('stairs','wall','flat_or_ramp') or (limit is not None and (not math.isfinite(limit) or limit<=0)):
            raise ValueError('finite positive conditional step limit or explicit unsupported null required')
    tread=float(profile.get('min_tread_m',.2))
    if not math.isfinite(tread) or tread<=0:raise ValueError('positive finite tread geometry definition required')
    margin=float(profile.get('tilt_margin_deg',0.))
    if not math.isfinite(margin) or margin<0:raise ValueError('nonnegative tilt uncertainty margin required')
    thresholds=np.asarray(profile.get('event_thresholds',[.5]*4),float)
    if thresholds.shape!=(4,) or not np.isfinite(thresholds).all() or ((thresholds<0)|(thresholds>1)).any():raise ValueError('four event probability thresholds required')
    if mode in ('event','hybrid') and (event_support is None or len(event_support)!=4 or not all(event_support)):raise ValueError('event policy requires actual training support for all four 2-second heads')
    mask=np.ones(14,bool) if candidate_mask is None else np.asarray(candidate_mask)
    if mask.shape!=(14,) or not np.isin(mask,[False,True]).all():raise ValueError('binary fourteen-candidate mask required')
    roll=np.max(np.abs(np.degrees(np.arctan2(t[:,:,5],t[:,:,6]))),axis=1)
    pitch=np.max(np.abs(np.degrees(np.arctan2(t[:,:,7],t[:,:,8]))),axis=1)
    reasons=[];geometries=[]
    for i in range(14):
        why=[]
        if not mask[i]:why.append('candidate_unavailable')
        geo=swept_geometry(d,t[i],footprint,min_tread_m=float(profile.get('min_tread_m',.2)));geometries.append(geo)
        if geo['unknown']:why.append('map_unknown')
        if mode in ('pose','hybrid'):
            if roll[i]+margin>=profile['roll_deg']:why.append('predicted_roll')
            if pitch[i]+margin>=profile['pitch_deg']:why.append('predicted_pitch')
            limits=profile.get('conditional_step_cm',{})
            limit=limits.get(geo['geometry_kind'],profile['step_cm'])
            if limit is None and geo['geometry_kind']!='unknown':why.append('geometry_threshold_unsupported')
            elif limit is not None and geo['max_step_m']*100>float(limit)+1e-8:why.append('DEM_step_'+geo['geometry_kind'])
        if mode in ('event','hybrid'):
            why.extend('event_'+name for name,prob,limit in zip(('tip_over','near_fall','linear_stall','turn_stall'),p[i],thresholds) if prob>=limit)
        reasons.append(why)
    # Two-second endpoint goal progress only; no legacy ten-second progress.
    progress=np.linalg.norm(goal)-np.linalg.norm(goal[None]-t[:,-1,:2],axis=1)
    allowed=np.array([not why for why in reasons])
    selected=int(np.argmax(np.where(allowed,progress,-np.inf))) if allowed.any() else None
    return dict(schema=POLICY_SCHEMA,mode=mode,profile_id=profile.get('id'),selected_index=selected,stop=selected is None,
        reasons=reasons,excluded=(~allowed).tolist(),coverage=[x['coverage'] for x in geometries],
        clearance_m=[x['clearance_m'] for x in geometries],geometry=geometries,
        max_predicted_roll_deg=roll.tolist(),max_predicted_pitch_deg=pitch.tolist(),
        predicted_goal_progress_m=progress.tolist(),processing_sec=time.perf_counter()-start)


def _support(rows):
    total=len(rows);fail=sum(bool(x['failed']) for x in rows)
    groups={}
    for row in rows:groups.setdefault(str(row['group_id']),[]).append(bool(row['failed']))
    group_fail=sum(any(values) for values in groups.values())
    return dict(sample_count=total,failure_count=fail,failure_rate=fail/total if total else None,
                failure_rate_wilson95=wilson(fail,total),
                branch_CI_assumption='descriptive; shared source groups are correlated',
                group_count=len(groups),groups_with_any_failure=group_fail,
                group_any_failure_wilson95=wilson(group_fail,len(groups)))


def derive_profiles(records,min_samples=30,min_success=10,min_failure=5):
    """Derive 12 validation mixture-quantile combinations or report shortage.

    Records are one actual observed candidate rollout, grouped before splitting;
    source IDs and all counts stay attached. Censored outcomes are excluded,
    never relabeled success. Both success and failure magnitudes enter the empirical quantiles; low-angle
    stalls do not imply a monotonic posture-failure boundary. Candidates require rollouts.
    """
    rows=[];censored=0
    for record in records:
        if record.get('split')!='validation':raise ValueError('threshold derivation accepts validation records only')
        if not record.get('complete',False):censored+=1;continue
        if not isinstance(record.get('failed'),(bool,np.bool_)):raise ValueError('observed boolean failed outcome required')
        if not record.get('group_id'):raise ValueError('source group identity required')
        for key in ('max_abs_pitch_deg','max_abs_roll_deg','step_cm'):
            if key not in record or not math.isfinite(record[key]) or record[key]<0:raise ValueError('finite observed '+key+' required')
        rows.append(record)
    support=_support(rows);success=[r for r in rows if not r['failed']];failure=[r for r in rows if r['failed']]
    groups={}
    for r in rows:groups.setdefault(str(r.get('geometry_kind','unknown')),[]).append(r)
    conditional={k:_support(v) for k,v in groups.items()}
    source=dict(**support,censored_excluded=censored,conditional=conditional,split='validation',
                method='observed_success_and_failure_mixture_quantiles_v2')
    if len(rows)<min_samples or len(success)<min_success or len(failure)<min_failure:
        return dict(status='insufficient_support',profiles=[],source=source,requirements=dict(min_samples=min_samples,min_success=min_success,min_failure=min_failure))
    source_hash=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest();source['records_sha256']=source_hash
    def quantile_threshold(data,key,q):
        good=[r[key] for r in data if not r['failed']];bad=[r[key] for r in data if r['failed']]
        if len(good)<5 or len(bad)<3:return None
        return max(.001,float(np.quantile(good+bad,q)))
    source['magnitude_distributions']={key:{name:dict(count=len(values),quantiles={str(q):float(np.quantile(values,q)) for q in (.1,.25,.5,.75,.9,.95)}) for name,values in [('success',[r[key] for r in success]),('failure',[r[key] for r in failure])]} for key in ('max_abs_pitch_deg','max_abs_roll_deg','step_cm')}
    profiles=[]
    for pi,qp in enumerate((.50,.65,.80,.95)):
        for ri,qr in enumerate((.50,.75,.95)):
            qs=(qp+qr)/2
            conditional_limits={}
            for kind in ('wall','stairs','flat_or_ramp'):
                data=groups.get(kind,[])
                conditional_limits[kind]=quantile_threshold(data,'step_cm',qs)
            profile=dict(schema=POLICY_SCHEMA,id=f'pose_q{pi}_{ri}',pitch_deg=quantile_threshold(rows,'max_abs_pitch_deg',qp),
                roll_deg=quantile_threshold(rows,'max_abs_roll_deg',qr),step_cm=quantile_threshold(rows,'step_cm',qs),
                conditional_step_cm=conditional_limits,quantiles=dict(pitch=qp,roll=qr,step=qs),
                tilt_margin_deg=0.,uncertainty_margin_status='none; prediction-error margin requires separate validation fit',
                min_tread_m=.2,tread_rule_source='explicit geometry definition, not a learned traversability threshold',
                event_thresholds=[.5]*4,event_threshold_source='uncalibrated probability comparator; must compare on validation',
                source=source.copy())
            # Rates use observations admitted by the joint envelope, not a claim
            # that low angle alone causally protects against failure.
            accepted=[r for r in rows if r['max_abs_pitch_deg']<profile['pitch_deg'] and r['max_abs_roll_deg']<profile['roll_deg'] and r['step_cm']<=profile['step_cm']]
            profile['observed_envelope_support']=_support(accepted)
            profiles.append(profile)
    unique={(x['pitch_deg'],x['roll_deg'],x['step_cm']) for x in profiles}
    if len(unique)<10:
        return dict(status='insufficient_support',profiles=[],source=source,reason='fewer_than_ten_distinct_data_derived_combinations',distinct_combinations=len(unique))
    return dict(status='derived',profiles=profiles,source=source,warning='candidate profiles only; no safety or superiority claim without closed-loop validation')


def rank_validation(rows,scales=None):
    """Lexicographic fall count, mean total STOP, then five equal-scale metrics."""
    if not rows:raise ValueError('validation evaluation rows required')
    trials={r['trials'] for r in rows}
    if len(trials)!=1 or next(iter(trials))<=0:raise ValueError('same positive trial counts required')
    identities=[r['id'] for r in rows]
    if len(set(identities))!=len(identities):raise ValueError('unique policy ids required')
    for row in rows:
        if row.get('split')!='validation':raise ValueError('ranking/freezing accepts validation only')
        if not isinstance(row['fall_count'],(int,np.integer)) or not 0<=row['fall_count']<=row['trials']:raise ValueError('valid fall count required')
        for name in ('mean_stop_sec',)+METRICS:
            if not math.isfinite(row[name]):raise ValueError('finite evaluation metrics required')
    if scales is None:
        scales={k:dict(origin=float(min(r[k] for r in rows)),scale=max(float(np.ptp([r[k] for r in rows])),1e-9)) for k in METRICS}
    if set(scales)!=set(METRICS) or any(not math.isfinite(v['origin']) or not math.isfinite(v['scale']) or v['scale']<=0 for v in scales.values()):raise ValueError('fixed finite validation scales required')
    result=[]
    for row in rows:
        normalized={k:(row[k]-scales[k]['origin'])/scales[k]['scale'] for k in METRICS}
        score=(-normalized['mean_progress_m']+sum(normalized[k] for k in METRICS[1:]))/5
        result.append(dict(row,third_priority_score=score,normalized_third_metrics=normalized,
                           fall_rate=row['fall_count']/row['trials'],fall_rate_wilson95=wilson(row['fall_count'],row['trials'])))
    result.sort(key=lambda r:(r['fall_count'],r['mean_stop_sec'],r['third_priority_score'],r['id']))
    return dict(ranking=result,scales=scales,third_metric_weights={k:.2 for k in METRICS},priority=['fall_count','mean_stop_sec','third_priority_score'])


def freeze_top_three(profiles,evaluations):
    if len(profiles)<10 or len({p['id'] for p in profiles}&{r['id'] for r in evaluations})<10:
        raise ValueError('at least ten pose profiles must actually be evaluated on validation before selecting three')
    if len({(p['pitch_deg'],p['roll_deg'],p['step_cm']) for p in profiles})<10:
        raise ValueError('ten distinct numeric threshold combinations required')
    ranked=rank_validation([r for r in evaluations if r['id'] in {p['id'] for p in profiles}])
    byid={p['id']:p for p in profiles}
    selected=[]
    for row in ranked['ranking'][:3]:
        if row['id'] not in byid:raise ValueError('missing exact selected profile')
        selected.append(byid[row['id']])
    result=dict(schema='eatnav_r6_frozen_top_three_v1',source='validation_only',profiles=selected,selection=ranked)
    result['freeze_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return result
