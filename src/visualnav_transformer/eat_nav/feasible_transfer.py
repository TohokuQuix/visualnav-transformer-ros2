"""Pure checks and metrics for frozen candidate coverage transfer analysis."""
from __future__ import annotations
from collections import defaultdict
from itertools import combinations
import numpy as np


def to_waypoints(increments):
    x=np.asarray(increments,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=2:raise ValueError('increments must be [point,xy]')
    return np.cumsum(x,axis=0)


def to_increments(waypoints):
    x=np.asarray(waypoints,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=2:raise ValueError('waypoints must be [point,xy]')
    return np.diff(np.vstack((np.zeros((1,2),dtype=np.float32),x)),axis=0)


def validate_candidate_identity(rows,schedule,offline_by_state):
    expected={r['run_id']:r for r in schedule if r['kind'] in ('nomad','bank')}
    seen=set();keys=[]
    for row in rows:
        run=row['run_id']
        if run in seen:raise ValueError('duplicate candidate run')
        seen.add(run)
        if run not in expected:raise ValueError('candidate identity run missing')
        planned=expected[run]
        cid=(f"nomad-seed-{planned['seed']}" if planned['kind']=='nomad'
             else f"bank-{planned['name']}")
        for key in ('state_id','terrain_type','kind','seed'):
            if row[key]!=planned[key]:raise ValueError(f'candidate identity {key} mismatch')
        if row['candidate_id']!=cid:raise ValueError('candidate identity ID mismatch')
        offline=offline_by_state.get(row['state_id'],{}).get(cid)
        if offline is None or offline['candidate_id']!=cid or offline['seed']!=row['seed']:
            raise ValueError('candidate identity offline mismatch')
        keys.append((row['state_id'],cid,run))
    if len(rows)!=len(expected):raise ValueError('candidate identity count mismatch')
    return keys


def state_pairwise_accuracy(rows,*,epsilon,score_key='score'):
    if epsilon<0:raise ValueError('epsilon must be nonnegative')
    by_state=defaultdict(list)
    for row in rows:by_state[row['state_id']].append(row)
    result={}
    for state,candidates in by_state.items():
        correct=[]
        for a,b in combinations(candidates,2):
            gap=float(a['delta_v'])-float(b['delta_v'])
            if abs(gap)<=epsilon:continue
            predicted=float(a[score_key])-float(b[score_key])
            correct.append(predicted*gap>0)
        result[state]={'support':len(correct),'accuracy':float(np.mean(correct)) if correct else None}
    supported=[v['accuracy'] for v in result.values() if v['accuracy'] is not None]
    result['state_macro_accuracy']=float(np.mean(supported)) if supported else None
    return result


def map_xy_to_base(map_xy,*,anchor_xy,yaw_rad):
    """Express map-frame XY samples in the anchor robot's planar base frame."""
    x=np.asarray(map_xy,dtype=float)-np.asarray(anchor_xy,dtype=float)
    c,s=np.cos(yaw_rad),np.sin(yaw_rad)
    return x@np.array([[c,-s],[s,c]],dtype=float)


def integrate_commands(commands,*,dt_sec):
    """Integrate recorded planar velocity commands from the anchor pose."""
    x=np.asarray(commands,dtype=float)
    if x.ndim!=2 or x.shape[1]!=3 or dt_sec<=0:raise ValueError('commands must be [step,3], positive dt')
    pose=np.zeros(3,dtype=float);points=[]
    for linear_x,linear_y,angular_z in x:
        c,s=np.cos(pose[2]),np.sin(pose[2])
        pose[:2]+=dt_sec*np.array([c*linear_x-s*linear_y,s*linear_x+c*linear_y])
        pose[2]+=dt_sec*angular_z
        points.append(pose[:2].copy())
    return np.asarray(points)


def cluster_labels(samples,*,threshold):
    """Connected components under a fixed RMS threshold, stable by first ID."""
    x=np.asarray(samples,dtype=float)
    if x.ndim<2 or not np.isfinite(x).all() or threshold<=0:
        raise ValueError('invalid samples or threshold')
    parent=list(range(len(x)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if np.sqrt(np.mean((x[i]-x[j])**2))<threshold:
                parent[find(j)]=find(i)
    roots=[find(i) for i in range(len(x))]
    order={root:index for index,root in enumerate(dict.fromkeys(roots))}
    return [order[root] for root in roots]


def feasible_mode_recall(candidate_ids,labels,feasible,nomad_order,*,ks):
    if not (len(candidate_ids)==len(labels)==len(feasible)):
        raise ValueError('mode input length mismatch')
    by_id=dict(zip(candidate_ids,zip(labels,feasible)))
    if len(by_id)!=len(candidate_ids):raise ValueError('duplicate candidate ID')
    denominator={label for label,valid in zip(labels,feasible) if valid}
    result=[]
    for k in ks:
        if k<1 or k>len(nomad_order):raise ValueError('K outside NoMaD set')
        covered={by_id[cid][0] for cid in nomad_order[:k] if cid in by_id and by_id[cid][1]}
        covered &= denominator
        result.append({'k':k,'numerator':len(covered),'denominator':len(denominator),
                       'covered_mode_ids':sorted(covered),'all_mode_ids':sorted(denominator),
                       'recall':len(covered)/len(denominator) if denominator else None})
    return result


def oracle_regret(rows,selected_ids,*,metric='delta_v'):
    all_values=[float(r[metric]) for r in rows if r.get(metric) is not None and np.isfinite(r[metric])]
    chosen=[float(r[metric]) for r in rows if r['candidate_id'] in set(selected_ids)
            and r.get(metric) is not None and np.isfinite(r[metric])]
    if not all_values or not chosen:return None
    return max(0.,max(all_values)-max(chosen))


def command_cluster_labels(commands,*,linear_threshold,angular_threshold):
    x=np.asarray(commands,dtype=float)
    if x.ndim!=3 or x.shape[2]!=3 or not np.isfinite(x).all():
        raise ValueError('commands must be [candidate,step,3]')
    parent=list(range(len(x)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            linear=np.sqrt(np.mean((x[i,:,0]-x[j,:,0])**2))
            angular=np.sqrt(np.mean((x[i,:,2]-x[j,:,2])**2))
            if linear<linear_threshold and angular<angular_threshold:
                parent[find(j)]=find(i)
    roots=[find(i) for i in range(len(x))]
    labels={root:index for index,root in enumerate(dict.fromkeys(roots))}
    return [labels[root] for root in roots]
