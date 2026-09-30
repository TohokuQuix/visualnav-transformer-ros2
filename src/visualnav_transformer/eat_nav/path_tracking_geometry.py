"""Pure planar geometry for frozen Path versus execution diagnostics."""
from __future__ import annotations
import math
import numpy as np


def map_from_base(points,anchor_map_xy,yaw):
    p=np.asarray(points,dtype=float);anchor=np.asarray(anchor_map_xy,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or anchor.shape!=(2,):raise ValueError('planar shape mismatch')
    c,s=math.cos(yaw),math.sin(yaw)
    return p@np.array([[c,s],[-s,c]])+anchor


def base_from_map(points,anchor_map_xy,yaw):
    p=np.asarray(points,dtype=float);anchor=np.asarray(anchor_map_xy,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or anchor.shape!=(2,):raise ValueError('planar shape mismatch')
    c,s=math.cos(yaw),math.sin(yaw)
    return (p-anchor)@np.array([[c,-s],[s,c]])


def to_waypoints(increments):
    a=np.asarray(increments,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2:raise ValueError('increments must be [point,xy]')
    return np.cumsum(a,axis=0)


def to_increments(waypoints):
    a=np.asarray(waypoints,dtype=float)
    if a.ndim!=2 or a.shape[1]!=2:raise ValueError('waypoints must be [point,xy]')
    return np.diff(np.vstack((np.zeros((1,2)),a)),axis=0)


def wrap_angle(a):return (np.asarray(a)+np.pi)%(2*np.pi)-np.pi


def project_polyline(path,points):
    """Nearest nonzero segment; equal-distance ties choose earliest segment."""
    path=np.asarray(path,dtype=float);points=np.asarray(points,dtype=float)
    if path.ndim!=2 or path.shape[1]!=2 or len(path)<2 or points.ndim!=2 or points.shape[1]!=2:
        raise ValueError('path and points must be [N,2]')
    if not np.isfinite(path).all() or not np.isfinite(points).all():raise ValueError('nonfinite planar input')
    d=np.diff(path,axis=0);length=np.linalg.norm(d,axis=1);valid=length>0
    if not valid.any():raise ValueError('Path has no nonzero segment')
    start=np.r_[0.,np.cumsum(length)[:-1]]
    cross=[];arcs=[];indices=[];tangents=[];overshoot=[];ambiguous=0
    last=d[np.flatnonzero(valid)[-1]]/length[np.flatnonzero(valid)[-1]]
    for point in points:
        candidate=[]
        for i in np.flatnonzero(valid):
            t=float(np.clip(np.dot(point-path[i],d[i])/length[i]**2,0.,1.))
            closest=path[i]+t*d[i]
            candidate.append((float(np.linalg.norm(point-closest)),int(i),start[i]+t*length[i],math.atan2(d[i,1],d[i,0])))
        smallest=min(v[0] for v in candidate)
        tied=[v for v in candidate if abs(v[0]-smallest)<=1e-12]
        ambiguous+=int(len(tied)>1)
        distance,index,arc,angle=min(tied,key=lambda v:v[1])
        cross.append(distance);indices.append(index);arcs.append(arc);tangents.append(angle)
        overshoot.append(max(0.,float(np.dot(point-path[-1],last))))
    index=np.asarray(indices,dtype=int)
    return {'cross_track_m':np.asarray(cross),'arc_m':np.asarray(arcs),
            'segment_index':index,'tangent_yaw_rad':np.asarray(tangents),
            'overshoot_m':np.asarray(overshoot),'ambiguous_count':ambiguous,
            'segment_jump_count':int(np.sum(np.abs(np.diff(index))>1)),
            'zero_length_segments':int(np.sum(~valid)),
            'path_length_m':float(np.sum(length))}


def tracking_metrics(path,positions,yaws,commands,*,dt,tolerance):
    path=np.asarray(path,dtype=float);positions=np.asarray(positions,dtype=float)
    yaws=np.asarray(yaws,dtype=float);commands=np.asarray(commands,dtype=float)
    if commands.ndim!=2 or commands.shape[1]!=3 or positions.shape!=(len(commands)+1,2):
        raise ValueError('positions must have one more sample than commands')
    if yaws.shape!=(len(positions),) or dt<=0 or tolerance<=0:raise ValueError('yaw/time/tolerance mismatch')
    if not all(np.isfinite(x).all() for x in (positions,yaws,commands)):
        raise ValueError('nonfinite execution')
    proj=project_polyline(path,positions)
    count=min(len(commands),int(round(2./dt)))
    def interval(steps):
        cross=proj['cross_track_m'][:steps+1];arc=proj['arc_m'][:steps+1]
        cmd=commands[:steps];moved=np.linalg.norm(np.diff(positions[:steps+1],axis=0),axis=1).sum()
        asked=float(np.linalg.norm(cmd[:,:2],axis=1).sum()*dt)
        heading=np.abs(wrap_angle(yaws[:steps+1]-proj['tangent_yaw_rad'][:steps+1]))
        waypoint_arc=np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))
        return {'cross_track_mean_m':float(np.mean(cross)),'cross_track_median_m':float(np.median(cross)),
            'cross_track_p95_m':float(np.percentile(cross,95)),'cross_track_max_m':float(np.max(cross)),
            'final_progress_m':float(arc[-1]),'max_progress_m':float(np.max(arc)),
            'progress_ratio':float(arc[-1]/proj['path_length_m']) if proj['path_length_m'] else None,
            'waypoint_pass_rate':float(np.mean(np.max(arc)>=waypoint_arc)) if len(waypoint_arc) else None,
            'heading_error_mean_rad':float(np.mean(heading)),'heading_error_p95_rad':float(np.percentile(heading,95)),
            'realized_path_length_m':float(moved),'commanded_distance_m':asked,
            'realized_over_commanded_ratio':float(moved/asked) if asked>0 else None,
            'forward_fraction':float(np.mean(cmd[:,0]>0)),'reverse_fraction':float(np.mean(cmd[:,0]<0)),
            'turn_in_place_fraction':float(np.mean((cmd[:,0]==0)&(cmd[:,1]==0)&(cmd[:,2]!=0))),
            'stop_fraction':float(np.mean(np.all(cmd==0,axis=1))),
            'projection_ambiguous_count':project_polyline(path,positions[:steps+1])['ambiguous_count'],
            'projection_segment_jump_count':int(np.sum(np.abs(np.diff(proj['segment_index'][:steps+1]))>1))}
    distance=np.linalg.norm(positions-path[-1],axis=1)
    inside=distance<=tolerance;where=np.flatnonzero(inside)
    first=int(where[0]) if len(where) else None
    endpoint={'min_distance_m':float(np.min(distance)),'final_distance_m':float(distance[-1]),
        'reached':first is not None,'first_arrival_sec':first*dt if first is not None else None,
        'escaped_after_arrival':bool(np.any(~inside[first+1:])) if first is not None else False,
        'dwell_sec':float(np.sum(inside[1:])*dt),'overshoot_m':float(np.max(proj['overshoot_m'])),
        'final_heading_error_rad':float(abs(wrap_angle(yaws[-1]-proj['tangent_yaw_rad'][-1])))}
    return {'two_second':interval(count),'six_second':interval(len(commands)),
            'endpoint':endpoint,'path_length_m':proj['path_length_m'],
            'projection_ambiguous_count':proj['ambiguous_count'],
            'projection_segment_jump_count':proj['segment_jump_count'],
            'zero_length_segments':proj['zero_length_segments']}


def classify_run(metrics,*,constraint_valid,tracking_failure_threshold,measurement_valid=True):
    """Descriptive label; all continuous metrics remain available alongside it."""
    if not measurement_valid:return 'measurement_invalid'
    if not constraint_valid:return 'constraint_invalid'
    end=metrics['endpoint'];two=metrics['two_second'];six=metrics['six_second']
    if end['reached'] and end['escaped_after_arrival']:return 'overshoot_or_escape'
    if end['reached']:return 'reached'
    if two['final_progress_m']<=0 or six['reverse_fraction']>six['forward_fraction']:
        return 'stalled_or_reverse'
    if two['cross_track_p95_m']>tracking_failure_threshold:return 'tracking_deviation'
    return 'partial_progress'
