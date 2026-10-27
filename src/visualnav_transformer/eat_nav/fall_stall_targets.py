"""Pure numerical teachers for passive, initial-decision fall/stall prediction.

No v1 labels or collector diagnostics are used to compute v2 measurements.
All times are relative to the initial Path activation, at post-action boundaries.
"""
from __future__ import annotations

import math


SCHEMA = 'eatnav_fall_stall_targets_v2'
EVENT_NAMES = ('tip_over', 'near_fall', 'linear_stall', 'turn_stall')
HORIZONS = (1, 2, 5, 10)
DT = .05
CYCLE_STEPS = 40
EPS = 1e-9  # Numerical tolerance only, not a physical threshold change.
PARALLEL_CENSOR = 'parallel_env_censored_by_other_env'


def finite_vector(value, size):
    try:
        result = [float(v) for v in value]
        return result if len(result) == size and all(math.isfinite(v) for v in result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def quaternion(root):
    q = finite_vector(root[3:7], 4) if isinstance(root, (list, tuple)) else None
    if q is None:
        return None
    scale = max(abs(v) for v in q)
    if scale == 0:
        return None
    scaled = [v/scale for v in q]
    norm = math.hypot(*scaled)
    return [v/norm for v in scaled]


def body_vector(root, world_vector):
    """Rotate a world vector by the inverse of a normalized wxyz root quaternion."""
    q = quaternion(root)
    v = finite_vector(world_vector, 3)
    if q is None or v is None:
        return None
    w, x, y, z = q
    # R_world_from_body transposed.
    return [(1-2*(y*y+z*z))*v[0]+2*(x*y+w*z)*v[1]+2*(x*z-w*y)*v[2],
            2*(x*y-w*z)*v[0]+(1-2*(x*x+z*z))*v[1]+2*(y*z+w*x)*v[2],
            2*(x*z+w*y)*v[0]+2*(y*z-w*x)*v[1]+(1-2*(x*x+y*y))*v[2]]


def root_yaw(root):
    q = quaternion(root)
    if q is None:
        return None
    w,x,y,z = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def post_state(row):
    terminal = row.get('terminal') or {}
    if terminal and terminal.get('termination_reason') != PARALLEL_CENSOR:
        return (terminal.get('pre_reset_state') or {}).get('root_state')
    return row.get('post_root_state')


def xy(root):
    return finite_vector(root[:2], 2) if isinstance(root, (list, tuple)) else None


def tilt(root):
    q = quaternion(root)
    if q is None:
        return None
    return math.degrees(math.acos(max(-1., min(1., 1-2*(q[1]**2+q[2]**2)))))


def termination_reasons(row):
    terminal = row.get('terminal') or {}
    snapshot = terminal.get('pre_reset_state') or {}
    reasons = snapshot.get('termination_reasons', terminal.get('termination_reasons'))
    if reasons is not None:
        return set(reasons)
    return set((terminal.get('termination_reason') or '').split('|')) - {''}


def measurement(label, reason):
    return {'label': label, 'target_mask': label is not None, 'reason': reason}


def world_path(body_path, root):
    origin, angle = xy(root), root_yaw(root)
    if origin is None or angle is None or not isinstance(body_path, (list, tuple)):
        return None
    points = [finite_vector(point, 2) for point in body_path]
    if len(points) < 2 or any(p is None for p in points):
        return None
    c,s = math.cos(angle), math.sin(angle)
    result = [(origin[0]+c*x-s*y, origin[1]+s*x+c*y) for x,y in points]
    return result if sum(math.dist(a,b) for a,b in zip(result,result[1:])) > 1e-12 else None


def path_arc(point, path):
    """Nearest clamped arc coordinate; equal-distance ties choose the earlier arc."""
    if point is None or path is None:
        return None
    best_distance, best_arc, offset = float('inf'), 0., 0.
    for a,b in zip(path,path[1:]):
        dx,dy = b[0]-a[0],b[1]-a[1]
        length = math.hypot(dx,dy)
        if length <= 1e-12:
            continue
        u = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/(length*length)))
        distance = (point[0]-a[0]-u*dx)**2 + (point[1]-a[1]-u*dy)**2
        arc = offset+u*length
        if distance < best_distance or (distance == best_distance and arc < best_arc):
            best_distance, best_arc = distance, arc
        offset += length
    return best_arc


def angular_response(row):
    terminal = row.get('terminal') or {}
    if terminal and terminal.get('termination_reason') != PARALLEL_CENSOR:
        root = post_state(row)
        return body_vector(root, root[10:13]) if isinstance(root, (list,tuple)) and len(root)>=13 else None
    return finite_vector(row.get('body_angular_velocity_radps'), 3)


def validate_trace(rows, decision_stamp_ns):
    if not rows or len(rows) > 200:
        raise ValueError('trace requires 1..200 control intervals')
    if decision_stamp_ns is None:
        decision_stamp_ns = rows[0].get('logical_sim_time_ns')
    if not isinstance(decision_stamp_ns, int):
        raise ValueError('integer activation timestamp required')
    for i,row in enumerate(rows):
        if row.get('control_step') != i or row.get('logical_sim_time_ns') != decision_stamp_ns+i*50_000_000:
            raise ValueError('noncontiguous control steps or 20 Hz timestamps')
        if row.get('candidate_cycle') != i//CYCLE_STEPS:
            raise ValueError('candidate cycle does not match frozen replanning period')
        if 'candidate_replan' in row and row['candidate_replan'] != bool(i and i%CYCLE_STEPS==0):
            raise ValueError('candidate replan marker mismatch')
        if row.get('terminal') and i != len(rows)-1:
            raise ValueError('trace continues after terminal or censor boundary')
    return decision_stamp_ns


def cycle_measurements(rows, body_path):
    cycles, progress, increments, offset = [], [], [], 0.
    for start in range(0,len(rows),CYCLE_STEPS):
        chunk = rows[start:start+CYCLE_STEPS]
        path = world_path(body_path, chunk[0].get('root_state'))
        commands = [finite_vector(row.get('follower_command'),3) for row in chunk]
        pre = [xy(row.get('root_state')) for row in chunk]
        posts = [xy(post_state(row)) for row in chunk]
        arcs_pre, arcs_post = [path_arc(p,path) for p in pre], [path_arc(p,path) for p in posts]
        complete = len(chunk)==CYCLE_STEPS
        spatial_valid = path is not None and all(v is not None for v in arcs_pre+arcs_post)
        command_valid = all(v is not None for v in commands)
        commanded = sum(abs(v[0])*DT for v in commands) if command_valid else None
        actual = sum(math.dist(a,b) for a,b in zip(pre,posts)) if spatial_valid else None
        best = max(arcs_pre+arcs_post)-arcs_pre[0] if spatial_valid else None
        ratio = actual/commanded if actual is not None and commanded and commanded>0 else None
        linear_valid = spatial_valid and command_valid
        if not complete:
            linear = measurement(None,'partial_cycle')
        elif not linear_valid:
            linear = measurement(None,'missing_linear_or_path_signal')
        else:
            linear = measurement(int(commanded>=.20-EPS and ratio is not None and ratio<=.25+EPS and best<=.05+EPS),
                                 'complete_cycle')
        angular = [angular_response(row) for row in chunk]
        angular_valid = command_valid and all(v is not None for v in angular)
        if not complete:
            turn = measurement(None,'partial_cycle')
        elif not angular_valid:
            turn = measurement(None,'missing_angular_signal')
        else:
            turn = measurement(int(all(abs(cmd[2])>=.01-EPS and abs(omega[2])<=.5*abs(cmd[2])+EPS
                                       for cmd,omega in zip(commands,angular))), 'complete_cycle')
        # Integrate signed arc changes. A replan is not a progress increment.
        for before,after in zip(arcs_pre,arcs_post):
            increments.append(after-before if before is not None and after is not None else None)
            if offset is None or before is None or after is None:
                offset = None
            else:
                offset += after-before
            progress.append(offset)
        cycles.append({'cycle':start//CYCLE_STEPS,'start_step':start,'end_step':start+len(chunk)-1,
                       'interval_count':len(chunk),'complete':complete,
                       'commanded_distance_m':commanded,'actual_distance_m':actual,
                       'actual_commanded_ratio':ratio,'best_progress_m':best,
                       'linear_stall':linear,'turn_stall':turn})
    return cycles,progress,increments


def build_targets(rows, body_path, *, decision_stamp_ns=None):
    """Calculate initial 1/2/5/10 s labels; preserve unknowns per event head."""
    activation = validate_trace(rows,decision_stamp_ns)
    tilts = [tilt(post_state(row)) for row in rows]
    cycles,progress,increments = cycle_measurements(rows,body_path)
    events, run_start, tipped = [], None, False
    for i,value in enumerate(tilts):
        high = value is not None and value >= 60.-EPS
        run_start = i if high and run_start is None else run_start if high else None
        if run_start is not None and i-run_start==4:
            events.append({'event':'near_fall','onset_step':run_start,'confirmation_step':i,
                           'event_step':i,'onset_sec':(run_start+1)*DT,'event_sec':(i+1)*DT})
        if not tipped and ((value is not None and value>=85.-EPS) or 'tilt_over' in termination_reasons(rows[i])):
            events.append({'event':'tip_over','onset_step':i,'event_step':i,'event_sec':(i+1)*DT})
            tipped = True
    for event in events:
        if event['event']=='near_fall':
            end=event['onset_step']
            while end+1<len(tilts) and tilts[end+1] is not None and tilts[end+1]>=60.-EPS:
                end+=1
            count=end-event['onset_step']+1
            event.update(run_end_step=end,consecutive_samples=count,
                         duration_control_sec=count*DT,sample_span_sec=(count-1)*DT)
    for cycle in cycles:
        for name in ('linear_stall','turn_stall'):
            if cycle[name]['label']==1:
                end=cycle['end_step']
                events.append({'event':name,'onset_step':cycle['start_step'],
                               'event_step':end,'event_sec':(end+1)*DT,'cycle':cycle['cycle']})
    events.sort(key=lambda e:(e['event_step'],EVENT_NAMES.index(e['event'])))
    horizons = {}
    for horizon in HORIZONS:
        intervals = horizon*20
        complete = len(rows)>=intervals
        head_targets = {}
        for name in EVENT_NAMES:
            occurred = any(e['event']==name and e['event_step']<intervals for e in events)
            if occurred:
                result = measurement(1,'event_confirmed')
            elif not complete:
                result = measurement(None,'right_censored')
            elif name in ('tip_over','near_fall'):
                valid = all(v is not None for v in tilts[:intervals])
                result = measurement(0,'complete_event_free_horizon') if valid else measurement(None,'missing_tilt_signal')
            else:
                eligible = [c for c in cycles if (c['cycle']+1)*CYCLE_STEPS<=intervals]
                valid = all(c[name]['target_mask'] for c in eligible)
                result = measurement(0,'complete_event_free_horizon') if valid else measurement(None,'missing_cycle_signal')
            head_targets[name]=result
        value=progress[intervals-1] if complete else None
        horizons[f'{horizon}s']={'observed_until_sec':min(len(rows)*DT,horizon),
                                'events':head_targets,
                                'progress':{'value_m':value,'target_mask':value is not None,
                                            'reason':'observed_progress' if value is not None else
                                                     'right_censored' if not complete else 'missing_path_or_pose'}}
    ten=list(horizons['10s']['events'].values())
    free = measurement(0,'known_event') if any(t['label']==1 for t in ten) else \
           measurement(1,'complete_event_free_trial') if all(t['label']==0 for t in ten) else \
           measurement(None,'censored_or_missing_signal')
    recoveries=[]
    for event in events:
        if event['event']=='tip_over':
            continue
        start=event['event_step'];end=start+100
        observed=end<len(rows)
        followup=increments[start+1:min(end+1,len(rows))]
        valid=increments[start] is not None and all(v is not None for v in followup)
        relative=[0.]
        if valid:
            for increment in followup:relative.append(relative[-1]+increment)
        gain=max(relative) if valid else None
        tip_after=any(e['event']=='tip_over' and start<=e['event_step']<=end for e in events)
        candidate_ids={r.get('candidate_id') for r in rows[start:min(end+1,len(rows))] if r.get('candidate_id') is not None}
        intervention=len(candidate_ids)>1 or any(r.get('external_intervention') for r in rows[start:min(end+1,len(rows))])
        if tip_after:
            recovery=measurement(0,'tip_over_within_5s')
        elif intervention:
            recovery=measurement(None,'candidate_changed_or_external_intervention')
        elif not observed:
            recovery=measurement(None,'five_second_followup_censored')
        elif not valid:
            recovery=measurement(None,'missing_followup_progress')
        else:
            recovery=measurement(int(gain>=.10-EPS),'observed_five_second_followup')
        controller_flags=[i for i in range(start,min(end+1,len(rows)))
                          if any((rows[i].get('follower_diagnostic') or {}).get(k)
                                 for k in ('linear_recovery_started','angular_recovery_started'))]
        recoveries.append({'event':event['event'],'origin_step':start,'origin_sec':(start+1)*DT,
                           **recovery,'progress_gain_m':gain,'five_sec_followup_observed':observed,
                           'tip_over_within_5s':tip_after,'controller_recovery_steps':controller_flags})
    return {'schema':SCHEMA,'control_hz':20,'replan_every_steps':40,'activation_stamp_ns':activation,
            'row_count':len(rows),'observed_until_sec':len(rows)*DT,
            'end_reason':(rows[-1].get('terminal') or {}).get('termination_reason','horizon_complete'),
            'events':events,'event_counts':{name:sum(e['event']==name for e in events) for name in EVENT_NAMES},
            'max_tilt_deg':max((v for v in tilts if v is not None),default=None),
            'unknown_tilt_samples':sum(v is None for v in tilts),
            'cycles':cycles,'horizons':horizons,'event_free_10s':free,'recoveries':recoveries,
            'diagnostics':{'effective_tilt_deg':tilts,'progress_m':progress,'progress_increment_m':increments}}
