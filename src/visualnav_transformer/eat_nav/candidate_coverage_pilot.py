"""Frozen 119-run pilot order and wall-clock start gate."""
from __future__ import annotations

_BANK = ('nominal', 'slight_left', 'slight_right', 'strong_left', 'strong_right')


def build_schedule(terrains, *, seeds):
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError('pilot requires eight distinct nested NoMaD seeds')
    if len(set(terrains)) != len(terrains):
        raise ValueError('duplicate terrain')
    rows = []
    for terrain in terrains:
        entries = [('anchor', None, None, 'freeze anchor state and map Path before candidate execution')]
        entries.extend(('duplicate', repeat, None, 'measure nominal duplicate variation') for repeat in range(3))
        entries.extend(('nomad', None, seed, 'measure NoMaD proposal outcome') for seed in seeds)
        entries.extend(('bank', name, None, 'measure fixed maneuver bank outcome') for name in _BANK)
        for kind, name, seed, purpose in entries:
            rows.append({'run_id':f'pilot-{len(rows):03d}', 'terrain_type':terrain,
                         'kind':kind, 'name':name, 'seed':seed, 'purpose':purpose})
    return rows


def may_start_next(*, elapsed_sec, predicted_remaining_sec, approved_budget_sec):
    if min(elapsed_sec, predicted_remaining_sec) < 0 or approved_budget_sec <= 0:
        raise ValueError('invalid wall-clock budget')
    return elapsed_sec < approved_budget_sec and elapsed_sec + predicted_remaining_sec <= approved_budget_sec


def bind_schedule(schedule, selected):
    """Attach frozen state identity while preserving the predeclared run order."""
    by_terrain = {row['terrain_type']: row for row in selected}
    if len(by_terrain) != len(selected):
        raise ValueError('duplicate selected terrain')
    if {row['terrain_type'] for row in schedule} != set(by_terrain):
        raise ValueError('schedule/selection terrain mismatch')
    return [{**run,
             'state_id': by_terrain[run['terrain_type']]['state_id'],
             'view_index': by_terrain[run['terrain_type']]['view_index'],
             'episode_key': by_terrain[run['terrain_type']]['episode_key'],
             'anchor_step': by_terrain[run['terrain_type']]['step'],
             'full_prefix_steps': by_terrain[run['terrain_type']]['full_prefix_steps']}
            for run in schedule]


def map_paths_from_anchor(anchor_summary, base_paths):
    """Transform all base-frame Paths using the frozen anchor map pose."""
    import math
    import numpy as np
    nominal=np.asarray(base_paths['bank-nominal'],dtype=float)
    frozen=np.asarray(anchor_summary['map_path'],dtype=float)
    if nominal.shape!=frozen.shape or not np.allclose(nominal[0],[0.,0.],atol=1e-12):
        raise ValueError('anchor nominal Path shape/origin mismatch')
    x,y=map(float,frozen[0]);root=anchor_summary['anchor']['root_state']
    w,qx,qy,qz=map(float,root[3:7])
    yaw=math.atan2(2*(w*qz+qx*qy),1-2*(qy*qy+qz*qz))
    c,s=math.cos(yaw),math.sin(yaw)
    result={}
    for name,points in base_paths.items():
        p=np.asarray(points,dtype=float)
        if p.ndim!=2 or p.shape[1]!=2 or len(p)!=len(nominal) or not np.isfinite(p).all():
            raise ValueError('candidate Path shape/nonfinite mismatch')
        result[name]=[[float(x+c*px-s*py),float(y+s*px+c*py)] for px,py in p]
    if not np.allclose(result['bank-nominal'],frozen,atol=1e-6,rtol=0):
        raise ValueError('anchor nominal map Path cannot be reconstructed')
    return result


def duplicate_gate(summaries, traces, *, xy_rms_limit_m):
    """Require matching anchor/Path/commands and 40-step XY repeatability."""
    import math
    if len(summaries)!=3 or len(traces)!=3 or any(len(trace)!=40 for trace in traces):
        raise ValueError('exactly three complete 40-step nominal duplicates required')
    if len({s['fixed_map_path_sha256'] for s in summaries})!=1:
        raise ValueError('duplicate Path SHA mismatch')
    critical=('root_state','joint_position','joint_velocity','command_manager','policy_pre_noise',
              'policy_post_noise_pre_delay','policy_actor_input','actor_latent','actor_action',
              'critic_observation','critic_value','action_manager_action','latency_substeps',
              'latency_seconds','latency_policy_obs_prev','action_delay_buffers','dem_history_sha256')
    if any(any(s['anchor'][key]!=summaries[0]['anchor'][key] for key in critical) for s in summaries[1:]):
        raise ValueError('duplicate anchor differs')
    if any(any(row['command']!=traces[0][k]['command'] for k,row in enumerate(trace))
           for trace in traces[1:]):
        raise ValueError('duplicate command series differs')
    def xy(row):
        terminal=row.get('terminal')
        return (terminal['root_state'] if terminal and 'root_state' in terminal else row['post_root_state'])[:2]
    values=[]
    for trace in traces[1:]:
        for k in range(40):
            for axis in range(2):
                delta=(xy(trace[k])[axis]-trace[0]['root_state'][axis])-(xy(traces[0][k])[axis]-traces[0][0]['root_state'][axis])
                values.append(delta*delta)
    rms=math.sqrt(sum(values)/len(values))
    if rms>xy_rms_limit_m:raise ValueError(f'duplicate XY RMS {rms:.6f} > {xy_rms_limit_m:.6f} m')
    return {'duplicate_xy_rms_m':rms,'fixed_map_path_sha256':summaries[0]['fixed_map_path_sha256'],
            'trace_steps':40,'repeat_count':3,'pass':True}
