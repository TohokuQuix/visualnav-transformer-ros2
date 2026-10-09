"""Post-hoc, read-only geometry and motion analysis of sealed P1 v2 traces."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


GROUPS = ('first-differential', 'second-differential', 'first-graded', 'second-graded')
METRICS = ('goal_progress_2s_m', 'goal_progress_6s_m', 'distance_2s_m',
           'net_displacement_2s_m', 'forward_2s_m', 'lateral_2s_m',
           'heading_change_2s_rad', 'tracking_error_2s_m', 'reward_2s_discounted',
           'reward_4s_undiscounted', 'roll_pitch_rms_2s_rad',
           'angular_speed_rms_2s_rad_s', 'command_angular_abs_mean_2s',
           'command_forward_mean_2s_m_s',
           'target_lateral_at_final_forward_2s_m', 'lateral_target_error_2s_m')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yaw(quaternion):
    w, x, y, z = quaternion
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def roll_pitch(quaternion):
    w, x, y, z = quaternion
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1., min(1., 2 * (w*y - z*x))))
    return roll, pitch


def initial_frame_xy(point, origin, initial_yaw):
    dx, dy = point[0]-origin[0], point[1]-origin[1]
    c, s = math.cos(initial_yaw), math.sin(initial_yaw)
    return (c*dx+s*dy, -s*dx+c*dy)


def goal_progress(goal, point):
    return math.hypot(*goal) - math.hypot(goal[0]-point[0], goal[1]-point[1])


def polyline_distance(point, line):
    best = math.inf
    for a, b in zip(line, line[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        denom = dx*dx+dy*dy
        t = max(0., min(1., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/denom)) if denom else 0.
        best = min(best, math.hypot(point[0]-(a[0]+t*dx), point[1]-(a[1]+t*dy)))
    return best


def path_y_at_forward(forward, line):
    if forward <= line[0][0]:return line[0][1]
    for a,b in zip(line,line[1:]):
        if a[0] <= forward <= b[0]:
            fraction=(forward-a[0])/(b[0]-a[0]) if b[0]!=a[0] else 0.
            return a[1]+fraction*(b[1]-a[1])
    return line[-1][1]


def path_geometry(line, direct):
    segments = [(b[0]-a[0], b[1]-a[1]) for a,b in zip(line,line[1:])]
    lengths = [math.hypot(*d) for d in segments]
    headings = [math.atan2(d[1],d[0]) for d in segments]
    turns = [math.atan2(math.sin(b-a),math.cos(b-a)) for a,b in zip(headings,headings[1:])]
    curvatures = [abs(t)/(0.5*(lengths[i]+lengths[i+1]))
                  for i,t in enumerate(turns) if lengths[i]+lengths[i+1]>0]
    return {'path_length_m':sum(lengths),
            'initial_path_heading_rad':headings[0],
            'total_abs_heading_change_rad':sum(map(abs,turns)),
            'max_abs_discrete_curvature_1_m':max(curvatures,default=0.),
            'max_abs_lateral_path_m':max(abs(y) for _,y in line),
            'max_deviation_from_direct_m':max(polyline_distance(p,direct) for p in line)}


def rms(values):
    return math.sqrt(statistics.mean(v*v for v in values))


def summarize_trial(row, trace, path_map, tile_origin, goal, direct_line, frozen_outcome):
    first = trace[0]['root_state']
    first_xy = first[:2]
    first_yaw = yaw(first[3:7])
    line = [initial_frame_xy((p[0]+tile_origin[0],p[1]+tile_origin[1]), first_xy,first_yaw)
            for p in path_map]
    start_local = (first[0]-tile_origin[0],first[1]-tile_origin[1])
    if math.hypot(path_map[0][0]-start_local[0],path_map[0][1]-start_local[1])>0.3:
        raise ValueError('Path/root coordinate mismatch: '+row['trial_id'])
    poses = [(0.,0.)] + [initial_frame_xy(t['post_root_state'][:2],first_xy,first_yaw)
                         for t in trace]
    headings = [math.atan2(math.sin(yaw(t['post_root_state'][3:7])-first_yaw),
                           math.cos(yaw(t['post_root_state'][3:7])-first_yaw)) for t in trace]
    distances = [math.dist(a,b) for a,b in zip(poses,poses[1:])]
    root_post = [t['post_root_state'] for t in trace]
    roll_pitch_2s = [math.hypot(*roll_pitch(r[3:7])) for r in root_post[:40]]
    angular_2s = [math.sqrt(sum(v*v for v in r[10:13])) for r in root_post[:40]]
    progress = [goal_progress(goal,p) for p in poses[1:]]
    discount = sum(.99**i * t['reward'] for i,t in enumerate(trace[:40]))
    if abs(discount-frozen_outcome['discounted_reward_2s'])>1e-8 or abs(progress[39]-frozen_outcome['global_goal_progress_2s_m'])>1e-6:
        raise ValueError('frozen Outcome mismatch: '+row['trial_id'])
    result = {k:row[k] for k in ('trial_id','anchor_id','batch_index','batch_seed','slot','path_identity','path_sha256')}
    result.update(path_geometry(line,direct_line))
    result.update({'path_length_ratio_to_direct':result['path_length_m']/sum(
        math.dist(a,b) for a,b in zip(direct_line,direct_line[1:])),
        'distance_2s_m':sum(distances[:40]),
        'distance_6s_m':sum(distances),
        'net_displacement_2s_m':math.hypot(*poses[40]),
        'forward_2s_m':poses[40][0],
        'lateral_2s_m':poses[40][1],
        'target_lateral_at_final_forward_2s_m':path_y_at_forward(poses[40][0],line),
        'lateral_target_error_2s_m':poses[40][1]-path_y_at_forward(poses[40][0],line),
        'heading_change_2s_rad':headings[39],
        'tracking_error_2s_m':statistics.mean(polyline_distance(p,line) for p in poses[1:41]),
        'tracking_error_6s_m':statistics.mean(polyline_distance(p,line) for p in poses[1:]),
        'goal_progress_2s_m':progress[39],
        'goal_progress_6s_m':progress[119],
        'goal_progress_diagnostic_4s_m':progress[119]-progress[39],
        'reward_2s_discounted':discount,
        'reward_4s_undiscounted':sum(t['reward'] for t in trace[40:]),
        'roll_pitch_rms_2s_rad':rms(roll_pitch_2s),
        'angular_speed_rms_2s_rad_s':rms(angular_2s),
        'command_angular_abs_mean_2s':statistics.mean(abs(t['follower_command'][2]) for t in trace[:40]),
        'command_forward_mean_2s_m_s':statistics.mean(t['follower_command'][0] for t in trace[:40]),
        'terminal_count':sum(t['terminal'] is not None for t in trace),
        'trace_sha256':sha(row['trace_path']),
        'path_xy_initial_frame':line,
        'actual_xy_initial_frame':poses,
        'progress_by_step_m':progress})
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot-root',type=Path,required=True)
    parser.add_argument('--existing-analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((args.pilot_root/'pilot-first-differential-b0-v1/input/assignment.json').read_text())
    config=json.loads((args.pilot_root/'pilot-first-differential-b0-v1/input/statistical-config.json').read_text())
    existing=json.loads((args.existing_analysis/'outcomes.json').read_text())
    frozen={x['trial_id']:x for x in existing}
    goal=tuple(config['common_global_goal_base_xy_m'])
    per_trial=[]
    for group in GROUPS:
        for batch in (0,1):
            root=args.pilot_root/f'pilot-{group}-b{batch}-v1'
            publication=json.loads((root/'output/path-publication.json').read_text())
            paths={x['trial_id']:x['map_xy'] for x in publication['trial_paths']}
            initial=json.loads((root/'output/initial-state.json').read_text())
            assigned=sorted((r for r in manifest['trials'] if r['anchor_id']==group and r['batch_index']==batch),key=lambda r:r['slot'])
            raw=[]
            for row in assigned:
                p=root/'output'/f"trace-slot-{row['slot']}.jsonl"
                trace=[json.loads(line) for line in p.read_text().splitlines()]
                if len(trace)!=120 or sha(p)!=json.loads((root/'output/complete.json').read_text())['trace_sha256'][row['slot']]:
                    raise ValueError('unsealed or incomplete trace: '+str(p))
                raw.append((dict(row,trace_path=str(p)),trace))
            direct_row,direct_trace=next((r,t) for r,t in raw if r['path_identity']=='direct_reference')
            start=direct_trace[0]['root_state'];origin=initial[direct_row['slot']]['tile_origin_xy']
            direct_line=[initial_frame_xy((p[0]+origin[0],p[1]+origin[1]),start[:2],yaw(start[3:7]))
                         for p in paths[direct_row['trial_id']]]
            for row,trace in raw:
                item=summarize_trial(row,trace,paths[row['trial_id']],
                                      initial[row['slot']]['tile_origin_xy'],goal,direct_line,frozen[row['trial_id']])
                per_trial.append(item)
    if len(per_trial)!=56:raise ValueError('incomplete pilot')
    per_path=[]
    for group in GROUPS:
        for identity in sorted({r['path_identity'] for r in per_trial if r['anchor_id']==group and r['path_identity']!='direct_reference'}):
            contrasts=[];geometry=[]
            for batch in (0,1):
                rows=[r for r in per_trial if r['anchor_id']==group and r['batch_index']==batch]
                candidate=next(r for r in rows if r['path_identity']==identity)
                direct=[r for r in rows if r['path_identity']=='direct_reference']
                contrasts.append({'batch_index':batch,'candidate_trial_id':candidate['trial_id'],
                                  'direct_trial_ids':[r['trial_id'] for r in direct],
                                  **{k:candidate[k]-statistics.mean(d[k] for d in direct) for k in METRICS}})
                geometry.append(candidate)
            per_path.append({'anchor_id':group,'path_identity':identity,
                             'path_sha256':geometry[0]['path_sha256'],
                             'independent_batch_count':2,
                             'nominal_path_length_ratio':statistics.mean(r['path_length_ratio_to_direct'] for r in geometry),
                             'max_deviation_from_direct_m':statistics.mean(r['max_deviation_from_direct_m'] for r in geometry),
                             'batch_contrasts':contrasts,
                             'mean_contrast':{k:statistics.mean(c[k] for c in contrasts) for k in METRICS}})
    compact=[{k:v for k,v in r.items() if k not in ('path_xy_initial_frame','actual_xy_initial_frame','progress_by_step_m')}
             for r in per_trial]
    (args.output/'per-trial.json').write_text(json.dumps(compact,indent=2,allow_nan=False)+'\n')
    (args.output/'per-path.json').write_text(json.dumps(per_path,indent=2,allow_nan=False)+'\n')
    # Fixed representative: first differential StateGroup, batch 0; no outcome-based selection.
    representative=[r for r in per_trial if r['anchor_id']=='first-differential' and r['batch_index']==0]
    fig,ax=plt.subplots(figsize=(7,6))
    for r in representative:
        if r['path_identity']=='direct_reference' and r['slot']!=min(x['slot'] for x in representative if x['path_identity']=='direct_reference'):continue
        color=None
        path=r['path_xy_initial_frame'];motion=r['actual_xy_initial_frame']
        line,=ax.plot(*zip(*path),linestyle='--',label=r['path_identity']+' Path')
        ax.plot(*zip(*motion),color=line.get_color(),label=r['path_identity']+' motion')
    ax.scatter(*goal,marker='*',s=160,color='black',label='global goal')
    ax.set(xlabel='initial forward x (m)',ylabel='initial lateral y (m)',title='Fixed example: first-differential, batch 0')
    ax.axis('equal');ax.grid();ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(args.output/'path-vs-motion.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for group in GROUPS:
        rows=[r for r in per_path if r['anchor_id']==group]
        ax.scatter([r['nominal_path_length_ratio'] for r in rows],
                   [r['mean_contrast']['goal_progress_2s_m'] for r in rows],label=group)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='nominal Path length / direct length',ylabel='candidate - direct progress at 2 s (m)',title='Descriptive: 2 batches per StateGroup')
    ax.grid();ax.legend(fontsize=8);fig.tight_layout();fig.savefig(args.output/'geometry-vs-progress.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True,sharey=True)
    for ax,group in zip(axes.flat,GROUPS):
        rows=[r for r in per_trial if r['anchor_id']==group]
        for label,subset in [('direct',[r for r in rows if r['path_identity']=='direct_reference']),
                             ('alternatives',[r for r in rows if r['path_identity']!='direct_reference'])]:
            means=[statistics.mean(r['progress_by_step_m'][i] for r in subset) for i in range(120)]
            ax.plot([(i+1)*.05 for i in range(120)],means,label=label)
        ax.axvline(2,color='black',ls=':',lw=.8);ax.set_title(group);ax.grid()
    axes[0,0].legend();fig.supxlabel('time after Path (s)');fig.supylabel('global-goal progress (m)');fig.tight_layout();fig.savefig(args.output/'progress-2s-vs-6s.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5))
    for group in GROUPS:
        rows=[r for r in per_path if r['anchor_id']==group]
        ax.scatter([r['mean_contrast']['angular_speed_rms_2s_rad_s'] for r in rows],
                   [r['mean_contrast']['reward_2s_discounted'] for r in rows],label=group)
    ax.axhline(0,color='black',lw=.8);ax.axvline(0,color='black',lw=.8)
    ax.set(xlabel='candidate - direct root angular-speed RMS (rad/s)',ylabel='candidate - direct 2 s discounted reward',title='Observed motion disturbance vs total reward')
    ax.grid();ax.legend(fontsize=8);fig.tight_layout();fig.savefig(args.output/'locomotion-behavior.png',dpi=160);plt.close(fig)
    ledger={'schema':'p1_path_behavior_readonly_v1','trial_count':len(per_trial),
            'source_assignment_sha256':sha(args.pilot_root/'pilot-first-differential-b0-v1/input/assignment.json'),
            'source_outcomes_sha256':sha(args.existing_analysis/'outcomes.json'),
            'source_analysis_seal_sha256':sha(args.existing_analysis/'seal.json'),
            'per_trial_sha256':sha(args.output/'per-trial.json'),
            'per_path_sha256':sha(args.output/'per-path.json')}
    (args.output/'ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps({'trial_count':len(per_trial),'path_comparisons':len(per_path),'output':str(args.output)}))


if __name__=='__main__':
    main()
