"""Read-only analysis of the bounded P1 forward-command intervention."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

from visualnav_transformer.eat_nav.p1_statistical_analysis import extract_diagnostic, extract_outcome


GROUPS=('first-differential','second-graded')
CONDITIONS=('direct','plus_normal','minus_normal','plus_controlled','minus_controlled')
GOAL=(.75,0.)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def yaw(q):
    w,x,y,z=q
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def world_to_body(q,v):
    w,x,y,z=q
    norm=w*w+x*x+y*y+z*z
    if abs(norm-1)>1e-3:raise ValueError('nonunit root quaternion')
    xx,xy,xz=v
    # Inverse quaternion rotation, equivalent to the transpose of R(q).
    return (
        (1-2*(y*y+z*z))*xx + 2*(x*y+w*z)*xy + 2*(x*z-w*y)*xz,
        2*(x*y-w*z)*xx + (1-2*(x*x+z*z))*xy + 2*(y*z+w*x)*xz,
        2*(x*z+w*y)*xx + 2*(y*z-w*x)*xy + (1-2*(x*x+y*y))*xz,
    )


def predicted_follower_forward(angular_command):
    # Frozen production params: angular_gain=1, angular_direction_sign=.5,
    # linear_speed=.3, turn_in_place_heading_threshold=.6, min_scale=0.
    heading_error=abs(angular_command)/.5
    return .3*(1-heading_error/math.pi) if heading_error<.6 else 0.


def roll_pitch(q):
    w,x,y,z=q
    roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    pitch=math.asin(max(-1.,min(1.,2*(w*y-z*x))))
    return roll,pitch


def local_xy(point,start,initial_yaw):
    dx,dy=point[0]-start[0],point[1]-start[1]
    c,s=math.cos(initial_yaw),math.sin(initial_yaw)
    return [c*dx+s*dy,-s*dx+c*dy]


def load_internal_samples(path):
    pattern=re.compile(r'pos=\(([-\d.]+), ([-\d.]+)\).*?carrot=\(([-\d.]+), ([-\d.]+)\).*?remaining_path=([\d.]+).*?target_path_dist=([\d.]+).*?heading_err=([-\d.]+).*?cmd=\(vx=([-\d.]+), wz=([-\d.]+)\).*?recovery_linear=(\w+) recovery_angular=(\w+)')
    result=[]
    for line in path.read_text(errors='replace').splitlines():
        if 'control: source=' not in line:continue
        match=pattern.search(line)
        if not match:continue
        v=match.groups()
        result.append({'pose_xy':[float(v[0]),float(v[1])],
                       'carrot_xy':[float(v[2]),float(v[3])],
                       'remaining_path_m':float(v[4]),'lookahead_path_m':float(v[5]),
                       'heading_error_rad':float(v[6]),'command_forward_m_s':float(v[7]),
                       'command_angular_rad_s':float(v[8]),
                       'linear_recovery':v[9]=='true','angular_recovery':v[10]=='true'})
    return result


def summarize(trace,row,flipper_ids):
    if len(trace)!=120 or [x['control_step'] for x in trace]!=list(range(120)):
        raise ValueError('incomplete trace')
    outcome=extract_outcome(trace,GOAL,.99)
    diagnostic=extract_diagnostic(trace,GOAL)
    if not outcome['valid_primary'] or not diagnostic['valid_to_6s']:
        raise ValueError('terminated diagnostic trial')
    first=trace[0]['root_state'];initial_yaw=yaw(first[3:7])
    points=[first[:2]]+[x['post_root_state'][:2] for x in trace]
    local=[local_xy(p,first[:2],initial_yaw) for p in points]
    steps=[]
    for t in trace:
        post=t['post_root_state']
        vel=world_to_body(post[3:7],post[7:10])
        angular=world_to_body(post[3:7],post[10:13])
        rp=roll_pitch(post[3:7])
        contact=t['contact_net_forces_w']
        if contact is None:raise ValueError('existing contact sensor missing')
        contact_mags=[math.sqrt(sum(c*c for c in force)) for force in contact]
        flipper_v=[t['joint_velocity'][i] for i in flipper_ids]
        flipper_p=[t['joint_position'][i] for i in flipper_ids]
        torque=t['applied_torque']
        if torque is None:raise ValueError('applied actuator torque missing')
        steps.append({'trial_id':row['trial_id'],'step':t['control_step'],
                      'logical_sim_time_ns':t['logical_sim_time_ns'],
                      'follower_command':t['follower_command'],
                      'applied_command':t['applied_command'],
                      'body_linear_velocity_m_s':vel,
                      'body_angular_velocity_rad_s':angular,
                      'initial_frame_xy_m':local[t['control_step']+1],
                      'root_state':post,
                      'roll_pitch_rad':rp,
                      'flipper_joint_position_rad':flipper_p,
                      'flipper_joint_velocity_rad_s':flipper_v,
                      'flipper_effort_Nm':[torque[i] for i in flipper_ids],
                      'contact_net_force_sum_N':sum(contact_mags),
                      'contact_net_force_max_N':max(contact_mags),
                      'contact_bodies_above_1N':sum(f>1 for f in contact_mags),
                      'reward':t['reward'],
                      'actor_input':t['actor_input'],
                      'actor_action':t['actor_action']})
    first40=steps[:40]
    mean=lambda field:statistics.mean(x[field] for x in first40)
    distance=sum(math.dist(a,b) for a,b in zip(points[:40],points[1:41]))
    forward=local[40][0]
    normal_error=[abs(x['follower_command'][0]-predicted_follower_forward(x['follower_command'][2]))
                  for x in first40 if x['follower_command'][0]>0 and abs(x['follower_command'][2])<.3]
    summary={k:row[k] for k in ('trial_id','anchor_id','batch_index','batch_seed','slot','condition','path_identity','path_sha256')}
    summary.update({'goal_progress_2s_m':outcome['global_goal_progress_2s_m'],
        'discounted_reward_2s':outcome['discounted_reward_2s'],
        'goal_progress_6s_m':diagnostic['global_goal_progress_6s_m'],
        'forward_displacement_2s_m':forward,
        'distance_traveled_2s_m':distance,
        'lateral_displacement_2s_m':local[40][1],
        'heading_change_2s_rad':math.atan2(math.sin(yaw(trace[39]['post_root_state'][3:7])-initial_yaw),
                                             math.cos(yaw(trace[39]['post_root_state'][3:7])-initial_yaw)),
        'follower_forward_mean_2s_m_s':statistics.mean(x['follower_command'][0] for x in first40),
        'applied_forward_mean_2s_m_s':statistics.mean(x['applied_command'][0] for x in first40),
        'follower_angular_abs_mean_2s_rad_s':statistics.mean(abs(x['follower_command'][2]) for x in first40),
        'actual_body_forward_mean_2s_m_s':statistics.mean(x['body_linear_velocity_m_s'][0] for x in first40),
        'actual_body_yaw_rate_mean_2s_rad_s':statistics.mean(x['body_angular_velocity_rad_s'][2] for x in first40),
        'command_to_body_forward_error_mean_2s_m_s':statistics.mean(
            x['body_linear_velocity_m_s'][0]-x['applied_command'][0] for x in first40),
        'turn_stop_steps_2s':sum(x['follower_command'][0]==0 and abs(x['follower_command'][2])>=.3 for x in first40),
        'controlled_changed_steps_2s':sum(abs(x['follower_command'][0]-x['applied_command'][0])>1e-9 for x in first40),
        'follower_formula_error_max_m_s':max(normal_error,default=None),
        'roll_pitch_norm_rms_2s_rad':math.sqrt(statistics.mean(sum(v*v for v in x['roll_pitch_rad']) for x in first40)),
        'flipper_speed_abs_mean_2s_rad_s':statistics.mean(statistics.mean(abs(v) for v in x['flipper_joint_velocity_rad_s']) for x in first40),
        'flipper_torque_abs_mean_2s_Nm':statistics.mean(statistics.mean(abs(v) for v in x['flipper_effort_Nm']) for x in first40),
        'contact_net_force_sum_mean_2s_N':mean('contact_net_force_sum_N'),
        'contact_net_force_max_mean_2s_N':mean('contact_net_force_max_N'),
        'trace_sha256':row['trace_sha256']})
    return steps,summary,local


def main():
    p=argparse.ArgumentParser();p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((args.study_root/'assignment.json').read_text())
    trials=[];by_id={};internals={};ledger=[];all_steps=[];traces={};paths={};origins={}
    for group in GROUPS:
        for block in range(3):
            root=args.study_root/f'pilot-{group}-b{block}-v1'
            marker=json.loads((root/'output/complete.json').read_text())
            seal=json.loads((root/'host-seal.json').read_text())
            if sha(root/'output/complete.json')!=seal['complete_sha256'] or marker['control_steps']!=120 or len(marker['trace_sha256'])!=5:
                raise ValueError('run seal/step contract')
            initial=json.loads((root/'output/initial-state.json').read_text())
            published=json.loads((root/'output/path-publication.json').read_text())
            path_map={x['trial_id']:x['map_xy'] for x in published['trial_paths']}
            fixture=json.loads((root/'input/anchor-fixtures.json').read_text())
            names=fixture[group]['prephysics_state']['joint_names']
            flipper_ids=[i for i,name in enumerate(names) if 'flipper_joint' in name]
            if len(flipper_ids)!=4:raise ValueError('flipper joint identity')
            rows=[r for r in manifest['trials'] if r['anchor_id']==group and r['batch_index']==block]
            if len(rows)!=5 or {r['condition'] for r in rows}!=set(CONDITIONS):raise ValueError('condition coverage')
            for row in rows:
                index=row['slot'];path=root/'output'/f'trace-slot-{index}.jsonl'
                if sha(path)!=marker['trace_sha256'][index]:raise ValueError('trace SHA mismatch')
                trace=[json.loads(line) for line in path.read_text().splitlines()]
                enriched=dict(row,trace_sha256=sha(path))
                steps,summary,local=summarize(trace,enriched,flipper_ids)
                if any(x['follower_command'][1:]!=x['applied_command'][1:] for x in steps):
                    raise ValueError('non-forward command intervention')
                trials.append(summary);by_id[row['trial_id']]=summary
                all_steps.extend(steps);traces[row['trial_id']]=steps
                paths[row['trial_id']]=path_map[row['trial_id']]
                origins[row['trial_id']]=(initial[index]['tile_origin_xy'],trace[0]['root_state'])
                log=root/'follower-logs'/f'reset-0001-branch-{index}.log'
                samples=load_internal_samples(log)
                if len(samples)<10:raise ValueError('follower internal logging incomplete')
                internals[row['trial_id']]={'source_log':str(log),'source_log_sha256':sha(log),'samples':samples}
            status=json.loads((root/'host-status.json').read_text())
            if status['native_exit']!=0 or status['resource_failure'] is not None:
                raise ValueError('resource or native failure')
            ledger.append({'group':group,'block':block,'root':str(root),
                           'host_seal_sha256':sha(root/'host-seal.json'),
                           'input_freeze_sha256':sha(root/'input/freeze.json'),
                           'peak_gpu_mib':status['peak_gpu_mib'],
                           'elapsed_s':status['elapsed_s']})
    contrasts=[]
    metric_names=('goal_progress_2s_m','discounted_reward_2s','goal_progress_6s_m',
                  'forward_displacement_2s_m','distance_traveled_2s_m',
                  'follower_forward_mean_2s_m_s','applied_forward_mean_2s_m_s',
                  'actual_body_forward_mean_2s_m_s','command_to_body_forward_error_mean_2s_m_s',
                  'contact_net_force_sum_mean_2s_N','flipper_speed_abs_mean_2s_rad_s',
                  'flipper_torque_abs_mean_2s_Nm','roll_pitch_norm_rms_2s_rad')
    for group in GROUPS:
        for block in range(3):
            block_rows={x['condition']:x for x in trials if x['anchor_id']==group and x['batch_index']==block}
            for side in ('plus','minus'):
                a,b,c=(block_rows[name] for name in ('direct',f'{side}_normal',f'{side}_controlled'))
                contrasts.append({'anchor_id':group,'block':block,'side':side,
                    'trial_ids':{'A':a['trial_id'],'B':b['trial_id'],'C':c['trial_id']},
                    'B_minus_A':{m:b[m]-a[m] for m in metric_names},
                    'C_minus_B':{m:c[m]-b[m] for m in metric_names},
                    'C_minus_A':{m:c[m]-a[m] for m in metric_names}})
    with gzip.open(args.output/'derived-steps.jsonl.gz','wt') as stream:
        for row in all_steps:stream.write(json.dumps(row,allow_nan=False)+'\n')
    for name,value in [('per-trial.json',trials),('matched-contrasts.json',contrasts),
                       ('follower-internal-samples.json',internals),('run-ledger.json',ledger)]:
        (args.output/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    # Fixed representative: first-differential, block 0, three conditions.
    first={r['condition']:r for r in trials if r['anchor_id']=='first-differential' and r['batch_index']==0}
    fig,ax=plt.subplots(figsize=(7,6))
    for condition in ('direct','plus_normal','plus_controlled'):
        item=first[condition];tid=item['trial_id'];xy=[x['initial_frame_xy_m'] for x in traces[tid]]
        origin,start=origins[tid];start_yaw=yaw(start[3:7])
        path=[local_xy((p[0]+origin[0],p[1]+origin[1]),start[:2],start_yaw) for p in paths[tid]]
        carrot=[local_xy((p['carrot_xy'][0]+origin[0],p['carrot_xy'][1]+origin[1]),start[:2],start_yaw)
                for p in internals[tid]['samples']]
        line,=ax.plot(*zip(*xy),label=condition+' motion')
        ax.plot(*zip(*path),color=line.get_color(),ls='--',label=condition+' Path')
        if carrot:ax.scatter(*zip(*carrot),color=line.get_color(),marker='.',s=8,alpha=.5)
    ax.scatter(.75,0,marker='*',color='black',s=120,label='global goal')
    ax.set(xlabel='initial forward (m)',ylabel='initial lateral (m)',title='First differential, block 0: Path / motion / sampled carrot')
    ax.axis('equal');ax.grid();ax.legend(fontsize=7);fig.tight_layout();fig.savefig(args.output/'path-motion-carrot.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for ax,group in zip(axes,GROUPS):
        group_rows={r['condition']:r for r in trials if r['anchor_id']==group and r['batch_index']==0}
        for condition in ('direct','plus_normal','plus_controlled'):
            seq=traces[group_rows[condition]['trial_id']][:40]
            ax.plot([i*.05 for i in range(40)],[x['applied_command'][0] for x in seq],label=condition+' command')
            ax.plot([i*.05 for i in range(40)],[x['body_linear_velocity_m_s'][0] for x in seq],ls='--',label=condition+' actual')
        ax.set_title(group);ax.grid();ax.legend(fontsize=7,ncol=2)
    fig.supxlabel('time (s)');fig.supylabel('body forward speed (m/s)');fig.tight_layout();fig.savefig(args.output/'command-vs-body-speed.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,group in zip(axes,GROUPS):
        for side,color in (('plus','tab:blue'),('minus','tab:orange')):
            rows=[(r['block'],r) for r in contrasts if r['anchor_id']==group and r['side']==side]
            ax.plot([b for b,_ in rows],[r['B_minus_A']['forward_displacement_2s_m'] for _,r in rows],marker='o',color=color,label=side+' B-A')
            ax.plot([b for b,_ in rows],[r['C_minus_A']['forward_displacement_2s_m'] for _,r in rows],marker='s',ls='--',color=color,label=side+' C-A')
        ax.axhline(0,color='black',lw=.8);ax.set(xlabel='independent block',title=group);ax.grid();ax.legend(fontsize=7)
    fig.supylabel('2 s forward displacement contrast (m)');fig.tight_layout();fig.savefig(args.output/'forward-displacement-contrasts.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,label in zip(axes,('contact_net_force_sum_mean_2s_N','flipper_torque_abs_mean_2s_Nm'),('contact net force sum (N)','flipper torque abs mean (Nm)')):
        for group in GROUPS:
            rows=[r for r in trials if r['anchor_id']==group]
            ax.scatter([r[key] for r in rows],[r['forward_displacement_2s_m'] for r in rows],label=group)
        ax.set(xlabel=label,ylabel='2 s forward displacement (m)');ax.grid();ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(args.output/'contact-actuator-response.png',dpi=150);plt.close(fig)
    provenance={'schema':'p1_causal_analysis_v1','independent_blocks_per_group':3,
        'trial_count':len(trials),'source_design_sha256':sha(args.study_root/'design-freeze.json'),
        'assignment_sha256':sha(args.study_root/'assignment.json'),
        'analysis_source_sha256':sha(__file__),
        'artifacts':{name:sha(args.output/name) for name in
                     ('derived-steps.jsonl.gz','per-trial.json','matched-contrasts.json',
                      'follower-internal-samples.json','run-ledger.json','path-motion-carrot.png',
                      'command-vs-body-speed.png','forward-displacement-contrasts.png',
                      'contact-actuator-response.png')}}
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({'trials':len(trials),'contrasts':len(contrasts),'output':str(args.output)}))


if __name__=='__main__':main()
