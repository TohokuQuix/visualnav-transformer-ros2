"""Ordered, tolerance-frozen comparison of direct and full episode replay traces."""
from __future__ import annotations
import argparse
import gzip
import json
import math
from pathlib import Path


def compare_signal(left,right,limit,*,exact=False):
    errors=[];diffs=[]
    def walk(a,b,path):
        if isinstance(a,dict) and isinstance(b,dict):
            if set(a)!=set(b):errors.append((path or '$','dict_keys'));return
            for key in sorted(a):walk(a[key],b[key],f'{path}.{key}' if path else key)
            return
        if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
            if len(a)!=len(b):errors.append((path or '$','shape'));return
            for i,(x,y) in enumerate(zip(a,b)):walk(x,y,f'{path}[{i}]')
            return
        if isinstance(a,bool) or isinstance(b,bool) or a is None or b is None or isinstance(a,str) or isinstance(b,str):
            if a!=b:errors.append((path or '$','exact_value'))
            return
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            if not (math.isfinite(a) and math.isfinite(b)):
                errors.append((path or '$','nonfinite'));return
            diff=abs(float(a)-float(b));diffs.append(diff)
            if (diff!=0 if exact else diff>limit):errors.append((path or '$','numeric'))
            return
        if a!=b:errors.append((path or '$','type_or_value'))
    walk(left,right,'')
    return {'pass':not errors,'first_path':None if not errors else errors[0][0],
            'first_reason':None if not errors else errors[0][1],
            'max_abs':max(diffs) if diffs else None,
            'rms':math.sqrt(sum(d*d for d in diffs)/len(diffs)) if diffs else None,
            'element_count':len(diffs),'limit':limit,'exact':exact}


def quaternion_angle_rad(a,b):
    if len(a)!=4 or len(b)!=4 or not all(math.isfinite(float(x)) for x in (*a,*b)):
        return float('inf')
    na=math.sqrt(sum(float(x)*float(x) for x in a));nb=math.sqrt(sum(float(x)*float(x) for x in b))
    if na==0 or nb==0:return float('inf')
    dot=abs(sum(float(x)*float(y) for x,y in zip(a,b))/(na*nb))
    return 2*math.acos(min(1.,dot))


def xy_rms(a,b):
    if len(a)!=len(b) or not a:raise ValueError('XY trajectory lengths differ or are empty')
    diff=[(float(x)-float(y))**2 for left,right in zip(a,b) for x,y in zip(left[:2],right[:2])]
    return math.sqrt(sum(diff)/len(diff))


def first_divergence(cases):
    for name,step,result in cases:
        if not result['pass']:
            return {'signal':name,'step':step,'max_abs':result['max_abs'],'limit':result['limit']}
    return None


def load_run(folder,repeat=0):
    stem=f'live-{repeat}' if (folder/f'live-{repeat}.json').exists() else f'anchor_live-{repeat}'
    summary=json.loads((folder/f'{stem}.json').read_text())
    with gzip.open(folder/f'{stem}.trace.json.gz','rt') as stream:trace=json.load(stream)
    return summary,trace


def compare_runs(full,direct,limits):
    a,ta=full;b,tb=direct
    cases=[]
    def add(name,step,x,y,key,exact=False):
        cases.append((name,step,compare_signal(x,y,limits[key],exact=exact)))
    add('identity.episode_key',-1,a['state']['episode_key'],b['state']['episode_key'],'exact',True)
    add('identity.path_sha',-1,a['fixed_map_path_sha256'],b['fixed_map_path_sha256'],'exact',True)
    prefix_mismatch=b.get('first_prefix_command_mismatch')
    if prefix_mismatch is not None:
        add('prefix.command_vs_collection',int(prefix_mismatch['episode_local_step']),
            0.,float(prefix_mismatch['max_abs']),'command')
    for field,key in [('root_state','root'),('joint_position','joint'),('joint_velocity','joint'),('joint_target','joint'),
                      ('dem_history','dem'),('command_manager','command'),
                      ('latency_substeps','exact'),('latency_seconds','delay'),
                      ('action_delay_buffers','delay'),('latency_policy_obs_prev','delay'),
                      ('latency_policy_obs_reset_mask','exact'),('action_manager_action','action'),
                      ('action_manager_previous_action','action'),('action_terms','action'),
                      ('flipper_previous_target','action'),('command_time_left','command'),
                      ('policy_pre_noise','observation'),('policy_post_noise_pre_delay','observation'),
                      ('policy_actor_input','observation'),('actor_latent','latent'),('actor_action','action'),
                      ('critic_observation','observation'),('critic_value','critic')]:
        add('anchor.'+field,-1,a['anchor'].get(field),b['anchor'].get(field),key,key=='exact')
    add('anchor.root_orientation_rad',-1,quaternion_angle_rad(a['anchor']['root_state'][3:7],
        b['anchor']['root_state'][3:7]),0.,'orientation_rad')
    add('anchor.dem_history_sha256',-1,a['anchor']['dem_history_sha256'],b['anchor']['dem_history_sha256'],'exact',True)
    add('anchor.observation_history',-1,a['anchor']['observation_history'],b['anchor']['observation_history'],'delay')
    if a.get('post_follower_anchor') is not None and b.get('post_follower_anchor') is not None:
        for field,key in [('command_manager','command'),('policy_pre_noise','observation'),
                          ('policy_post_noise_pre_delay','observation'),('policy_actor_input','observation'),
                          ('actor_latent','latent'),('actor_action','action'),('critic_observation','observation'),
                          ('critic_value','critic')]:
            add('post_follower_anchor.'+field,-1,a['post_follower_anchor'].get(field),
                b['post_follower_anchor'].get(field),key)
    add('trace.length',0,len(ta),len(tb),'exact',True)
    for step,(x,y) in enumerate(zip(ta,tb)):
        for field,key in [('command','command'),('actor_input','observation'),('actor_latent','latent'),
                          ('actor_action','action'),('root_state','root'),('post_root_state','root'),
                          ('contact_force','contact'),('reward','reward'),('critic_value','critic')]:
            add('trace.'+field,step,x.get(field),y.get(field),key)
        add('trace.root_orientation_rad',step,quaternion_angle_rad(x['root_state'][3:7],
            y['root_state'][3:7]),0.,'orientation_rad')
        add('trace.dem.transformed',step,x['dem']['transformed'],y['dem']['transformed'],'dem')
        add('trace.dem.source_mask',step,x['dem']['source_mask'],y['dem']['source_mask'],'exact',True)
        add('trace.terminal',step,x.get('terminal'),y.get('terminal'),'exact',True)
    trajectory_xy_rms=xy_rms([x['post_root_state'][:2] for x in ta[:len(tb)]],
                             [x['post_root_state'][:2] for x in tb[:len(ta)]]) if ta and tb else None
    if trajectory_xy_rms is not None:
        add('trajectory.xy_rms',len(ta)-1,trajectory_xy_rms,0.,'xy_rms')
    first=first_divergence(cases)
    return {'pass':first is None,'first_divergence':first,'trajectory_xy_rms_m':trajectory_xy_rms,
            'signals':[{'signal':name,'step':step,**result} for name,step,result in cases],
            'full_steps':len(ta),'direct_steps':len(tb)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--full',type=Path,required=True)
    p.add_argument('--direct',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import yaml
    limits=yaml.safe_load(a.config.read_text())['tolerances']
    result=compare_runs(load_run(a.full),load_run(a.direct),limits)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix('.json.tmp');temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(a.output)
    print('DIRECT_COMPARISON',result['pass'],result['first_divergence'],flush=True)

if __name__=='__main__':main()
