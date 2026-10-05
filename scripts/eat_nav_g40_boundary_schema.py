"""Derive Critic concatenation from the saved training observation config."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/quality-identifiability-gate/g40-boundary-freeze'
ENV=Path('/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/params/env.yaml')

META={
 'base_lin_vel':('m/s','base','base linear velocity'),
 'base_ang_vel':('rad/s','base','base angular velocity'),
 'projected_gravity':('unitless','base','projected gravity direction'),
 'velocity_commands':('m/s, m/s, rad/s','base-yaw command','current command; no Path history'),
 'flipper_fold_mode':('categorical -1/0/1','controller','current mode'),
 'joint_pos':('rad','joint-relative','relative to default joint positions'),
 'joint_vel':('rad/s','joint','mean right and left track speed'),
 'actions':('normalized action units','action-manager','last action'),
 'control_latency':('fraction of max latency','controller','current latency'),
 'roller_contact_binary':('0/1','per-contact-body','force > 1 N'),
 'joint_effort':('0.01 * clipped N m','joint','four flippers + signed right/left track sums'),
 'height_scan':('m','yaw-aligned scanner','sensor height minus ray hit z minus 0.5 m'),
}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def derive_schema(path=ENV,*,action_dim=6,command_dim=3):
    cfg=yaml.load(Path(path).read_text(),Loader=yaml.BaseLoader)
    group=cfg['observations']['critic']
    if group['concatenate_terms']!='true' or group['enable_corruption']!='false':
        raise ValueError('Critic concatenation/corruption differs')
    scanner=cfg['scene']['height_scanner_critic']['pattern_cfg']
    nx=round(float(scanner['size'][0])/float(scanner['resolution']))+1
    ny=round(float(scanner['size'][1])/float(scanner['resolution']))+1
    expected_dims={
        'base_lin_vel':3,'base_ang_vel':3,'projected_gravity':3,
        'velocity_commands':command_dim,'flipper_fold_mode':1,
        'joint_pos':len(group['joint_pos']['params']['asset_cfg']['joint_names']),
        'joint_vel':len([x for x in ('right_track_cfg','left_track_cfg') if x in group['joint_vel']['params']]),
        'actions':action_dim,'control_latency':1,
        'roller_contact_binary':len(group['roller_contact_binary']['params']['sensor_cfg']['body_names']),
        'joint_effort':len(group['joint_effort']['params']['flipper_cfg']['joint_names'])+2,
        'height_scan':nx*ny}
    fields=[];cursor=0
    for name,term in group.items():
        if not isinstance(term,dict) or 'func' not in term:continue
        if name not in expected_dims:raise ValueError(f'unknown Critic term {name}')
        if term['history_length']!='0':raise ValueError(f'unexpected Critic history {name}')
        width=expected_dims[name];unit,frame,meaning=META[name]
        fields.append({'name':name,'start':cursor,'end':cursor+width,'shape':[width],
                       'function':term['func'],'unit':unit,'frame':frame,'meaning':meaning,
                       'normalization':{'scale':None if term['scale']=='null' else float(term['scale']),
                                        'empirical':False},
                       'clip':None if term['clip']=='null' else [float(v) for v in term['clip']],
                       'noise_configured':term['noise']!='null',
                       'noise_applied':False,'phase':'pre-step',
                       'candidate_path_relation':'none' if name!='velocity_commands' else 'current follower command only',
                       'command_relation':'direct' if name=='velocity_commands' else 'held state feature'})
        cursor+=width
    if cursor!=267:raise ValueError(f'Critic width {cursor}, expected 267')
    command=next(f for f in fields if f['name']=='velocity_commands')
    if command['shape']!=[3] or command['normalization']['scale'] is not None:
        raise ValueError('command transform differs')
    return {'total_dim':cursor,'command_field':command,'fields':fields,
            'config_sha256':digest(path),
            'ordering_source':'IsaacLab ObservationManager.compute_group: ordered config terms, then concatenate',
            'command_source':'IsaacLab mdp.generated_commands -> env.command_manager.get_command(base_velocity)',
            'normalizer_source':'training agent.yaml empirical_normalization=false; Critic normalizer disabled',
            'height_scan_grid':{'nx':nx,'ny':ny,'resolution_m':float(scanner['resolution'])}}

def replace_command_only(observation,start,end,command):
    obs=np.asarray(observation,dtype=np.float32)
    cmd=np.asarray(command,dtype=np.float32)
    if obs.shape!=(267,):raise ValueError('Critic shape mismatch')
    if cmd.shape!=(end-start,):raise ValueError('command width mismatch')
    if not np.isfinite(obs).all() or not np.isfinite(cmd).all():raise ValueError('nonfinite input')
    result=obs.copy();result[start:end]=cmd
    if not np.array_equal(result[:start],obs[:start]) or not np.array_equal(result[end:],obs[end:]):
        raise ValueError('non-command term changed')
    return result

if __name__=='__main__':
    schema=derive_schema()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'critic-observation-schema.json').write_text(json.dumps(schema,indent=2,sort_keys=True)+'\n')
    print('Critic schema',schema['total_dim'],'command',schema['command_field']['start'],schema['command_field']['end'])
