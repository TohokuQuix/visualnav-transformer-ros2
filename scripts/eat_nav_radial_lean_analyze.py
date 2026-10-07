"""Offline review of the immutable lean five-pose measurement."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+'\n')


def read_stream(path):
    content=Path(path).read_bytes()
    if not content.endswith(b'\n'):raise ValueError('interrupted JSONL tail '+str(path))
    return [json.loads(line) for line in content.splitlines()]


def main(root):
    root=Path(root);run=root/'run';fig=root/'figures';fig.mkdir(exist_ok=True)
    manifest=read(root/'pose-manifest.json');freeze=read(root/'input-freeze.json')
    status=read(root/'status.json')
    if status.get('stage')!='full_complete':raise ValueError('simulator not complete')
    all_rows={};summaries=[];dem_grid=[];labels=[]
    limits=freeze['numeric_diagnostics'];old=Path('/tmp/eatnav-contact-index-20261007/cycle-2-full/run/trace')
    for pose in manifest['poses']:
        name=pose['pose_id'];labels.append(pose['terrain'])
        rows=read_stream(run/'trace'/f'{name}.jsonl')
        if len(rows)!=200 or [r['physics_step'] for r in rows]!=list(range(1,201)):
            raise ValueError('step stream incomplete '+name)
        if any(r['contact']['buffer_capacity']!=4096 or not r['contact']['bounds_valid'] or
               r['contact']['buffer_full_or_truncation_suspected'] or not r['contact']['finite'] for r in rows):
            raise ValueError('contact integrity invalid '+name)
        all_rows[name]=rows
        frames=[]
        for index,step in enumerate((50,100,150,200)):
            frame=read(run/'dem'/name/f'frame-{index:02d}.json')
            if frame['physics_step']!=step or frame['shape']!=[2,20,20] or not frame['finite']:
                raise ValueError('DEM history incomplete '+name)
            frames.append(frame)
        pre=read(run/'actor'/name/'prephysics.json')
        actor=read(run/'actor'/name/'settled_step_200.json')
        if any(x['shape']!=[22] or not x['finite'] or len(x['field_order'])!=22 for x in (pre,actor)):
            raise ValueError('Actor observation invalid '+name)
        dem_grid.append([np.asarray(f['transformed'][1]) for f in frames])
        settled=rows[160:]
        def series(path):
            out=[]
            for r in settled:
                x=r['state']
                for k in path:x=x[k]
                out.append(float(x))
            return out
        xy=series(('xy_drift_m',));h1=series(('h1_signed_rad',))
        lin=series(('linear_speed_mps',));ang=series(('angular_speed_radps',))
        mean=lambda x:float(statistics.mean(x));rms=lambda x:float(math.sqrt(mean([a*a for a in x])))
        stable=all(x<=limits['xy_drift_max_m'] and abs(h)<=limits['h1_abs_max_rad'] and
                   v<=limits['linear_speed_max_mps'] and a<=limits['angular_speed_max_radps']
                   for x,h,v,a in zip(xy,h1,lin,ang))
        contacts=[r['contact'] for r in rows]
        occupancy=[c['buffer_occupancy'] for c in contacts]
        active=[c['active_force_distribution_n'] for c in contacts]
        body={}
        for c in contacts:
            for key,value in c['body_categories'].items():
                slot=body.setdefault(key,{'contact_count':0,'force_sum_n':0.,'force_max_n':0.,'min_separation_m':None})
                slot['contact_count']+=value['contact_count'];slot['force_sum_n']+=value['force_sum_n']
                slot['force_max_n']=max(slot['force_max_n'],value['force_max_n'])
                sep=value['min_separation_m'];slot['min_separation_m']=sep if slot['min_separation_m'] is None else min(slot['min_separation_m'],sep)
        prior=None
        op=old/f'{name}.jsonl'
        if op.exists():
            old_rows=read_stream(op)[:180]
            common=min(len(old_rows),180)
            differences=[np.asarray(rows[i]['state']['root_state_w'])-np.asarray(old_rows[i]['state']['root_state_w']) for i in range(common)]
            prior={'compared_steps':common,'root_state_rms':float(np.sqrt(np.mean(np.square(differences)))) if differences else None,
                   'contact_count_equal_steps':sum(rows[i]['contact']['contact_count']==old_rows[i]['contact']['contact_count'] for i in range(common))}
        summaries.append({'pose_id':name,'terrain':pose['terrain'],'env_index':pose['env_index'],
            'steps':200,'settled_40_all_within_limits':stable,
            'xy_drift_final_m':xy[-1],'xy_drift_max_m':max(xy),
            'h1_abs_final_rad':abs(h1[-1]),'h1_abs_max_rad':max(map(abs,h1)),
            'linear_speed_final_mps':lin[-1],'linear_speed_mean_mps':mean(lin),
            'linear_speed_max_mps':max(lin),'linear_speed_rms_mps':rms(lin),
            'angular_speed_final_radps':ang[-1],'angular_speed_mean_radps':mean(ang),
            'angular_speed_max_radps':max(ang),'angular_speed_rms_radps':rms(ang),
            'roll_final_rad':rows[-1]['state']['roll_rad'],'pitch_final_rad':rows[-1]['state']['pitch_rad'],
            'tilt_final_rad':rows[-1]['state']['tilt_rad'],
            'dem_valid_fraction':[f['valid_fraction'] for f in frames],
            'dem_four_frame_complete':True,'actor_pre_and_settled_complete':True,
            'first_contact_step':next((r['physics_step'] for r in rows if r['contact']['contact_count']>0),None),
            'contact_count_total':sum(c['contact_count'] for c in contacts),
            'occupancy_max':max(occupancy),'occupancy_ratio_max':max(occupancy)/4096,
            'minimum_separation_m':min((c['min_separation_m'] for c in contacts if c['min_separation_m'] is not None),default=None),
            'max_decoded_penetration_m':max(c['max_decoded_penetration_m'] for c in contacts),
            'final40_max_penetration_mean_m':mean([c['max_decoded_penetration_m'] for c in contacts[160:]]),
            'final40_max_penetration_max_m':max(c['max_decoded_penetration_m'] for c in contacts[160:]),
            'final40_max_penetration_distribution_m':sorted(c['max_decoded_penetration_m'] for c in contacts[160:]),
            'all_force_sum_n':sum(c['force_sum_n'] for c in contacts),
            'all_force_max_n':max(c['force_max_n'] for c in contacts),
            'active_force_count':sum(x['count'] for x in active),
            'active_force_max_n':max((x['max'] or 0.) for x in active),
            'zero_force_fraction_weighted':(sum(c['zero_force_fraction']*c['contact_count'] for c in contacts if c['zero_force_fraction'] is not None)/sum(c['contact_count'] for c in contacts)) if sum(c['contact_count'] for c in contacts) else None,
            'body_categories':body,'bounds_full_nonfinite_flags':False,'prior_step_1_180':prior})
    raw=list((run/'raw').rglob('*.npz'))
    for path in raw:
        meta=read(path.with_suffix('.json'))
        if sha(path)!=meta['raw_sha256']:raise ValueError('raw SHA mismatch '+str(path))
    if len(raw)>35:raise ValueError('unexpected raw count')
    resource=read_stream(root/'resource-series.jsonl')
    peaks={'gpu_device_peak_mib':max(r['gpu_device_used_mib'] for r in resource),
           'host_available_min_bytes':min(r['host_available_bytes'] for r in resource),
           'container_anon_shmem_peak_bytes':max(r['container_anon_shmem_bytes'] for r in resource),
           'storage_peak_bytes':max(r['storage_bytes'] for r in resource),
           'runtime_sec':max(r['elapsed_sec'] for r in resource)}
    exceeded=any(not x['settled_40_all_within_limits'] or min(x['dem_valid_fraction'])<limits['dem_valid_fraction_min'] for x in summaries)
    decision='five_pose_complete_with_numeric_exceedance_review_required' if exceeded else 'five_pose_complete_112_gate_review_required'
    result={'decision':decision,'pose_count':5,'physics_steps':200,'raw_dumps':len(raw),
            'poses':summaries,'resource':peaks,'native_exit_code':status['native_exit_code'],
            'owned_residual':status['owned_residual'],'input_freeze_sha256':sha(root/'input-freeze.json'),
            'sha_ledger_sha256':sha(root/'sha-ledger.json')}
    write(root/'analysis.json',result)
    times=np.arange(1,201)*.005
    fig1,axs=plt.subplots(2,1,figsize=(11,8),sharex=True)
    fig2,bxs=plt.subplots(2,1,figsize=(11,8),sharex=True)
    fig3,cxs=plt.subplots(2,1,figsize=(11,8))
    cats=sorted({k for x in summaries for k in x['body_categories']})
    fig4,dx=plt.subplots(figsize=(11,5))
    for j,(pose,x) in enumerate(zip(manifest['poses'],summaries)):
        rows=all_rows[pose['pose_id']];label=pose['terrain']
        axs[0].plot(times,[r['state']['xy_drift_m'] for r in rows],label=label)
        axs[1].plot(times,[r['state']['linear_speed_mps'] for r in rows],label=label)
        bxs[0].plot(times,[r['contact']['contact_count'] for r in rows],label=label)
        bxs[1].plot(times,[r['contact']['buffer_occupancy'] for r in rows],label=label)
        cxs[0].hist([r['contact']['min_separation_m'] for r in rows if r['contact']['min_separation_m'] is not None],bins=30,alpha=.35,label=label)
        cxs[1].hist([r['contact']['max_decoded_penetration_m'] for r in rows],bins=30,alpha=.35,label=label)
    axs[0].set_ylabel('XY drift (m)');axs[1].set_ylabel('linear speed (m/s)')
    bxs[0].set_ylabel('contact count');bxs[1].set_ylabel('buffer occupancy / 4096')
    cxs[0].set_xlabel('minimum separation per step (m)');cxs[1].set_xlabel('maximum penetration per step (m)')
    for ax in (axs[1],bxs[1]):ax.set_xlabel('time since placement (s)')
    for ax in (axs[0],bxs[0],cxs[0]):ax.legend(fontsize=8)
    for figg,name in ((fig1,'01-root-drift-speed.png'),(fig2,'02-contact-count-occupancy.png'),(fig3,'03-separation-penetration.png')):
        figg.tight_layout();figg.savefig(fig/name,dpi=140);plt.close(figg)
    width=.15
    for j,x in enumerate(summaries):
        dx.bar(np.arange(len(cats))+(j-2)*width,[x['body_categories'].get(k,{}).get('contact_count',0) for k in cats],width,label=x['terrain'])
    dx.set_xticks(np.arange(len(cats)),cats,rotation=25);dx.set_ylabel('decoded contact entries');dx.legend(fontsize=8)
    fig4.tight_layout();fig4.savefig(fig/'04-body-category.png',dpi=140);plt.close(fig4)
    fig5,ex=plt.subplots(figsize=(10,4));vals=np.array([[x['xy_drift_max_m']/limits['xy_drift_max_m'],x['h1_abs_max_rad']/limits['h1_abs_max_rad'],
        x['linear_speed_max_mps']/limits['linear_speed_max_mps'],x['angular_speed_max_radps']/limits['angular_speed_max_radps']]
        for x in summaries]);im=ex.imshow(vals,aspect='auto',vmin=0,vmax=max(1,float(vals.max())),cmap='viridis')
    ex.set_yticks(range(5),labels);ex.set_xticks(range(4),['XY drift','H1 heading','linear speed','angular speed']);fig5.colorbar(im,ax=ex,label='final40 max / frozen limit')
    fig5.tight_layout();fig5.savefig(fig/'05-final40-stability.png',dpi=140);plt.close(fig5)
    fig6,fx=plt.subplots(5,4,figsize=(9,10));
    for i in range(5):
        for j in range(4):
            fx[i,j].imshow(dem_grid[i][j],vmin=0,vmax=1,cmap='gray_r');fx[i,j].set_title(f'{labels[i]} frame {j+1}',fontsize=8);fx[i,j].axis('off')
    fig6.tight_layout();fig6.savefig(fig/'06-dem-mask-history.png',dpi=140);plt.close(fig6)
    fig7,gx=plt.subplots(figsize=(12,3));gx.axis('off')
    stages=['SHA + offline tests','112-env RL reset','5 radial placements','200 physics steps','DEM/Actor + contact audit','native exit + SHA','review 112 gate']
    for i,label in enumerate(stages):
        x=.02+i*.14;gx.text(x,.5,label,ha='center',va='center',fontsize=8,bbox={'boxstyle':'round','fc':'#e5f1fb'})
        if i<len(stages)-1:gx.annotate('',xy=(x+.115,.5),xytext=(x+.06,.5),arrowprops={'arrowstyle':'->'})
    fig7.tight_layout();fig7.savefig(fig/'07-lifecycle-decision.png',dpi=140);plt.close(fig7)
    proposal={'status':'proposal_only_not_executed','basis':'five_pose_200_step_complete',
              'capacity':4096,'complete_gate':200,'dem_persistence':'per frame atomic',
              'numeric_limits':limits,'contact_review':'threshold-free',
              'resource_limits':freeze['limits'],
              'predicted_112_pose_storage_bytes':math.ceil(peaks['storage_peak_bytes']*112/5*1.2),
              'single_vs_sharded':'shard by terrain or 16-pose batch if projected peak exceeds 500 MB; remeasure startup overhead before freezing',
              'second_seed':'defer until first-seed 112-pose gate is reviewed',
              'resume_boundary':'new immutable run/cycle root after a failing shard; never combine partial pose traces'}
    write(root/'112-proposal.json',proposal)
    print(json.dumps({'decision':decision,'raw_dumps':len(raw),'resource':peaks,'figures':len(list(fig.glob('*.png')))},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);main(parser.parse_args().root)
