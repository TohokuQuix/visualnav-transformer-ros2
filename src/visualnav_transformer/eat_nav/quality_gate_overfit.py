"""Scratch, same-condition diffusion overfit for EAT Quality expressivity."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
from .model import EATNav
from .quality_gate_data import build_canonical_pair


def prepare_overfit_pairs(windows, pair_indices, device):
    rows=[]
    used=set()
    for low,high in pair_indices:
        if low in used or high in used:
            raise ValueError('window reused in overfit pairs')
        used.update((low,high))
        pair=build_canonical_pair(windows[low],windows[high])
        if not all(np.asarray(row['action_mask']).all() for row in pair):
            raise ValueError('overfit targets must have all eight points')
        rows.extend(pair)
    if not rows:
        raise ValueError('no overfit pairs')
    tensor=lambda name:torch.from_numpy(np.stack([np.asarray(row[name],dtype=np.float32) for row in rows])).to(device)
    result={'dem':tensor('dem'),'goal':tensor('goal'),'q':torch.tensor([[row['quality']] for row in rows],device=device,dtype=torch.float32),
            'action':tensor('action'),'mask':torch.from_numpy(np.stack([row['action_mask'] for row in rows])).to(device)}
    if not torch.equal(result['dem'][0::2],result['dem'][1::2]) or not torch.equal(result['goal'][0::2],result['goal'][1::2]):
        raise AssertionError('paired conditions differ')
    return result


def make_scratch_policy(config, stats, *, seed, device):
    if not config.quality_enabled:
        raise ValueError('overfit requires Quality-enabled model')
    torch.manual_seed(seed)
    model=EATNav(config).to(device)
    model.set_action_statistics(stats.mean,stats.std)
    return model


def fixed_bank_loss(model,batch,seed=991):
    model.eval()
    generator=torch.Generator(device=batch['action'].device).manual_seed(seed)
    noise=torch.randn(batch['action'].shape,device=batch['action'].device,generator=generator)
    steps=torch.arange(len(noise),device=noise.device)%model.scheduler.config.num_train_timesteps
    with torch.no_grad():
        context=model.encode(batch['dem'],batch['goal'],batch['q'])
        return float(model.noise_loss(batch['action'],batch['mask'],context,noise=noise,timesteps=steps))


def train_overfit(config,stats,batch,*,seed,settings,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=make_scratch_policy(config,stats,seed=seed,device=batch['dem'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings.overfit_learning_rate)
    history=[];start=time.monotonic()
    for epoch in range(1,settings.overfit_epochs+1):
        model.train()
        order=torch.randperm(len(batch['q']),device=batch['dem'].device)
        losses=[];gradients=[]
        for ids in order.split(settings.overfit_batch_size):
            optimizer.zero_grad(set_to_none=True)
            context=model.encode(batch['dem'][ids],batch['goal'][ids],batch['q'][ids])
            loss=model.noise_loss(batch['action'][ids],batch['mask'][ids],context)
            loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)))
            optimizer.step()
            losses.append(float(loss.detach()))
        fixed=fixed_bank_loss(model,batch)
        row={'epoch':epoch,'train_loss':float(np.mean(losses)),'fixed_bank_loss':fixed,
             'gradient_norm_median':float(np.median(gradients))}
        history.append(row)
        if epoch==1 or epoch%10==0:
            print('OVERFIT_EPOCH',seed,epoch,round(fixed,6),flush=True)
        if epoch%100==0 or epoch==settings.overfit_epochs:
            torch.save({'state_dict':model.state_dict(),'epoch':epoch,'seed':seed,
                        'fixed_bank_loss':fixed,'config_metadata':config.checkpoint_metadata()},
                       output/f'epoch-{epoch}.pt')
        if epoch%10==0 or epoch==settings.overfit_epochs:
            (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    report={'training_seed':seed,'epochs_ran':len(history),'initial_fixed_bank_loss':history[0]['fixed_bank_loss'],
            'final_fixed_bank_loss':history[-1]['fixed_bank_loss'],
            'fixed_bank_loss_min':min(row['fixed_bank_loss'] for row in history),
            'final_gradient_norm_median':history[-1]['gradient_norm_median'],
            'training_converged':history[-1]['fixed_bank_loss']<=settings.overfit_loss_ceiling,
            'wall_seconds':time.monotonic()-start,'checkpoint':str(output/f'epoch-{settings.overfit_epochs}.pt')}
    (output/'summary.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    return report


def pair_effect_metrics(generated_low,generated_high,target_low,target_high):
    arrays=[np.asarray(x,dtype=np.float64) for x in (generated_low,generated_high,target_low,target_high)]
    gl,gh,tl,th=arrays
    if any(x.shape!=gl.shape for x in arrays) or gl.ndim!=3 or gl.shape[-1]!=2:
        raise ValueError('pair trajectory shape mismatch')
    rms=lambda a,b:np.sqrt(np.square(a-b).sum(axis=-1).mean(axis=-1))
    generated=rms(gl,gh);target=rms(tl,th)
    ll=rms(gl,tl);lh=rms(gl,th);hh=rms(gh,th);hl=rms(gh,tl)
    def features(x):
        steps=np.diff(np.concatenate((np.zeros((len(x),1,2)),x),axis=1),axis=1)
        forward=x[:,-1,0]
        path=np.linalg.norm(steps,axis=-1).sum(axis=1)
        angles=np.arctan2(steps[:,:,1],steps[:,:,0])
        turning=np.abs(np.arctan2(np.sin(np.diff(angles,axis=1)),np.cos(np.diff(angles,axis=1)))).sum(axis=1)
        return (forward,path,turning)
    dl=[b-a for a,b in zip(features(tl),features(th))]
    dg=[b-a for a,b in zip(features(gl),features(gh))]
    direction=[]
    for actual,pred in zip(dl,dg):
        valid=np.abs(actual)>1e-5
        direction.append(float(np.mean(np.sign(actual[valid])==np.sign(pred[valid]))) if valid.any() else None)
    return {'generated_effect_m':float(np.median(generated)),
            'target_effect_m':float(np.median(target)),
            'effect_fraction':float(np.median(generated)/max(float(np.median(target)),1e-12)),
            'low_to_low_m':float(np.median(ll)),'low_to_high_m':float(np.median(lh)),
            'high_to_high_m':float(np.median(hh)),'high_to_low_m':float(np.median(hl)),
            'low_correct_fraction':float(np.mean(ll<lh)),
            'high_correct_fraction':float(np.mean(hh<hl)),
            'correct_selection_fraction':float(np.mean((ll<lh)&(hh<hl))),
            'forward_direction_fraction':direction[0],
            'path_direction_fraction':direction[1],
            'turning_direction_fraction':direction[2],
            'direction_fraction':min(v for v in direction if v is not None)}
