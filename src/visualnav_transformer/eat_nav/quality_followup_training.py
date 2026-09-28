"""Resumable scratch training for the frozen Quality follow-up diagnosis."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import torch

from .quality_followup import atomic_json, stop_overfit
from .quality_gate_probe import (DeterministicProbe, _validation_loss, masked_mse,
                                 shuffle_q_within_terrain)
from .quality_gate_overfit import make_scratch_policy, fixed_bank_loss


def _save_torch(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def train_probe_resumable(data, validation, *, kind, seed, shuffle_seed, settings,
                          output, fingerprint, max_epochs=None):
    """Equal graphs and epoch orders for fixed/real/shuffled q; latest is same-run only."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if kind=='probe_0':
        train_q=torch.full_like(data['q'],settings.fixed_probe_q)
        val_q=torch.full_like(validation['q'],settings.fixed_probe_q)
    elif kind=='probe_q':
        train_q,val_q=data['q'],validation['q']
    elif kind=='probe_q_shuffled':
        if shuffle_seed is None:raise ValueError('shuffle seed required')
        train_q=torch.from_numpy(shuffle_q_within_terrain(data['q'].cpu().numpy().ravel(),data['terrain'],shuffle_seed)).to(data['q'].device).reshape(-1,1)
        val_q=validation['q']
    else:raise ValueError('unknown probe kind')
    torch.manual_seed(seed)
    model=DeterministicProbe().to(data['dem'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings.probe_learning_rate)
    history=[];best=float('inf');best_epoch=0;bad=0;first=1
    latest=output/'latest.pt'
    if latest.exists():
        state=torch.load(latest,map_location=data['dem'].device,weights_only=True)
        if state['fingerprint']!=fingerprint or state['kind']!=kind or state['seed']!=seed or state['shuffle_seed']!=shuffle_seed:
            raise ValueError('probe resume fingerprint or run identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];best=state['best'];best_epoch=state['best_epoch'];bad=state['bad']
        first=state['epoch']+1
    limit=settings.max_probe_epochs if max_epochs is None else max_epochs
    for epoch in range(first,limit+1):
        if bad>=settings.probe_patience:break
        generator=torch.Generator(device=data['dem'].device).manual_seed(seed*100000+epoch)
        order=torch.randperm(len(train_q),generator=generator,device=data['dem'].device)
        model.train();losses=[]
        for indices in order.split(settings.probe_batch_size):
            optimizer.zero_grad(set_to_none=True)
            prediction=model(data['dem'][indices],data['goal'][indices],train_q[indices])
            loss=masked_mse(prediction,data['action'][indices],data['mask'][indices])
            if not torch.isfinite(loss):raise ValueError('nonfinite Probe training loss')
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        val=_validation_loss(model,validation,val_q,settings.probe_batch_size)
        if not np.isfinite(val):raise ValueError('nonfinite Probe validation loss')
        improved=val<best-settings.probe_min_delta
        if improved:
            best=val;best_epoch=epoch;bad=0
            _save_torch(output/'best.pt',{'model':model.state_dict(),'kind':kind,'training_seed':seed,
                                        'shuffle_seed':shuffle_seed,'validation_loss':val,'epoch':epoch})
        else:bad+=1
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':val,'best':improved})
        atomic_json(output/'history.json',history)
        _save_torch(latest,{'fingerprint':fingerprint,'kind':kind,'seed':seed,'shuffle_seed':shuffle_seed,
                            'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'history':history,'best':best,'best_epoch':best_epoch,'bad':bad})
        if epoch==1 or epoch%10==0 or improved:
            print('FOLLOWUP_PROBE_EPOCH',kind,seed,shuffle_seed,epoch,round(val,6),flush=True)
    if not history:raise ValueError('probe ran no epochs')
    return {'kind':kind,'training_seed':seed,'shuffle_seed':shuffle_seed,
            'best_validation_loss':best,'best_epoch':best_epoch,'epochs_ran':len(history),
            'validation_improving_at_cap':len(history)==settings.max_probe_epochs and best_epoch>=len(history)-2,
            'checkpoint':str(output/'best.pt')}


def train_overfit_resumable(config,stats,batch,*,seed,settings,output,fingerprint,
                            max_epochs=1000,checkpoint_interval=25):
    """Scratch run with exact optimizer/RNG recovery; save at short epoch intervals."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model=make_scratch_policy(config,stats,seed=seed,device=batch['dem'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings.overfit_learning_rate)
    history=[];first=1;latest=output/'latest.pt';start=time.monotonic()
    if latest.exists():
        state=torch.load(latest,map_location=batch['dem'].device,weights_only=True)
        if state['fingerprint']!=fingerprint or state['seed']!=seed:
            raise ValueError('overfit resume fingerprint or seed mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        history=state['history'];first=state['epoch']+1
        torch.set_rng_state(state['cpu_rng'].cpu())
        if batch['dem'].device.type=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu(),batch['dem'].device)
    def checkpoint(epoch):
        _save_torch(latest,{'fingerprint':fingerprint,'seed':seed,'epoch':epoch,
                            'model':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,
                            'cpu_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state(batch['dem'].device) if batch['dem'].device.type=='cuda' else torch.empty(0,dtype=torch.uint8)})
        atomic_json(output/'history.json',history)
    for epoch in range(first,max_epochs+1):
        if history and stop_overfit([r['fixed_bank_loss'] for r in history],settings.overfit_loss_ceiling)[0]:break
        model.train();order=torch.randperm(len(batch['q']),device=batch['dem'].device)
        losses=[];gradients=[]
        for indices in order.split(settings.overfit_batch_size):
            optimizer.zero_grad(set_to_none=True)
            context=model.encode(batch['dem'][indices],batch['goal'][indices],batch['q'][indices])
            loss=model.noise_loss(batch['action'][indices],batch['mask'][indices],context)
            if not torch.isfinite(loss):raise ValueError('nonfinite overfit training loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite overfit gradient norm')
            gradients.append(float(norm));optimizer.step();losses.append(float(loss.detach()))
        fixed=fixed_bank_loss(model,batch)
        if not np.isfinite(fixed):raise ValueError('nonfinite fixed-bank loss')
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),
                        'fixed_bank_loss':fixed,'gradient_norm_median':float(np.median(gradients))})
        done,reason=stop_overfit([r['fixed_bank_loss'] for r in history],settings.overfit_loss_ceiling)
        if epoch%checkpoint_interval==0 or done or epoch==max_epochs:checkpoint(epoch)
        if epoch==1 or epoch%10==0 or done:
            print('FOLLOWUP_OVERFIT_EPOCH',seed,epoch,round(fixed,6),reason,flush=True)
        if done:break
    if not history:raise ValueError('overfit ran no epochs')
    final=history[-1];epoch=final['epoch']
    if not latest.exists() or torch.load(latest,map_location='cpu',weights_only=True)['epoch']!=epoch:
        checkpoint(epoch)
    _save_torch(output/'final.pt',{'state_dict':model.state_dict(),'epoch':epoch,'seed':seed,
                                  'fixed_bank_loss':final['fixed_bank_loss'],
                                  'config_metadata':config.checkpoint_metadata()})
    stopped,reason=stop_overfit([r['fixed_bank_loss'] for r in history],settings.overfit_loss_ceiling)
    report={'training_seed':seed,'epochs_ran':len(history),'initial_fixed_bank_loss':history[0]['fixed_bank_loss'],
            'final_fixed_bank_loss':final['fixed_bank_loss'],
            'fixed_bank_loss_min':min(row['fixed_bank_loss'] for row in history),
            'final_gradient_norm_median':final['gradient_norm_median'],
            'training_converged':final['fixed_bank_loss']<=settings.overfit_loss_ceiling,
            'stop_reason':reason if stopped else 'max_epochs',
            'wall_seconds':time.monotonic()-start,'checkpoint':str(output/'final.pt')}
    atomic_json(output/'summary.json',report)
    return report
