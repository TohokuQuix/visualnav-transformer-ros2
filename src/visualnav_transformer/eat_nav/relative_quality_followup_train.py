"""Validation-only, epoch-resumable scratch trainer for follow-up Probes."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .quality_gate_probe import masked_mse
from .relative_quality_followup_probe import FollowupProbe


def _validation_loss(model,data,q,batch_size):
    model.eval();total=0.;count=0
    with torch.no_grad():
        for start in range(0,len(q),batch_size):
            sl=slice(start,start+batch_size)
            prediction=model(data['dem'][sl],data['goal'][sl],q[sl]);target=data['action'][sl]
            total+=float((prediction-target).square().sum());count+=target.numel()
    return total/count


def _save(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temp);temp.replace(path)


def train_followup_probe(train,val,train_q,val_q,*,kind,seed,settings,output,fingerprint):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if train_q.shape[1]!=val_q.shape[1]:raise ValueError('Quality dimension mismatch')
    torch.manual_seed(int(seed));torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    model=FollowupProbe(train_q.shape[1]).to(train['dem'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['probe_learning_rate'])
    latest=output/'latest.pt';bestpath=output/'best.pt'
    best=float('inf');best_epoch=0;bad=0;history=[];start_epoch=1
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['fingerprint']!=fingerprint or saved['kind']!=kind or saved['seed']!=seed:
            raise ValueError('Probe resume fingerprint mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad']
        history=saved['history'];start_epoch=saved['epoch']+1
    for epoch in range(start_epoch,settings['probe_max_epochs']+1):
        generator=torch.Generator(device=train['dem'].device).manual_seed(seed*100000+epoch)
        order=torch.randperm(len(train_q),generator=generator,device=train['dem'].device)
        model.train();losses=[]
        for ix in order.split(settings['probe_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            pred=model(train['dem'][ix],train['goal'][ix],train_q[ix])
            loss=masked_mse(pred,train['action'][ix],train['mask'][ix]);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        validation=_validation_loss(model,val,val_q,settings['probe_batch_size'])
        if not np.isfinite(validation):raise ValueError('nonfinite Probe validation loss')
        improved=validation<best-settings['probe_min_delta']
        if improved:
            best=validation;best_epoch=epoch;bad=0
            _save(bestpath,{'model':model.state_dict(),'kind':kind,'seed':seed,'validation_loss':validation,
                            'epoch':epoch,'fingerprint':fingerprint,'parameter_count':sum(p.numel() for p in model.parameters())})
        else:bad+=1
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),
                        'validation_loss':validation,'best':improved})
        _save(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),'kind':kind,
                      'seed':seed,'fingerprint':fingerprint,'epoch':epoch,'best':best,
                      'best_epoch':best_epoch,'bad':bad,'history':history})
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if epoch==1 or epoch%10==0 or improved:
            print('FOLLOWUP_PROBE_EPOCH',kind,seed,epoch,round(validation,6),flush=True)
        if bad>=settings['probe_patience']:break
    return {'kind':kind,'seed':seed,'best_validation_loss':best,'best_epoch':best_epoch,
            'epochs_ran':len(history),
            'validation_improving_at_cap':len(history)==settings['probe_max_epochs'] and best_epoch>=len(history)-2,
            'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint':str(bestpath)}


def predict_followup(checkpoint,data,q,batch_size):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=FollowupProbe(q.shape[1]).to(data['dem'].device).eval();model.load_state_dict(payload['model'])
    with torch.no_grad():
        return torch.cat([model(data['dem'][i:i+batch_size],data['goal'][i:i+batch_size],q[i:i+batch_size])
                          for i in range(0,len(q),batch_size)],dim=0)
