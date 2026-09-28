"""Equal-graph deterministic trajectory probes for frozen post-hoc labels."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .quality_gate_probe import DeterministicProbe,masked_mse


class RelativeProbe(DeterministicProbe):
    def __init__(self,quality_dim:int=1):
        if quality_dim not in (1,4):raise ValueError('quality_dim must be 1 or 4')
        super().__init__();self.quality_dim=quality_dim
        if quality_dim==4:self.quality_mlp[0]=nn.Linear(4,64)

    def forward(self,dem,goal,quality):
        if quality.shape!=(len(dem),self.quality_dim):raise ValueError('quality shape mismatch')
        if self.quality_dim==1:return super().forward(dem,goal,quality)
        if dem.shape[1:]!=(4,2,20,20) or goal.shape!=(len(dem),2):raise ValueError('probe input shape mismatch')
        frames=self.dem_cnn(dem.reshape(-1,2,20,20)).reshape(len(dem),4,64)
        tokens=torch.cat((frames,self.goal_mlp(goal).unsqueeze(1),self.quality_mlp(quality).unsqueeze(1)),dim=1)+self.position
        return self.head(self.transformer(tokens).mean(dim=1)).reshape(len(dem),8,2)


def quality_tensor(data,kind:str,device):
    n=len(data['quality_abs'])
    if kind=='probe_0':return torch.full((n,1),.5,dtype=torch.float32,device=device)
    if kind=='probe_abs':fields=('quality_abs',)
    elif kind=='probe_adv':fields=('z_advantage_h',)
    elif kind=='probe_drawdown':fields=('z_drawdown',)
    elif kind=='probe_multi':fields=('z_v0','z_advantage_h','z_drawdown','z_recovery')
    else:raise ValueError(f'unknown probe kind {kind}')
    values=np.column_stack([np.asarray(data[name],dtype=np.float32) for name in fields])
    if not np.isfinite(values).all():raise ValueError('nonfinite Quality input')
    return torch.from_numpy(values).to(device)


def tensorize(data,action_mean,action_std,device):
    dem=torch.from_numpy(np.asarray(data['dem'],dtype=np.float32)).to(device)
    goal=torch.from_numpy(np.asarray(data['goal'],dtype=np.float32)).to(device)
    action=torch.from_numpy((np.asarray(data['action'],dtype=np.float32)-action_mean)/action_std).to(device)
    if dem.shape[1:]!=(4,2,20,20) or goal.shape!=(len(dem),2) or action.shape!=(len(dem),8,2):
        raise ValueError('relative probe dataset shape mismatch')
    if not torch.isfinite(dem).all() or not torch.isfinite(goal).all() or not torch.isfinite(action).all():
        raise ValueError('nonfinite probe data')
    return {'dem':dem,'goal':goal,'action':action,'mask':torch.ones((len(dem),8),dtype=torch.bool,device=device),
            'terrain':np.asarray(data['terrain']),'episode':[tuple(e) for e in data['episode']],
            'index':np.asarray(data['index'])}


def _validation_loss(model,data,q,batch_size):
    model.eval();total=0.;count=0
    with torch.no_grad():
        for start in range(0,len(q),batch_size):
            sl=slice(start,start+batch_size)
            pred=model(data['dem'][sl],data['goal'][sl],q[sl]);target=data['action'][sl]
            total+=float((pred-target).square().sum());count+=int(np.prod(target.shape))
    return total/count


def _save_torch(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temp);temp.replace(path)


def train_relative_probe(train,val,train_q,val_q,*,kind,seed,settings,output,fingerprint):
    """Validation-selected scratch training, resumable at epoch boundaries."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(int(seed));torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    quality_dim=train_q.shape[1];model=RelativeProbe(quality_dim).to(train['dem'].device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['probe_learning_rate'])
    latest=output/'latest.pt';bestpath=output/'best.pt'
    best=float('inf');best_epoch=0;bad=0;history=[];start_epoch=1
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        if saved['fingerprint']!=fingerprint or saved['kind']!=kind or saved['seed']!=seed:
            raise ValueError('Probe resume fingerprint mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        best=saved['best'];best_epoch=saved['best_epoch'];bad=saved['bad'];history=saved['history'];start_epoch=saved['epoch']+1
    for epoch in range(start_epoch,settings['probe_max_epochs']+1):
        generator=torch.Generator(device=train['dem'].device).manual_seed(seed*100000+epoch)
        order=torch.randperm(len(train_q),generator=generator,device=train['dem'].device)
        model.train();losses=[]
        for ix in order.split(settings['probe_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            pred=model(train['dem'][ix],train['goal'][ix],train_q[ix])
            loss=masked_mse(pred,train['action'][ix],train['mask'][ix])
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        validation=_validation_loss(model,val,val_q,settings['probe_batch_size'])
        if not np.isfinite(validation):raise ValueError('nonfinite Probe validation loss')
        improved=validation<best-settings['probe_min_delta']
        if improved:
            best=validation;best_epoch=epoch;bad=0
            _save_torch(bestpath,{'model':model.state_dict(),'kind':kind,'seed':seed,
                                  'validation_loss':validation,'epoch':epoch,'fingerprint':fingerprint,
                                  'parameter_count':sum(p.numel() for p in model.parameters())})
        else:bad+=1
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':validation,'best':improved})
        _save_torch(latest,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                            'kind':kind,'seed':seed,'fingerprint':fingerprint,'epoch':epoch,
                            'best':best,'best_epoch':best_epoch,'bad':bad,'history':history})
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if epoch==1 or epoch%10==0 or improved:
            print('RELATIVE_PROBE_EPOCH',kind,seed,epoch,round(validation,6),flush=True)
        if bad>=settings['probe_patience']:break
    return {'kind':kind,'seed':seed,'best_validation_loss':best,'best_epoch':best_epoch,
            'epochs_ran':len(history),'validation_improving_at_cap':len(history)==settings['probe_max_epochs'] and best_epoch>=len(history)-2,
            'parameter_count':sum(p.numel() for p in model.parameters()),'checkpoint':str(bestpath)}


def predict(checkpoint,data,quality,batch_size):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=RelativeProbe(quality.shape[1]).to(data['dem'].device).eval();model.load_state_dict(payload['model'])
    with torch.no_grad():
        return torch.cat([model(data['dem'][i:i+batch_size],data['goal'][i:i+batch_size],quality[i:i+batch_size])
                          for i in range(0,len(quality),batch_size)],dim=0)


def permutation_comparison(observed_loss:float,shuffled_losses) -> dict:
    """Positive difference means the observed Quality input beats shuffled inputs."""
    differences=[float(x-observed_loss) for x in shuffled_losses]
    if not differences or not np.isfinite(differences).all():raise ValueError('invalid permutation losses')
    return {'observed_mse':float(observed_loss),'shuffle_mse':[float(x) for x in shuffled_losses],
            'observed_minus_shuffle_gains':differences,
            'mean_observed_minus_shuffle_gain':float(np.mean(differences)),
            'observed_outperforms_all_shuffles':bool(all(x>0 for x in differences))}
