"""Quality routing for post-hoc Probe ablations; original Probe stays untouched."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .quality_gate_probe import DeterministicProbe

FIELDS={
    'probe_delta':('z_delta_v',),
    'probe_recovery':('z_recovery',),
    'probe_return':('z_return_h',),
    'probe_delta_drawdown':('z_delta_v','z_drawdown'),
    'probe_relative_triplet':('z_delta_v','z_drawdown','z_recovery'),
    'probe_multi_reward_free':('z_v0','z_delta_v','z_drawdown','z_recovery'),
    'probe_multi_no_v0':('z_advantage_h','z_drawdown','z_recovery'),
}


def followup_quality_tensor(data,kind:str,device):
    if kind=='probe_0':
        return torch.full((len(data['quality_abs']),1),.5,dtype=torch.float32,device=device)
    if kind=='probe_abs':fields=('quality_abs',)
    elif kind in FIELDS:fields=FIELDS[kind]
    else:raise ValueError(f'unknown quality input: {kind}')
    result=np.column_stack([np.asarray(data[field],dtype=np.float32) for field in fields])
    if not np.isfinite(result).all():raise ValueError('nonfinite Quality input')
    return torch.from_numpy(result).to(device)


class FollowupProbe(DeterministicProbe):
    """Existing Probe graph with only Quality MLP first-layer width changed."""
    def __init__(self,quality_dim:int=1):
        if quality_dim not in (1,2,3,4):raise ValueError('quality dimension must be 1-4')
        super().__init__();self.quality_dim=quality_dim
        if quality_dim>1:self.quality_mlp[0]=nn.Linear(quality_dim,64)

    def forward(self,dem,goal,quality):
        if quality.shape!=(len(dem),self.quality_dim):raise ValueError('quality shape mismatch')
        if self.quality_dim==1:return super().forward(dem,goal,quality)
        if dem.shape[1:]!=(4,2,20,20) or goal.shape!=(len(dem),2):
            raise ValueError('probe input shape mismatch')
        frames=self.dem_cnn(dem.reshape(-1,2,20,20)).reshape(len(dem),4,64)
        tokens=torch.cat((frames,self.goal_mlp(goal).unsqueeze(1),
                          self.quality_mlp(quality).unsqueeze(1)),dim=1)+self.position
        return self.head(self.transformer(tokens).mean(dim=1)).reshape(len(dem),8,2)
