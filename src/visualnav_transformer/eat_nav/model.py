"""Scratch EAT policy with shared DEM CNN and NoMaD diffusion components."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from visualnav_transformer.train.vint_train.models.vint.self_attention import PositionalEncoding
from .config import EATConfig


class EATNav(nn.Module):
    def __init__(self, config:EATConfig):
        super().__init__()
        self.config=config
        self.dem_cnn=nn.Sequential(
            nn.Conv2d(2,16,3,padding=1),nn.GELU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.GELU(),
            nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(64,256),
        )
        self.goal_mlp=nn.Sequential(nn.Linear(2,128),nn.GELU(),nn.Linear(128,256))
        self.quality_mlp=nn.Sequential(nn.Linear(1,128),nn.GELU(),nn.Linear(128,256)) if config.quality_enabled else None
        self.positional_encoding=PositionalEncoding(256,max_seq_len=config.dem_frame_count+1+int(config.quality_enabled))
        layer=nn.TransformerEncoderLayer(d_model=256,nhead=4,dim_feedforward=1024,
                                         activation='gelu',batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=4,enable_nested_tensor=False)
        self.noise_pred_net=ConditionalUnet1D(input_dim=2,global_cond_dim=256,
                                               down_dims=[64,128,256],cond_predict_scale=False)
        self.scheduler=DDPMScheduler(num_train_timesteps=10,beta_schedule='squaredcos_cap_v2',
                                     clip_sample=True,prediction_type='epsilon')
        self.register_buffer('action_mean',torch.zeros(2,dtype=torch.float32))
        self.register_buffer('action_std',torch.ones(2,dtype=torch.float32))

    def set_action_statistics(self,mean,std):
        mean=torch.as_tensor(mean,device=self.action_mean.device,dtype=self.action_mean.dtype)
        std=torch.as_tensor(std,device=self.action_std.device,dtype=self.action_std.dtype)
        if mean.shape!=(2,) or std.shape!=(2,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():
            raise ValueError('action mean/std must be finite XY vectors with positive std')
        self.action_mean.copy_(mean)
        self.action_std.copy_(std)

    def encode_dem_frames(self,dem):
        if dem.ndim!=5 or dem.shape[1:]!=(self.config.dem_frame_count,2,self.config.dem_height_cells,self.config.dem_width_cells):
            raise ValueError('DEM history shape mismatch')
        batch,frames=dem.shape[:2]
        return self.dem_cnn(dem.reshape(batch*frames,2,self.config.dem_height_cells,self.config.dem_width_cells)).reshape(batch,frames,256)

    def encode_tokens(self,dem,goal,quality):
        tokens=[self.encode_dem_frames(dem),self.goal_mlp(goal.float()).unsqueeze(1)]
        if self.config.quality_enabled:
            if quality is None:
                raise ValueError('quality target required when quality conditioning is enabled')
            q=torch.as_tensor(quality,dtype=goal.dtype,device=goal.device).reshape(goal.shape[0],1)
            tokens.append(self.quality_mlp(q).unsqueeze(1))
        return torch.cat(tokens,dim=1)

    def encode(self,dem,goal,quality):
        tokens=self.positional_encoding(self.encode_tokens(dem,goal,quality))
        return self.transformer(tokens).mean(dim=1)

    def noise_loss(self,action_delta,valid_mask,context,*,noise=None,timesteps=None):
        if action_delta.ndim!=3 or action_delta.shape[1:]!=(self.config.len_traj_pred,2):
            raise ValueError('action delta shape mismatch')
        if valid_mask.shape!=action_delta.shape[:2] or context.shape!=(len(action_delta),256):
            raise ValueError('action mask/context shape mismatch')
        valid_mask=valid_mask.bool()
        if not valid_mask.any():
            raise ValueError('batch has no valid future action')
        if noise is None:noise=torch.randn_like(action_delta)
        if timesteps is None:
            timesteps=torch.randint(0,self.scheduler.config.num_train_timesteps,(len(action_delta),),device=action_delta.device)
        masked=valid_mask.unsqueeze(-1)
        action=((action_delta-self.action_mean)/self.action_std)*masked
        target_noise=noise*masked
        noisy=self.scheduler.add_noise(action,target_noise,timesteps)
        prediction=self.noise_pred_net(sample=noisy,timestep=timesteps,global_cond=context)
        squared=F.mse_loss(prediction,target_noise,reduction='none')*masked
        return squared.sum()/(2*valid_mask.sum())

    @torch.no_grad()
    def sample(self,dem,goal,quality,*,seed=None,initial_noise=None):
        context=self.encode(dem,goal,quality)
        generator=None
        if seed is not None:
            generator=torch.Generator(device=context.device).manual_seed(int(seed))
        shape=(len(dem),self.config.len_traj_pred,2)
        if initial_noise is None:
            action=torch.randn(shape,dtype=context.dtype,device=context.device,generator=generator)
        else:
            if initial_noise.shape!=shape:
                raise ValueError('initial diffusion noise shape mismatch')
            action=initial_noise.clone()
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps,device=context.device)
        for timestep in self.scheduler.timesteps:
            predicted=self.noise_pred_net(sample=action,timestep=timestep,global_cond=context)
            action=self.scheduler.step(predicted,timestep,action,generator=generator).prev_sample
        delta=action*self.action_std+self.action_mean
        return torch.cumsum(delta,dim=1)
