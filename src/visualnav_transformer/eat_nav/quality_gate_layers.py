"""Activation and paired-noise tracing through the unchanged EAT network."""
from __future__ import annotations
import numpy as np
import torch


def trace_network_layers(model, dem, goal, action, *, seed=11):
    model.eval()
    device=next(model.parameters()).device
    dem=dem.to(device);goal=goal.to(device);action=action.to(device)
    generator=torch.Generator(device=device).manual_seed(seed)
    fixed_noise=torch.randn(action.shape,device=device,generator=generator)
    normalized=(action-model.action_mean)/model.action_std
    representative=model.scheduler.config.num_train_timesteps//2
    def capture(q):
        found={};hooks=[]
        def hook(name,selector=lambda x:x):
            def fn(_module,_inputs,output):found[name]=selector(output).detach().clone()
            return fn
        hooks.append(model.quality_mlp.register_forward_hook(hook('quality_mlp')))
        hooks.append(model.positional_encoding.register_forward_hook(hook('quality_token_input',lambda x:x[:,-1])))
        hooks.append(model.positional_encoding.register_forward_hook(hook('goal_token_input',lambda x:x[:,-2])))
        for i,layer in enumerate(model.transformer.layers):
            hooks.append(layer.register_forward_hook(hook(f'transformer_layer_{i}_quality',lambda x:x[:,-1])))
            hooks.append(layer.register_forward_hook(hook(f'transformer_layer_{i}_goal',lambda x:x[:,-2])))
        unet=model.noise_pred_net
        for i,blocks in enumerate(unet.down_modules):
            hooks.append(blocks[0].cond_encoder.register_forward_hook(hook(f'film_down_{i}')))
            hooks.append(blocks[1].register_forward_hook(hook(f'down_{i}')))
        for i,block in enumerate(unet.mid_modules):
            hooks.append(block.cond_encoder.register_forward_hook(hook(f'film_mid_{i}')))
            hooks.append(block.register_forward_hook(hook(f'mid_{i}')))
        for i,blocks in enumerate(unet.up_modules):
            hooks.append(blocks[0].cond_encoder.register_forward_hook(hook(f'film_up_{i}')))
            hooks.append(blocks[1].register_forward_hook(hook(f'up_{i}')))
        hooks.append(unet.final_conv.register_forward_hook(hook('final_conv')))
        with torch.no_grad():
            quality=torch.full((len(dem),1),q,device=device)
            context=model.encode(dem,goal,quality)
            found['final_context']=context.detach().clone()
            steps=torch.full((len(dem),),representative,device=device,dtype=torch.long)
            noisy=model.scheduler.add_noise(normalized,fixed_noise,steps)
            prediction=unet(sample=noisy,timestep=steps,global_cond=context)
            found['final_predicted_noise']=prediction.detach().clone()
        for item in hooks:item.remove()
        return found,noisy
    low,noisy_low=capture(.1)
    middle,noisy_middle=capture(.5)
    high,noisy_high=capture(.95)
    keys=set(low)&set(middle)&set(high)
    sensitivity={name:float(torch.linalg.vector_norm(high[name]-low[name])/
                             (torch.linalg.vector_norm(middle[name])+1e-8)) for name in sorted(keys)}
    with torch.no_grad():
        qlow=torch.full((len(dem),1),.1,device=device)
        qhigh=torch.full((len(dem),1),.95,device=device)
        qmid=torch.full((len(dem),1),.5,device=device)
        low_context=model.encode(dem,goal,qlow)
        high_context=model.encode(dem,goal,qhigh)
        mid_context=model.encode(dem,goal,qmid)
        shifted_context=model.encode(dem,goal+torch.tensor([.1,0.],device=device),qmid)
        epsilon=[]
        for k in range(model.scheduler.config.num_train_timesteps):
            steps=torch.full((len(dem),),k,device=device,dtype=torch.long)
            noisy=model.scheduler.add_noise(normalized,fixed_noise,steps)
            lo=model.noise_pred_net(sample=noisy,timestep=steps,global_cond=low_context)
            hi=model.noise_pred_net(sample=noisy,timestep=steps,global_cond=high_context)
            epsilon.append(float(torch.linalg.vector_norm(hi-lo)))
        initial=fixed_noise.clone()
        outlo=model.sample(dem,goal,qlow,seed=seed,initial_noise=initial)
        outhi=model.sample(dem,goal,qhigh,seed=seed,initial_noise=initial)
        second_generator=torch.Generator(device=device).manual_seed(seed+18)
        second_initial=torch.randn(action.shape,device=device,generator=second_generator)
        outmid1=model.sample(dem,goal,qmid,seed=seed,initial_noise=initial)
        outmid2=model.sample(dem,goal,qmid,seed=seed+18,initial_noise=second_initial)
        effect=torch.linalg.vector_norm(outhi-outlo,dim=-1).square().mean(dim=-1).sqrt()
        sampling=torch.linalg.vector_norm(outmid2-outmid1,dim=-1).square().mean(dim=-1).sqrt()
    return {'relative_sensitivity':sensitivity,'epsilon_delta_by_timestep':epsilon,
            'noisy_input_max_difference':float(torch.max(torch.abs(noisy_low-noisy_high))),
            'q_context_delta_norm':float(torch.linalg.vector_norm(high_context-low_context)),
            'goal_0p1m_context_delta_norm':float(torch.linalg.vector_norm(shifted_context-mid_context)),
            'final_trajectory_q_effect_median_m':float(torch.median(effect)),
            'final_trajectory_sampling_median_m':float(torch.median(sampling)),
            'final_effect_sampling_ratio':float(torch.median(effect)/torch.median(sampling).clamp_min(1e-8))}
