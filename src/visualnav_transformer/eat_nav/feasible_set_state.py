"""In-memory full-scene snapshot for same-env serial counterfactual replay.

PhysX contact warm-start caches and sensor internals are not serializable here.
Nominal duplicates measure the resulting replay variation before outcomes count.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch


def _tensor_fields(env):
    fields=[]
    def add(owner,name,label=None):
        value=getattr(owner,name,None)
        if isinstance(value,torch.Tensor):fields.append((owner,name,label or name))
    for name in ('episode_length_buf','reset_terminated','reset_time_outs','_latency_substeps',
                 '_onix_latency_seconds','_latency_policy_obs_prev','_latency_policy_obs_reset_mask',
                 '_terrain_path_length_xy','_terrain_path_initialized','_terrain_prev_root_pos_xy',
                 '_onix_flipper_fold_mode'):
        add(env,name,'env.'+name)
    command=env.command_manager._terms['base_velocity']
    for name in ('vel_command_b','heading_target','is_heading_env','is_standing_env','time_left','command_counter'):
        add(command,name,'command.'+name)
    manager=env.action_manager
    for name in ('_action','_prev_action'):
        add(manager,name,'action_manager.'+name)
    for term_name,term in manager._terms.items():
        for name in ('_raw_actions','_processed_actions','_pre_clip_processed_actions',
                     '_previous_targets','_travel_direction'):
            add(term,name,f'action.{term_name}.{name}')
    for name,buffer in getattr(env,'_action_latency_buffers',{}).items():
        add(buffer,'_time_lags',f'delay.{name}.time_lags')
        ring=buffer._circular_buffer
        add(ring,'_num_pushes',f'delay.{name}.num_pushes')
        add(ring,'_buffer',f'delay.{name}.ring')
    history=getattr(env.observation_manager,'_group_obs_term_history_buffer',{})
    for group,terms in history.items():
        for name,ring in terms.items():
            add(ring,'_num_pushes',f'obs_history.{group}.{name}.num_pushes')
            add(ring,'_buffer',f'obs_history.{group}.{name}.ring')
    return fields


class SerialStateSnapshot:
    """Copy visible state of all envs, including RNG and delay ring pointers."""
    def __init__(self,env):
        self.env=env
        robot=env.scene['robot']
        self.root=robot.data.root_state_w.detach().clone()
        self.joint_pos=robot.data.joint_pos.detach().clone()
        self.joint_vel=robot.data.joint_vel.detach().clone()
        self.tensors=[(owner,name,label,getattr(owner,name).detach().clone())
                      for owner,name,label in _tensor_fields(env)]
        self.scalars=[]
        for name,buffer in getattr(env,'_action_latency_buffers',{}).items():
            ring=buffer._circular_buffer
            self.scalars.append((ring,'_pointer',f'delay.{name}.pointer',ring._pointer))
        history=getattr(env.observation_manager,'_group_obs_term_history_buffer',{})
        for group,terms in history.items():
            for name,ring in terms.items():
                self.scalars.append((ring,'_pointer',f'obs_history.{group}.{name}.pointer',ring._pointer))
        for name in ('common_step_counter',):
            if hasattr(env,name):self.scalars.append((env,name,'env.'+name,copy.deepcopy(getattr(env,name))))
        command=env.command_manager._terms['base_velocity']
        sampler=getattr(command,'sampler',None)
        self.sampler=sampler
        if sampler is not None:
            self.sampler_states=[copy.deepcopy(rng.bit_generator.state) for rng in sampler.rngs]
            self.sampler_held=copy.deepcopy(sampler._states)
            self.sampler_times=copy.deepcopy(sampler._last_times)
        self.initial_rngs=getattr(env,'eat_nav_initial_rngs',None)
        if self.initial_rngs is not None:
            self.initial_rng_states=[copy.deepcopy(rng.bit_generator.state) for rng in self.initial_rngs]
        self.python_rng=random.getstate()
        self.numpy_rng=np.random.get_state()
        self.cpu_rng=torch.get_rng_state().clone()
        self.cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        self.unreplicated=['PhysX_contact_warm_start_cache','contact_sensor_history',
                           'raycaster_history','simulation_clock','ROS_follower_process_state']

    @torch.inference_mode()
    def restore(self):
        env=self.env
        robot=env.scene['robot']
        ids=torch.arange(env.num_envs,device=env.device)
        robot.write_root_state_to_sim(self.root,env_ids=ids)
        robot.write_joint_state_to_sim(self.joint_pos,self.joint_vel,env_ids=ids)
        for owner,name,label,value in self.tensors:
            current=getattr(owner,name)
            if not isinstance(current,torch.Tensor) or current.shape!=value.shape:
                raise ValueError(f'serial state field changed shape: {label}')
            current.copy_(value)
        for owner,name,label,value in self.scalars:
            setattr(owner,name,copy.deepcopy(value))
        if self.sampler is not None:
            for rng,state in zip(self.sampler.rngs,self.sampler_states):
                rng.bit_generator.state=copy.deepcopy(state)
            self.sampler._states=copy.deepcopy(self.sampler_held)
            self.sampler._last_times=copy.deepcopy(self.sampler_times)
        if self.initial_rngs is not None:
            for rng,state in zip(self.initial_rngs,self.initial_rng_states):
                rng.bit_generator.state=copy.deepcopy(state)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:torch.cuda.set_rng_state_all(self.cuda_rng)
        env.scene.write_data_to_sim()
        env.sim.forward()
        root_error=float((robot.data.root_state_w-self.root).abs().max())
        joint_error=float((robot.data.joint_pos-self.joint_pos).abs().max())
        if max(root_error,joint_error)>1e-4:
            raise ValueError(f'serial restore inconsistency: root={root_error}, joints={joint_error}')
        return {'root_max_abs':root_error,'joint_max_abs':joint_error,
                'copied_fields':[label for _,_,label,_ in self.tensors],
                'unreplicated':list(self.unreplicated)}
