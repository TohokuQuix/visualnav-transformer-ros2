"""Dump the live frozen RL Actor/Critic observation layout without extra computes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import gymnasium as gym
import yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import onix_isaac_lab.tasks
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.phase_c_clone_cfg import make_phase_c_clone_cfg
checkpoint=Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt')
saved=yaml.load((checkpoint.parent/'params'/'env.yaml').read_text(),Loader=yaml.BaseLoader)
toggles={n:saved[n].lower()=='true' for n in ('use_last_action_obs','use_joint_effort_obs','use_height_scan_obs')}
cfg=make_phase_c_clone_cfg(OnixRoughEnvCfg(**toggles),EATConfig(quality_enabled=False),branches=8)
env=RslRlVecEnvWrapper(gym.make('Isaac-Velocity-Rough-Onix-v0',cfg=cfg))
obs,_=env.reset()
manager=env.unwrapped.observation_manager
result={'shapes':{k:list(v.shape) for k,v in obs.items()},'terms':{}}
for group in ('policy','critic'):
    result['terms'][group]=[{'name':n,'shape':list(dim),'noise':type(term.noise).__name__ if term.noise is not None else None,
                             'clip':term.clip,'scale':term.scale}
                            for n,dim,term in zip(manager.active_terms[group],manager.group_obs_term_dim[group],manager._group_obs_term_cfgs[group])]
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
print('PHASE_C_LAYOUT',json.dumps({'shapes':result['shapes'],'critic_terms':len(result['terms']['critic'])}),flush=True)
env.close();app.close()
