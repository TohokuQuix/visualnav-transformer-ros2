from isaaclab.app import AppLauncher
app=AppLauncher(headless=True,device='cpu').app
import json,sys
from pathlib import Path
import onix_isaac_lab.tasks
from onix_isaac_lab.tasks.locomotion.velocity.config.onix.rough_env_cfg import OnixRoughEnvCfg
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.demo_isaac_env_cfg import make_collection_cfg
from visualnav_transformer.eat_nav.demo_reset_policy import configure_demo_reset_policy
from visualnav_transformer.eat_nav.mi_fi_v1 import configure_initialization
base=OnixRoughEnvCfg(use_last_action_obs=False,use_joint_effort_obs=True,use_height_scan_obs=False)
cfg=make_collection_cfg(base,load_eat_config('/tmp/eatnav-single-dem-pose2s-6251fac2ec/eat_nav.yaml'),123,30)
configure_demo_reset_policy(cfg);configure_initialization(cfg,'front_folded')
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if x is None or isinstance(x,(str,int,float,bool)):return x
 if callable(x):return getattr(x,'__module__','')+':'+getattr(x,'__name__',str(x))
 return str(x)
Path('/tmp/eatnav_lr_current_cfg.json').write_text(json.dumps({'base':clean(base.to_dict()),'collection':clean(cfg.to_dict())},indent=2))
print('CONFIG_DUMP_COMPLETE',flush=True)
app.close()
