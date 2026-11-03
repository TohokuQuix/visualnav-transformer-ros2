"""Load the checkpoint-bound command-free R5/R4 predictor without simulator imports."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .inputs import finite

LEGACY_CHECKPOINT='docs/eat-nav/all-fields-learning-20261023/models/random_rough/seed123/checkpoint.pt'
DEFAULT_CHECKPOINT='docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/models/random_rough/seed123/checkpoint.pt'


def bootstrap(repository):
    root=Path(repository).expanduser().resolve()
    if not any((root/'checkpoints/eat_nav'/name).is_file() for name in
               ('fall_stall_r4_index.json','fall_stall_r4star_r5_index.json')):
        raise ValueError('repository must point to visualnav-transformer-ros2')
    for child in ('src','scripts'):
        value=str(root/child)
        if value not in sys.path:sys.path.insert(0,value)
    return root


class Evaluator:
    def __init__(self,repository,checkpoint=None,expected_sha='',navigation_mode='r5_14'):
        self.root=bootstrap(repository)
        self.navigation_mode=navigation_mode
        if navigation_mode=='r5_14':
            from eat_nav_r4_star14 import candidate_paths,pack_inputs,CANDIDATE_IDS,DIRECTIONS
            from eat_nav_r5_14 import R5FourteenGuard
            self.guard_class=R5FourteenGuard;self.condition='R5'
            index_name='fall_stall_r4star_r5_index.json';default=DEFAULT_CHECKPOINT
            self.required_heads=(2,3)
        elif navigation_mode=='r4':
            from eat_nav_risk_navigation import candidate_paths,pack_inputs,CANDIDATE_IDS
            from eat_nav_r4 import R4Guard
            DIRECTIONS=np.asarray([1,1,1,-1],dtype=np.float32)
            self.guard_class=R4Guard;self.condition='R4'
            index_name='fall_stall_r4_index.json';default=LEGACY_CHECKPOINT
            self.required_heads=(0,1,2,3)
        else:raise ValueError('navigation_mode must be r5_14 or r4')
        self.bank=candidate_paths();self.pack_inputs=pack_inputs
        self.candidate_ids=tuple(CANDIDATE_IDS);self.directions=np.asarray(DIRECTIONS).copy()
        path=Path(checkpoint or default).expanduser()
        if not path.is_absolute():path=self.root/path
        self.path=path.resolve()
        index=json.loads((self.root/'checkpoints/eat_nav'/index_name).read_text())
        if not expected_sha:
            entries=[x for x in index['checkpoints'] if (self.root/x['path']).resolve()==self.path]
            if len(entries)!=1:raise ValueError('checkpoint not listed in mode index; explicit SHA-256 required')
            expected_sha=entries[0]['sha256']
        self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.sha!=expected_sha:raise ValueError('checkpoint SHA-256 mismatch')
        import torch
        from eat_nav_command_free_learning import load_checkpoint
        torch.set_num_threads(1)
        model,payload=load_checkpoint(self.path)
        if payload['variant']!='dem_state_candidate' or not all(payload['support']['active'][i] for i in self.required_heads) or \
                not all(n>0 for n in payload['normalization']['progress_count']):
            raise ValueError('unsupported checkpoint heads or normalization')
        if navigation_mode=='r5_14' and tuple(payload.get('config',{}).get('training_candidate_ids',()))!=self.candidate_ids:
            raise ValueError('R5 requires the checkpoint trained with the fourteen-candidate bank')
        self.fit=dict(payload,model=model)

    def predict(self,dem,actor,body):
        from eat_nav_command_free_learning import predict
        p,q=predict(self.fit,self.pack_inputs(dem,actor,body),[0])
        shape=(len(self.bank),4)
        p=finite(p[0],shape);q=finite(q[0],shape)
        if ((p<0)|(p>1)).any():raise ValueError('invalid model probabilities')
        return p,q


def path_poses(points,pose,direction):
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or not np.isfinite(points).all():
        raise ValueError('invalid path')
    if direction not in (-1,1):raise ValueError('invalid path direction')
    x,y,yaw=finite(pose,(3,));c,s=np.cos(yaw),np.sin(yaw)
    xy=points@np.array([[c,s],[-s,c]])+np.array([x,y])
    delta=np.diff(xy,axis=0)
    headings=np.arctan2(delta[:,1],delta[:,0])+(np.pi if direction<0 else 0)
    headings=np.arctan2(np.sin(headings),np.cos(headings))
    return xy,np.r_[headings,headings[-1]]
