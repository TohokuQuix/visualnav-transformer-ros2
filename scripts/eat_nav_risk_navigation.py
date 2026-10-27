"""ROS-free live risk Evaluator adapter; fixed four-candidate bank.

This script is outside the byte-bound collection package intentionally.
Scores below are fixed exploratory utility weights, not calibrated risk costs.
"""
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.bidirectional_bank import candidate
from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import CANDIDATE_IDS
from visualnav_transformer.eat_nav.fall_stall_targets import body_vector
from visualnav_transformer.eat_nav.fall_stall_audit import sha,require
from visualnav_transformer.eat_nav.fall_stall_learning import prepare_inputs
from visualnav_transformer.eat_nav.fall_stall_training import load_checkpoint,predict

REPO=Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT=REPO/'docs/eat-nav/current-dataset-learning-20261021/artifacts/dem_state_candidate-seed123/checkpoint.pt'
CHECKPOINT_SHA='924f6438ca7addb94f48dd358b16fd6cc182d75004c3505340dbe456f5f4d0e7'
RISK_WEIGHTS=np.asarray([2.,.5,1.,.5])


def candidate_paths():
    return np.stack([candidate(d,a).waypoints for d,a in ((1,0),(1,-30),(1,30),(-1,0))]).astype(np.float32)


def body_state(root):
    root=np.asarray(root,dtype=np.float32).tolist()
    require(len(root)==13 and np.isfinite(root).all(),'invalid root state')
    values=[body_vector(root,v) for v in ([0.,0.,-1.],root[7:10],root[10:13])]
    require(all(v is not None for v in values),'invalid root quaternion')
    return np.concatenate(values).astype(np.float32)


def validate_history(stamps,decision_stamp_ns):
    require(len(stamps)==4 and np.array_equal(np.diff(stamps),[250_000_000]*3) and
            0<=decision_stamp_ns-stamps[-1]<250_000_000,'invalid or stale DEM history')


def pack_inputs(dem,actor_state,body):
    dem=np.asarray(dem,dtype=np.float32).copy()
    actor_state=np.asarray(actor_state,dtype=np.float32);body=np.asarray(body,dtype=np.float32)
    require(dem.shape==(4,2,20,20) and actor_state.shape==(22,) and body.shape==(9,),
            'live input shape mismatch')
    mask=dem[:,1]
    require(np.isfinite(mask).all() and np.isin(mask,[0,1]).all() and
            (mask.mean((1,2))>=.25).all(),'invalid DEM mask/coverage')
    require(np.isfinite(dem[:,0][mask==1]).all() and np.isfinite(actor_state).all() and
            np.isfinite(body).all(),'nonfinite live input')
    dem[:,0]=np.where(mask==1,dem[:,0],0)
    return dict(dem=dem[None],actor_state=actor_state[None],body_state=body[None],
        paths=candidate_paths()[None],directions=np.asarray([1,1,1,-1],dtype=np.float32).reshape(1,4,1),
        candidate_mask=np.ones((1,4),dtype=bool))


class OnlineEvaluator:
    def __init__(self,checkpoint):
        require(sha(checkpoint)==CHECKPOINT_SHA,'checkpoint SHA mismatch')
        # Keep CPU inference thread count for exact checkpoint replay. Never
        # change the global determinism flag: Isaac's CUDA buffers share torch.
        torch.set_num_threads(1)
        model,payload=load_checkpoint(checkpoint)
        require(payload['variant']=='dem_state_candidate' and all(payload['support']['active']) and
                all(n>0 for n in payload['normalization']['progress_count']),
                'unsupported live checkpoint')
        self.payload=payload;self.fit=dict(payload,model=model)

    def predict(self,arrays):
        # predict() reuses exactly the trained normalizer and progress units.
        p,progress=predict(self.fit,arrays,[0])
        return p[0],progress[0]


def select_candidate(probabilities,progress,goal_body):
    p=np.asarray(probabilities,dtype=float);progress=np.asarray(progress,dtype=float)
    goal=np.asarray(goal_body,dtype=float)
    require(p.shape==(4,4) and progress.shape==(4,4) and np.isfinite(p).all() and
            np.isfinite(progress).all() and ((p>=0)&(p<=1)).all(),'invalid prediction')
    require(goal.shape==(2,) and np.isfinite(goal).all(),'invalid goal')
    endpoints=candidate_paths()[:,-1].astype(float)
    unit=endpoints/np.linalg.norm(endpoints,axis=1,keepdims=True)
    # Progress is distance along a continuing candidate, not predicted endpoint XY.
    # This is explicitly a heuristic goal-progress proxy, capped at 1m/s * 10s.
    travel=np.clip(progress[:,3],0.,10.)
    goal_proxy=np.linalg.norm(goal)-np.linalg.norm(goal[None]-unit*travel[:,None],axis=1)
    penalty=p@RISK_WEIGHTS;scores=goal_proxy-penalty
    selected=int(np.argmax(scores))
    return dict(selected_index=selected,selected_candidate=CANDIDATE_IDS[selected],
        goal_only_index=int(np.argmax(goal_proxy)),scores=scores.tolist(),goal_progress_proxy_m=goal_proxy.tolist(),
        risk_penalty=penalty.tolist(),risk_weights=RISK_WEIGHTS.tolist(),
        probabilities=p.tolist(),progress_m=progress.tolist(),
        scoring='goal_distance_reduction_proxy_minus_weighted_risk; first_frozen_candidate_tie')
