"""ROS-free R6 planner boundary: explicit2s outputs and frozen profile selection.
This module publishes no paths or velocity commands. The caller must supply
causal sensor inputs and measured body/flipper dimensions in metres.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .r6_model import load_checkpoint,predict
from .r6_policy import Footprint,select_candidate
from .r6_provenance import verified_freeze_data,verify_execution_freeze,runtime_source_hashes


def verified_freeze(path):
 return verified_freeze_data(json.loads(Path(path).read_text()))


class R6Planner:
 def __init__(self,checkpoint,frozen_selection,*,rank=1,mode='pose',expected_sha256):
  if rank not in (1,2,3):raise ValueError('explicit frozen top1/2/3 rank required')
  if hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()!=expected_sha256:raise ValueError('checkpoint SHA mismatch')
  self.model,self.payload=load_checkpoint(checkpoint)
  if not self.model.use_dem:raise ValueError('state-only diagnostic model is not deployable R6 predictor')
  frozen=verify_execution_freeze(verified_freeze(frozen_selection),expected_sha256,runtime_source_hashes(Path(__file__).resolve().parents[3]))
  self.profile=frozen['profiles'][rank-1];self.mode=mode;self.frozen_sha=frozen['freeze_sha256']
 def select(self,arrays,goal_body,*,measured_footprint):
  if not isinstance(measured_footprint,Footprint):raise ValueError('measured real body/flipper footprint required')
  from .bidirectional_bank import bank
  reference=bank();paths=np.stack([c.waypoints for c in reference]).astype(np.float32);directions=np.asarray([c.direction for c in reference],np.float32).reshape(14,1)
  if not np.array_equal(arrays['paths'][0],paths) or not np.array_equal(arrays['directions'][0],directions):raise ValueError('R6 deployment candidate bank geometry/order mismatch')
  prediction=predict(self.model,self.payload['normalization'],arrays,[0])
  choice=select_candidate(prediction['trajectory'][0],prediction['events'][0],arrays['dem'][0,-1],goal_body,self.profile,footprint=measured_footprint,mode=self.mode,event_support=self.payload['support']['active'],candidate_mask=arrays['candidate_mask'][0],event_horizon_sec=2)
  choice.update(event_horizon_sec=2,trajectory_times_sec=(np.arange(1,21)/10).tolist(),trajectory_time_unit='seconds',predicted_trajectory=prediction['trajectory'][0].tolist(),event_probabilities=prediction['events'][0].tolist(),freeze_sha256=self.frozen_sha)
  return choice
