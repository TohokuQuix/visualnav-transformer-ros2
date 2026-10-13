import copy
import numpy as np
import pytest
import torch

from visualnav_transformer.eat_nav.evaluator.inference import Evaluator, load_checkpoint
from visualnav_transformer.eat_nav.evaluator.model import BaselineEvaluator
from visualnav_transformer.eat_nav.evaluator.preprocessing import fit_normalization


def test_checkpoint_roundtrip_and_schema_rejection(tmp_path):
    dem=np.zeros((1,4,2,20,20),np.float32);dem[:,:,1]=1
    state=np.zeros((1,22),np.float32)
    paths=np.arange(16,dtype=np.float32).reshape(1,1,8,2)/20
    stats=fit_normalization(dem,state,paths)
    model=BaselineEvaluator()
    before=Evaluator(model,stats).predict(dem,state,paths)
    base={'schema':'eatnav_baseline_checkpoint_v1','input_schema':'eatnav_baseline_input_v1',
          'label_schema':'eatnav_baseline_labels_v1','label_type':'common_goal_progress',
          'dataset_sha256':'abc','normalization':stats,'model_config':model.config,
          'model_state':model.state_dict(),'label_transform':{'mean':0.,'std':1.}}
    p=tmp_path/'model.pt';torch.save(base,p)
    loaded,_=load_checkpoint(p,expected_label_type='common_goal_progress',expected_dataset_sha256='abc')
    assert np.array_equal(loaded.predict(dem,state,paths),before)
    with pytest.raises(ValueError,match='label'):
        load_checkpoint(p,expected_label_type='realized_reward')
    bad=copy.deepcopy(base);bad['normalization']['path_points']=9
    torch.save(bad,p)
    with pytest.raises(ValueError,match='shape'):
        load_checkpoint(p)
    bad=copy.deepcopy(base);bad['normalization']['state_mean']=[0.]
    torch.save(bad,p)
    with pytest.raises(ValueError,match='normalization'):
        load_checkpoint(p)
