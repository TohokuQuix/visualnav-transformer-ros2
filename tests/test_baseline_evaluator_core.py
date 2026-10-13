import numpy as np
import pytest
import torch
import builtins
import importlib
import sys

from visualnav_transformer.eat_nav.evaluator.preprocessing import (
    body_local_path, fit_normalization, prepare_inputs, resample_path,
)
from visualnav_transformer.eat_nav.evaluator.model import BaselineEvaluator
from visualnav_transformer.eat_nav.evaluator.inference import Evaluator, NoValidCandidateError


def fixture(k=3):
    dem = np.zeros((2, 4, 2, 20, 20), np.float32)
    dem[:, :, 1] = 1
    state = np.zeros((2, 22), np.float32)
    paths = np.stack([np.stack([np.linspace(0, 1, 8), np.full(8, i * .2)], -1)
                      for i in range(k)], 0).astype(np.float32)
    return dem, state, np.stack([paths, paths])


def test_path_transform_and_resample():
    path = np.array([[1., 2.], [2., 2.]])
    local = body_local_path(path, robot_xy=(1., 1.), yaw_rad=np.pi / 2)
    assert np.allclose(local, [[1., 0.], [1., -1.]], atol=1e-6)
    sampled = resample_path(local, 8)
    assert sampled.shape == (8, 2)
    assert np.allclose(sampled[[0, -1]], local)


def test_preprocessing_mask_and_nonfinite():
    dem, state, paths = fixture()
    dem[0, 0, 0, 0, 0] = np.nan
    dem[0, 0, 1, 0, 0] = 0
    stats = fit_normalization(dem, state, paths)
    inputs = prepare_inputs(dem, state, paths, stats)
    assert inputs['dem'].shape == (2, 8, 20, 20)
    assert inputs['dem'][0, 0, 0, 0] == 0
    assert inputs['dem'][0, 1, 0, 0] == 0
    dem[0, 0, 0, 0, 1] = np.nan
    with pytest.raises(ValueError, match='DEM'):
        prepare_inputs(dem, state, paths, stats)


def test_dem_center_crop_preserves_center_cell():
    dem,state,paths=fixture()
    big=np.zeros((2,4,2,24,24),np.float32);big[:,:,1]=1
    big[:,:,0,12,12]=.5
    stats=fit_normalization(big,state,paths)
    values=prepare_inputs(big,state,paths,stats)['dem']
    assert values.shape==(2,8,20,20)
    assert values[0,0,10,10] > values[0,0,0,0]


def test_model_permutation_mask_and_backward():
    torch.manual_seed(2)
    dem, state, paths = fixture()
    stats = fit_normalization(dem, state, paths)
    model = BaselineEvaluator()
    api = Evaluator(model, stats)
    scores = api.predict(dem, state, paths)
    assert scores.shape == (2, 3)
    assert np.allclose(api.predict(dem, state, paths[:, [2, 0, 1]]), scores[:, [2, 0, 1]])
    assert np.allclose(api.predict(dem, state, paths), scores)
    masked = api.predict(dem, state, paths, candidate_mask=np.array([[1, 0, 1], [1, 1, 1]], bool))
    assert np.isneginf(masked[0, 1])
    with pytest.raises(NoValidCandidateError):
        api.predict(dem, state, paths, candidate_mask=np.zeros((2, 3), bool))
    tensors = prepare_inputs(dem, state, paths, stats)
    pred = model(torch.from_numpy(tensors['dem']), torch.from_numpy(tensors['state']),
                 torch.from_numpy(tensors['paths']))
    pred.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


def test_variable_candidate_count():
    dem, state, paths = fixture(k=5)
    stats = fit_normalization(dem, state, paths)
    assert Evaluator(BaselineEvaluator(), stats).predict(dem, state, paths).shape == (2, 5)


def test_core_import_does_not_require_ros_or_isaac(monkeypatch):
    original=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name.split('.')[0] in {'rclpy','tf2_ros','isaaclab','omni','pxr'}:
            raise AssertionError('ROS or Isaac import in evaluator core: '+name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    importlib.reload(sys.modules['visualnav_transformer.eat_nav.evaluator.preprocessing'])
    importlib.reload(sys.modules['visualnav_transformer.eat_nav.evaluator.model'])
    importlib.reload(sys.modules['visualnav_transformer.eat_nav.evaluator.inference'])
