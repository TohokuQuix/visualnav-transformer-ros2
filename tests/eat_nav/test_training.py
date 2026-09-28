from dataclasses import replace

import numpy as np
import pytest
import torch

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.quality import ActionStatistics, EmpiricalCDF
from visualnav_transformer.eat_nav.model import EATNav
from visualnav_transformer.eat_nav.checkpoint import save_checkpoint, load_checkpoint
from visualnav_transformer.eat_nav.evaluation import evaluate_batch
from visualnav_transformer.eat_nav.visualize import trajectory_plot_data
from visualnav_transformer.train.train_eat_nav import EarlyStopper, train_batch


torch.set_num_threads(1)


def batch(length):
    return {'dem':torch.rand(2,4,2,20,20),'goal':torch.tensor([[.3,0.],[.2,.1]]),
            'quality':torch.tensor([[.1],[.9]]),'action':torch.rand(2,length,2)*.1,
            'action_mask':torch.ones(2,length,dtype=torch.bool)}


@pytest.mark.parametrize('length,quality_enabled',[(8,True),(12,False)])
def test_one_batch_training_and_evaluation_shapes(length,quality_enabled):
    config=replace(EATConfig(),len_traj_pred=length,quality_enabled=quality_enabled)
    model=EATNav(config)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    original=model.dem_cnn[0].weight.detach().clone()
    loss=train_batch(model,opt,batch(length))
    assert np.isfinite(loss)
    assert not torch.equal(original,model.dem_cnn[0].weight)
    report=evaluate_batch(model,batch(length),seed=5)
    assert len(report['pointwise_xy_error_m'])==length
    assert len(trajectory_plot_data(torch.zeros(length,2)))==length


def test_checkpoint_roundtrip_and_all_shape_metadata_mismatch(tmp_path):
    config=EATConfig()
    model=EATNav(config)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    stats=ActionStatistics(np.array([.1,.2],dtype=np.float32),np.array([.3,.4],dtype=np.float32),10)
    ecdf=EmpiricalCDF.fit([1,2,3])
    path=tmp_path/'best.pt'
    save_checkpoint(model,opt,stats,ecdf,config,'split123',path,epoch=2,best_metric=.4,ema_model=model)
    restored,restored_stats,restored_ecdf=load_checkpoint(path,config)
    assert not restored.training
    torch.testing.assert_close(restored.action_mean,torch.tensor([.1,.2]))
    np.testing.assert_allclose(restored_stats.std,[.3,.4])
    assert restored_ecdf.values==(1.,2.,3.)
    for bad in (replace(config,len_traj_pred=12),replace(config,trajectory_interval_sec=.2),
                replace(config,context_size=5),replace(config,dem_resolution_m=.2)):
        with pytest.raises(ValueError,match='mismatch'):
            load_checkpoint(path,bad)


def test_early_stop_tracks_best_and_patience():
    stopper=EarlyStopper(mode='min',patience=2,min_delta=.01)
    assert stopper.update(1.,epoch=1)
    assert stopper.update(.9,epoch=2)
    assert not stopper.update(.895,epoch=3)
    assert not stopper.should_stop
    assert not stopper.update(.91,epoch=4)
    assert stopper.should_stop
    assert stopper.best_epoch==2
