import json
import numpy as np
import pytest
import torch

from visualnav_transformer.eat_nav.quality_gate_config import GateSettings
from visualnav_transformer.eat_nav.quality_followup_training import train_probe_resumable


def tiny_data():
    dem=torch.zeros((4,4,2,20,20));dem[:,:,1]=1
    goal=torch.zeros((4,2));q=torch.tensor([[.1],[.2],[.8],[.9]])
    action=torch.zeros((4,8,2));action[:, :, 0]=torch.tensor([.01,.02,.08,.09]).unsqueeze(1)
    return {'dem':dem,'goal':goal,'q':q,'action':action,'mask':torch.ones((4,8),dtype=torch.bool),
            'terrain':np.array(['rough']*4),'episode':[(0,0,i) for i in range(4)],'index':np.arange(4)}


def test_probe_resumes_without_warm_start_and_preserves_fixed_q(tmp_path):
    settings=GateSettings(max_probe_epochs=2,probe_patience=3,probe_batch_size=4)
    data=tiny_data()
    full=train_probe_resumable(data,data,kind='probe_0',seed=11,shuffle_seed=None,settings=settings,
                               output=tmp_path/'full',fingerprint='same',max_epochs=2)
    train_probe_resumable(data,data,kind='probe_0',seed=11,shuffle_seed=None,settings=settings,
                          output=tmp_path/'resumed',fingerprint='same',max_epochs=1)
    resumed=train_probe_resumable(data,data,kind='probe_0',seed=11,shuffle_seed=None,settings=settings,
                                  output=tmp_path/'resumed',fingerprint='same',max_epochs=2)
    assert full['best_validation_loss']==pytest.approx(resumed['best_validation_loss'],abs=1e-7)
    assert json.loads((tmp_path/'resumed'/'history.json').read_text())==json.loads((tmp_path/'full'/'history.json').read_text())
    with pytest.raises(ValueError,match='fingerprint'):
        train_probe_resumable(data,data,kind='probe_0',seed=11,shuffle_seed=None,settings=settings,
                              output=tmp_path/'resumed',fingerprint='changed',max_epochs=2)


def test_overfit_resume_restores_rng_optimizer_and_rejects_wrong_run(tmp_path):
    from visualnav_transformer.eat_nav.config import EATConfig
    from visualnav_transformer.eat_nav.quality import ActionStatistics
    from visualnav_transformer.eat_nav.quality_followup_training import train_overfit_resumable
    device='cuda' if torch.cuda.is_available() else 'cpu'
    config=EATConfig()
    stats=ActionStatistics(np.zeros(2,dtype=np.float32),np.ones(2,dtype=np.float32),2)
    batch={'dem':torch.zeros((2,4,2,20,20),device=device),
           'goal':torch.zeros((2,2),device=device),
           'q':torch.tensor([[.1],[.9]],device=device),
           'action':torch.zeros((2,8,2),device=device),
           'mask':torch.ones((2,8),device=device,dtype=torch.bool)}
    batch['dem'][:,:,1]=1
    batch['action'][0,:,0]=.01;batch['action'][1,:,0]=.1
    settings=GateSettings(overfit_batch_size=2,overfit_loss_ceiling=0.000001)
    full=train_overfit_resumable(config,stats,batch,seed=11,settings=settings,
                                 output=tmp_path/'full',fingerprint='same',max_epochs=2,checkpoint_interval=1)
    train_overfit_resumable(config,stats,batch,seed=11,settings=settings,
                            output=tmp_path/'resumed',fingerprint='same',max_epochs=1,checkpoint_interval=1)
    resumed=train_overfit_resumable(config,stats,batch,seed=11,settings=settings,
                                   output=tmp_path/'resumed',fingerprint='same',max_epochs=2,checkpoint_interval=1)
    assert full['final_fixed_bank_loss']==pytest.approx(resumed['final_fixed_bank_loss'],abs=1e-6)
    with pytest.raises(ValueError,match='fingerprint'):
        train_overfit_resumable(config,stats,batch,seed=11,settings=settings,
                               output=tmp_path/'resumed',fingerprint='other',max_epochs=2,checkpoint_interval=1)
