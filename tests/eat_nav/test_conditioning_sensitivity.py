import numpy as np
import torch

from visualnav_transformer.eat_nav.evaluation import diagnose_conditioning_sensitivity


class ToyPolicy:
    def __init__(self,uses_q):
        self.uses_q=uses_q
        self.training=True
        self.calls=[]
    def eval(self):
        self.training=False
        return self
    def train(self,value=True):
        self.training=value
    def sample(self,dem,goal,quality,seed):
        assert not self.training
        self.calls.append((seed,tuple(float(x) for x in quality[:,0])))
        step=torch.arange(1,9,dtype=torch.float32).view(1,8,1)
        q=quality.view(-1,1,1) if self.uses_q else torch.zeros_like(quality).view(-1,1,1)
        noise=(seed-1)*.005
        return torch.cat((step*(.05+.08*q)+noise,step**2*.001*q),dim=2)


def windows(n):
    return [{'dem':np.ones((4,2,20,20),dtype=np.float32),
             'goal':np.array([.4,0],dtype=np.float32),
             'action':np.full((8,2),.04 if i%2 else .08,dtype=np.float32),
             'quality':.1 if i%2 else .9} for i in range(n)]


def test_paired_q_sweep_measures_effect_against_same_q_sampling():
    q_values=[.1,.3,.5,.7,.9,.95]
    model=ToyPolicy(True)
    report=diagnose_conditioning_sensitivity(model,windows(50),q_values,[0,1,2])
    assert report['status']=='pass'
    assert report['input_count']==50
    assert report['q_values']==q_values
    assert report['paired_trajectory_difference_m']>report['sampling_trajectory_difference_m']
    assert report['effect_size_ratio']>1
    assert report['endpoint_difference_m']>0
    assert len(report['pointwise_difference_m'])==8
    assert report['high_q_forward_progress_delta_m']>0
    assert report['data_high_quality_forward_progress_delta_m']>0
    assert report['direction_agreement_forward_progress']
    assert model.training
    assert [c[0] for c in model.calls]==[0]*6+[1]*6+[2]*6


def test_ignored_quality_fails_gate_and_small_sample_is_inconclusive():
    report=diagnose_conditioning_sensitivity(ToyPolicy(False),windows(50),[.1,.5,.9,.95],[0,1,2])
    assert report['status']=='fail'
    assert report['paired_trajectory_difference_m']<1e-6
    assert diagnose_conditioning_sensitivity(ToyPolicy(True),windows(49),[.1,.5,.9,.95],[0,1,2])['status']=='inconclusive'
