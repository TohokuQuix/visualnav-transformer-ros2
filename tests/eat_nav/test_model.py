from dataclasses import replace

import torch

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.model import EATNav


torch.set_num_threads(1)


def inputs(batch=2,frames=4):
    return torch.rand(batch,frames,2,20,20),torch.tensor([[.3,-.1]]*batch),torch.full((batch,1),.5)


def test_shared_cnn_and_token_counts():
    for context,expected in ((3,6),(5,8)):
        model=EATNav(replace(EATConfig(),context_size=context))
        dem,goal,q=inputs(frames=context+1)
        dem[:,1]=dem[:,0]
        frame_tokens=model.encode_dem_frames(dem)
        assert frame_tokens.shape==(2,context+1,256)
        torch.testing.assert_close(frame_tokens[:,0],frame_tokens[:,1])
        assert model.encode_tokens(dem,goal,q).shape==(2,expected,256)
        assert model.encode(dem,goal,q).shape==(2,256)
        baseline=EATNav(replace(EATConfig(),context_size=context,quality_enabled=False))
        assert baseline.encode_tokens(dem,goal,q).shape==(2,expected-1,256)


def test_masked_noise_loss_and_variable_output_length():
    for length in (8,12):
        model=EATNav(replace(EATConfig(),len_traj_pred=length))
        dem,goal,q=inputs()
        context=model.encode(dem,goal,q)
        action=torch.rand(2,length,2)
        mask=torch.ones(2,length,dtype=torch.bool)
        mask[:,-2:]=False
        noise=torch.rand_like(action)
        timestep=torch.tensor([3,7])
        loss=model.noise_loss(action,mask,context,noise=noise,timesteps=timestep)
        changed=action.clone(); changed[:,-2:]=1000
        torch.testing.assert_close(loss,model.noise_loss(changed,mask,context,noise=noise,timesteps=timestep))
        assert torch.isfinite(loss)
        model.eval()
        xy=model.sample(dem,goal,q,seed=7)
        assert xy.shape==(2,length,2)
        torch.testing.assert_close(xy,model.sample(dem,goal,q,seed=7))


def test_quality_disabled_model_ignores_requested_q():
    model=EATNav(replace(EATConfig(),quality_enabled=False))
    model.eval()
    dem,goal,q=inputs()
    low=model.sample(dem,goal,torch.full_like(q,.1),seed=3)
    high=model.sample(dem,goal,torch.full_like(q,.9),seed=3)
    torch.testing.assert_close(low,high)
