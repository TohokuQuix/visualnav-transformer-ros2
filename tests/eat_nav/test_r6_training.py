import pytest
from test_r6_model import arrays
from eat_nav_r6_train import fit
from visualnav_transformer.eat_nav.r6_model import load_checkpoint

def test_minibatch_fit_saves_real_two_second_model_and_refuses_overwrite(tmp_path):
 a=arrays();index=[{'role':'train','episode_group':'train/a'},{'role':'train','episode_group':'train/b'},{'role':'validation','episode_group':'val/c'}]
 roles={'train':[0,1],'validation':[2]}
 r=fit(a,index,roles,tmp_path/'run',epochs=2,batch_size=1,device='cpu',seed=123)
 assert r['epochs_completed']==2 and r['device']=='cpu'
 _,p=load_checkpoint(tmp_path/'run/checkpoint.pt');assert p['contract']['event_horizon_sec']==2
 with pytest.raises(FileExistsError):fit(a,index,roles,tmp_path/'run',epochs=2,device='cpu')

def test_training_rejects_episode_leakage_before_normalization(tmp_path):
 a=arrays();index=[{'role':'train','episode_group':'same'},{'role':'train','episode_group':'b'},{'role':'validation','episode_group':'same'}]
 with pytest.raises(ValueError,match='leak'):fit(a,index,{'train':[0,1],'validation':[2]},tmp_path/'bad',epochs=1,device='cpu')
