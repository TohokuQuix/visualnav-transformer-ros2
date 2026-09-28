import torch
from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.model import EATNav
from visualnav_transformer.eat_nav.quality_gate_layers import trace_network_layers


def test_layer_trace_covers_token_context_film_blocks_and_shared_noisy_input():
    model=EATNav(EATConfig()).eval()
    dem=torch.zeros(1,4,2,20,20)
    goal=torch.tensor([[0.3,0.0]])
    action=torch.zeros(1,8,2)
    result=trace_network_layers(model,dem,goal,action,seed=11)
    for key in ('quality_mlp','quality_token_input','transformer_layer_0_quality',
                'transformer_layer_3_quality','final_context','film_down_0',
                'down_0','mid_0','up_0','final_predicted_noise'):
        assert key in result['relative_sensitivity']
    assert result['noisy_input_max_difference']==0.0
    assert len(result['epsilon_delta_by_timestep'])==10
