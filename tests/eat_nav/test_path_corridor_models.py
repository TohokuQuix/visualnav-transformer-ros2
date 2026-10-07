import torch

from visualnav_transformer.eat_nav.path_corridor_models import PathCorridorEvaluator


def test_four_models_share_six_head_contract_and_have_finite_gradients():
    inputs = {'common': torch.randn(2, 184), 'full_dem': torch.randn(2, 8, 20, 20),
              'corridor': torch.randn(2, 8, 16, 7), 'profile': torch.randn(2, 37, 16)}
    for variant in ('A1', 'A2', 'A3', 'A4'):
        model = PathCorridorEvaluator(variant, 184, 37)
        assert not any(isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                       for layer in model.modules())
        result = model(**inputs)
        assert result.shape == (2, 6)
        result.square().mean().backward()
        assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                   if parameter.grad is not None)
