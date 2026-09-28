import random
import numpy as np
import torch

from visualnav_transformer.eat_nav.phase_c_runner import capture_rng, restore_rng, gate_c0_status


def test_rng_roundtrip_covers_python_numpy_torch_and_sampler():
    sampler = torch.Generator().manual_seed(21)
    random.seed(1); np.random.seed(2); torch.manual_seed(3)
    state = capture_rng(sampler)
    expected = (random.random(), np.random.random(), torch.rand(1).item(), torch.rand(1, generator=sampler).item())
    restore_rng(state, sampler)
    actual = (random.random(), np.random.random(), torch.rand(1).item(), torch.rand(1, generator=sampler).item())
    assert actual == expected


def test_c0_cap_while_best_still_improving_is_inconclusive():
    assert gate_c0_status([1.0, 0.9, 0.8], 3, 3) == 'inconclusive'
    assert gate_c0_status([1.0, 0.9, 0.91], 3, 3) == 'inconclusive'
    assert gate_c0_status([1.0, 0.9, 0.91], 3, 100) == 'pending'


def test_sampler_order_resumes_at_epoch_boundary():
    from torch.utils.data import DataLoader
    sampler = torch.Generator().manual_seed(123)
    loader = DataLoader(list(range(40)), batch_size=7, shuffle=True, generator=sampler)
    list(loader)
    state = capture_rng(sampler)
    expected = [batch.tolist() for batch in loader]
    resumed_sampler = torch.Generator().manual_seed(999)
    restore_rng(state, resumed_sampler)
    resumed = DataLoader(list(range(40)), batch_size=7, shuffle=True, generator=resumed_sampler)
    assert [batch.tolist() for batch in resumed] == expected


def test_best_checkpoint_is_weights_only_compatible(tmp_path):
    from visualnav_transformer.eat_nav.phase_c_runner import best_checkpoint_payload
    payload = {'metadata': {'schema_version': 1}, 'model_state_dict': {'w': torch.ones(1)},
               'ema_state_dict': {'w': torch.ones(1)}, 'rng_state': {'numpy': np.random.get_state()},
               'sampler_state': torch.Generator().get_state(), 'history': [1],
               'software_provenance': {'torch': torch.__version__}}
    best = best_checkpoint_payload(payload)
    path = tmp_path / 'best.pt'
    torch.save(best, path)
    loaded = torch.load(path, weights_only=True)
    assert 'rng_state' not in loaded and 'sampler_state' not in loaded
    assert torch.equal(loaded['ema_state_dict']['w'], torch.ones(1))


def test_saved_window_id_gzip_is_byte_deterministic(tmp_path):
    import time
    from visualnav_transformer.eat_nav.phase_c_runner import save_identities
    class Data:
        windows = [{'episode_key': (0,1,2), 'step': 4, 'terrain_seed': 42, 'terrain_type': 'rough'}]
    class View:
        dataset = Data()
        indices = [0]
        identities = [((0,1,2),4)]
    first = save_identities(tmp_path/'first.jsonl.gz', View(), View())
    time.sleep(1.1)
    second = save_identities(tmp_path/'second.jsonl.gz', View(), View())
    assert first == second
