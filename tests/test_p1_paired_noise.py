from types import SimpleNamespace

import torch

from visualnav_transformer.eat_nav.p1_paired_noise import EXPECTED, PairedPolicyNoise, key_seed


def original(data, cfg):
    return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min


def manager():
    names = list(EXPECTED)
    cfgs = []
    for name in names:
        noise = SimpleNamespace(operation='add', n_min=-0.2, n_max=0.2, func=original)
        cfgs.append(SimpleNamespace(noise=noise))
    return SimpleNamespace(_group_obs_term_names={'policy': names},
                           _group_obs_term_cfgs={'policy': cfgs})


def test_all_conditions_get_identical_sample_and_clean_state_survives():
    mgr = manager()
    pair = PairedPolicyNoise(mgr, 'first-differential', 123)
    try:
        pair.set_step(0)
        baseline = torch.arange(4, dtype=torch.float32).reshape(4, 1)
        outputs = [cfg.noise.func(baseline, cfg.noise) for cfg in mgr._group_obs_term_cfgs['policy']]
        samples = pair.verify_step()
        for name, output in zip(EXPECTED, outputs):
            assert torch.allclose(output - baseline, samples[name].expand_as(output), atol=2e-7)
            assert output.shape == baseline.shape
    finally:
        pair.close()
    assert all(cfg.noise.func is original for cfg in mgr._group_obs_term_cfgs['policy'])


def test_keyed_samples_repeat_with_same_block_but_change_across_blocks():
    def sample(seed):
        mgr = manager()
        pair = PairedPolicyNoise(mgr, 'first-differential', seed)
        try:
            pair.set_step(11)
            for cfg in mgr._group_obs_term_cfgs['policy']:
                cfg.noise.func(torch.zeros(4, 100), cfg.noise)
            return pair.verify_step()
        finally:
            pair.close()
    a, b, c = sample(123), sample(123), sample(124)
    assert all(torch.equal(a[name], b[name]) for name in EXPECTED)
    assert all(not torch.equal(a[name], c[name]) for name in EXPECTED)


def test_original_distribution_and_replay_equivalence():
    mgr = manager()
    pair = PairedPolicyNoise(mgr, 'second-graded', 456)
    try:
        pair.set_step(7)
        cfg = mgr._group_obs_term_cfgs['policy'][0].noise
        output = cfg.func(torch.ones(4, 10000), cfg)
        noise = output[1] - 1
        assert torch.all(noise >= -0.2) and torch.all(noise <= 0.2)
        assert abs(float(noise.mean())) < 0.01
        assert abs(float(noise.std()) - 0.4 / 12**0.5) < 0.01
        assert torch.equal(output[1], output[0])
        assert torch.equal(output[2], output[0])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(key_seed('second-graded', 456, 7, EXPECTED[0]))
            original_b = original(torch.ones(1, 10000), cfg)
        assert torch.allclose(output[:1], original_b, atol=2e-7)
    finally:
        pair.close()
