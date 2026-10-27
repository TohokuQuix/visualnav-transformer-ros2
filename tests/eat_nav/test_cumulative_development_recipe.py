"""Real sealed-source acceptance and leakage guards for the offline pool recipe."""
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from visualnav_transformer.eat_nav.fall_stall_audit import read, sha, seal_derived

RECIPE = Path('docs/eat-nav/additional-development-batch02-cumulative-20261021/pool_recipe.py')


def api():
    spec = importlib.util.spec_from_file_location('cumulative_development_recipe', RECIPE)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def test_pool_keeps_source_arrays_masks_roles_and_positive_seed_support(tmp_path):
    module = api()
    sources = module.default_sources()
    out = tmp_path / 'pool'
    module.merge_sources(sources, out)
    coverage, verification = module.verify_pool(out, sources)
    assert verification['state_groups'] == 128 and verification['candidate_outcomes'] == 512
    assert coverage['roles']['train']['base_seeds'] == 24
    assert coverage['roles']['validation']['base_seeds'] == 8
    for head, expected in {'tip_over': (10, 1), 'near_fall': (7, 1),
                           'linear_stall': (8, 4), 'turn_stall': (2, 1)}.items():
        assert tuple(coverage['heads'][head][role]['positive_base_seed_count']
                     for role in ('train', 'validation')) == expected
    by_id = {spec['batch_id']: spec for spec in sources}
    index = read(out / 'sample-index.json')
    with np.load(out / 'samples.npz', allow_pickle=False) as merged:
        assert (~merged['event_mask']).any()
        assert (merged['event_targets'][~merged['event_mask']] == -1).all()
        for i, sample in enumerate(index):
            spec = by_id[sample['batch_id']]
            source_index = read(spec['root'] / 'sample-index.json')
            position = sample['source_sample_index']
            assert sample['role'] == source_index[position]['role']
            with np.load(spec['root'] / 'samples.npz', allow_pickle=False) as original:
                for key in merged.files:
                    assert np.array_equal(merged[key][i], original[key][position])


def test_reserved_role_in_source_metadata_rejected_before_pool_creation(tmp_path):
    module = api()
    sources = module.default_sources()
    changed = tmp_path / 'changed'
    shutil.copytree(sources[0]['root'], changed)
    index = read(changed / 'sample-index.json')
    index[0]['role'] = 'reserved_test'
    import json
    (changed / 'sample-index.json').write_text(json.dumps(index))
    for name in ('sha-ledger.json', 'derived-seal.json'):
        (changed / name).unlink()
    seal_derived(changed)
    sources[0] = dict(sources[0], root=changed, ledger=sha(changed / 'sha-ledger.json'))
    out = tmp_path / 'pool'
    with pytest.raises(ValueError):
        module.merge_sources(sources, out)
    assert not out.exists()


def test_duplicate_batch_cannot_inflate_independent_seed_support(tmp_path):
    module = api()
    sources = module.default_sources()
    out = tmp_path / 'pool'
    with pytest.raises(ValueError):
        module.merge_sources([sources[0], sources[0]], out)
    assert not out.exists()
