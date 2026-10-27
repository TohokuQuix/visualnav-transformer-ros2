"""Batch02 preserves historical bytes and consumes an immutable preparation."""
import copy
import importlib
import json
from pathlib import Path
import shutil

import pytest

from test_additional_blind_verification import additional_fixture, source
from visualnav_transformer.eat_nav.fall_stall_audit import read, seal_derived, sha

FREEZE = Path('docs/eat-nav/fall-stall-additional-batch02-freeze-20261021/artifacts')


def module(name):
    return importlib.import_module('visualnav_transformer.eat_nav.fall_stall_batch02_' + name)


def runtime(contract):
    result = dict(container_image='image', container_id='container', python='3.11', numpy='1.26.4',
                  torch='2.7', isaac_version='5.1.0', cuda=True, package_versions={'isaaclab': '2.3'},
                  resource_preflight={'gpu_uuid': 'gpu', 'gpu_capacity_mib': 16376,
                                      'gpu_device_total_mib': 17, 'host_mem_available_kib': 8000000,
                                      'host_batch_storage_bytes': 0}, onix_source_parity='pass')
    sources = {p: sha(contract['repo'] / p) for p in module('runner').ADDED_SOURCES}
    result['stage_parity'] = {'pass': True, 'stage': '/tmp/eatnav-batch02-preflight-fixture',
                              'files_sha256': module('runtime').expected_stage_hashes(contract, sources)}
    return result


@pytest.fixture
def fixture(additional_fixture, tmp_path):
    raw, original = additional_fixture
    c = module('contract').load_freeze(FREEZE)
    c['sources']['pilot_critical_sha256'] = original['sources']['pilot_critical_sha256']
    c['rows'] = original['rows'][:2]
    c['manifest']['protocol_sha256'] = original['manifest']['protocol_sha256']
    c['manifest']['output_root'] = str(tmp_path / 'collection')
    return raw, c, Path(c['manifest']['output_root'])


def test_batch02_identity_and_historical_loader_remain_separate(tmp_path):
    c = module('contract').load_freeze(FREEZE)
    assert len(c['rows']) == 120 and c['rows'][0]['base_group'].startswith('batch02-')
    assert c['manifest']['batch_id'].endswith('v0.2-batch02')
    with pytest.raises(ValueError):
        module('contract').load_freeze('docs/eat-nav/fall-stall-additional-collection-freeze-20261020/artifacts')
    copied = tmp_path / 'freeze'
    shutil.copytree(FREEZE, copied)
    (copied / 'assignment.json').write_text('{}')
    for name in ('sha-ledger.json', 'derived-seal.json'):
        (copied / name).unlink()
    seal_derived(copied)
    with pytest.raises(ValueError):
        module('contract').load_freeze(copied)


def test_prepared_snapshot_is_sealed_and_source_changes_reject_before_creation(fixture, tmp_path):
    _, c, root = fixture
    bound = module('runner').create_binding(c, runtime(c))
    prepared = tmp_path / 'prepared'
    module('runner').save_preparation(c, prepared, bound)
    assert not root.exists()
    assert module('runner').load_preparation(c, prepared) == bound
    snapshot = prepared / 'sources/scripts/eat_nav_additional_batch02_collection.py'
    snapshot.write_text('changed')
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').load_preparation(c, prepared)
    changed = copy.deepcopy(bound)
    changed['source_sha256']['scripts/eat_nav_additional_batch02_collection.py'] = '0' * 64
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').initialize_collection(c, root, changed)
    assert not root.exists()


def test_dispatch_is_blind_and_resume_reuses_exact_binding(fixture, capsys):
    raw, c, root = fixture
    calls = []
    def execute(target, **kwargs):
        calls.append(kwargs)
        print('PRIVATE_tip_over_89_progress_12.34')
        shutil.copytree(raw / target.name, target)
    bound = module('runner').create_binding(c, runtime(c))
    result = module('runner').run_collection(c, root, bound, executor=execute)
    assert result['completed_processes'] == 2 and result['reserved_test'] == 'unopened'
    assert len(calls) == 2 and all(k['control_steps'] == 200 and k['replan_every_steps'] == 40 for k in calls)
    assert 'PRIVATE' not in capsys.readouterr().out
    module('runner').run_collection(c, root, bound, executor=execute, resume=True)
    assert len(calls) == 2
    changed = copy.deepcopy(bound)
    changed['runtime']['container_id'] = 'changed'
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').run_collection(c, root, changed, executor=execute, resume=True)


def test_failed_attempt_is_retained_and_not_retried(fixture, capsys):
    _, c, root = fixture
    calls = []
    def execute(target, **kwargs):
        calls.append(target)
        target.mkdir(parents=True)
        raise RuntimeError('PRIVATE_tip_over_89_progress_12.34')
    bound = module('runner').create_binding(c, runtime(c))
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').run_collection(c, root, bound, executor=execute)
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').run_collection(c, root, bound, executor=execute, resume=True)
    assert len(calls) == 1 and calls[0].exists()
    assert 'PRIVATE' not in capsys.readouterr().out + (root / 'status.json').read_text()


@pytest.mark.parametrize('role', [('reserved_test',), ('train', 'reserved_test')])
def test_reserved_export_denied_before_any_evidence_read(tmp_path, role):
    with pytest.raises(module('runner').BlindVerificationError) as error:
        module('targets').build_development_dataset(tmp_path / 'missing', {}, tmp_path / 'out', roles=role)
    assert error.value.code == 'RESERVED_TEST_DENIED' and not (tmp_path / 'out').exists()


@pytest.mark.parametrize('key', ['container_image', 'container_id', 'python', 'numpy', 'torch',
                                'isaac_version', 'cuda', 'package_versions', 'gpu_uuid'])
def test_runtime_drift_rejected(fixture, monkeypatch, key):
    _, c, _ = fixture
    saved = runtime(c)
    bound = module('runner').create_binding(c, saved)
    changed = copy.deepcopy(saved)
    if key == 'gpu_uuid':
        changed['resource_preflight'][key] = 'changed'
    else:
        changed[key] = 'changed'
    monkeypatch.setattr(module('runtime'), 'probe_runtime', lambda c: changed)
    monkeypatch.setattr(module('runtime'), 'check_staged_runtime', lambda *a: None)
    with pytest.raises(module('runner').BlindVerificationError):
        module('runtime').recheck_runtime(c, bound)


def test_stage_mismatch_blocks_preparation(fixture, monkeypatch):
    _, c, _ = fixture
    monkeypatch.setattr(module('runtime'), 'container_hashes', lambda paths: {})
    sources = {p: sha(c['repo'] / p) for p in module('runner').ADDED_SOURCES}
    expected = module('runtime').expected_stage_hashes(c, sources)
    files = sorted(key for key in expected if key.startswith('src/visualnav_transformer/'))
    monkeypatch.setattr(module('runtime'), 'native', lambda *args: json.dumps(
        {'source_files': files, 'namespace_initializers': [], 'bytecode_files': []}))
    with pytest.raises(ValueError, match='staged'):
        module('runtime').stage_preflight(c, sources)


@pytest.mark.parametrize('mutation', ['cuda', 'source_membership', 'stage_files'])
def test_invalid_runtime_proof_refused(fixture, mutation):
    _, c, root = fixture
    bound = module('runner').create_binding(c, runtime(c))
    if mutation == 'cuda':
        bound['runtime']['cuda'] = False
    elif mutation == 'source_membership':
        bound['source_sha256'].pop('scripts/eat_nav_additional_batch02_collection.py')
    else:
        bound['runtime']['stage_parity']['files_sha256'] = {}
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').initialize_collection(c, root, bound)
    assert not root.exists()


@pytest.mark.parametrize('addition', ['src/visualnav_transformer/eat_nav/__init__.py',
                                    'src/visualnav_transformer/eat_nav.py'])
def test_added_initializer_rejects_saved_preparation_before_root_creation(fixture, tmp_path, addition):
    _, c, root = fixture
    original = c['repo']
    isolated = tmp_path / 'isolated-repo'
    paths = set(c['sources']['repo_files_sha256']) | set(module('runner').ADDED_SOURCES)
    for relative in paths:
        target = isolated / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / relative, target)
    c['repo'] = isolated
    prepared = tmp_path / 'prepared'
    bound = module('runner').create_binding(c, runtime(c))
    module('runner').save_preparation(c, prepared, bound)
    (isolated / addition).write_text('# added after binding\n')
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').load_preparation(c, prepared)
    assert not root.exists()


def test_unexpected_staged_module_is_rejected(fixture, monkeypatch):
    _, c, _ = fixture
    sources = {p: sha(c['repo'] / p) for p in module('runner').ADDED_SOURCES}
    expected = module('runtime').expected_stage_hashes(c, sources)
    files = sorted(key for key in expected if key.startswith('src/visualnav_transformer/'))
    files.append('src/visualnav_transformer/eat_nav/unbound.py')
    monkeypatch.setattr(module('runtime'), 'native', lambda *a: json.dumps(
        {'source_files': files, 'namespace_initializers': [], 'bytecode_files': []}))
    with pytest.raises(ValueError, match='staged'):
        module('runtime').check_stage_membership('/tmp/fixture', expected)


def test_development_export_uses_batch02_binding_and_initial_inputs(additional_fixture, tmp_path):
    import numpy as np
    from visualnav_transformer.eat_nav.bidirectional_bank import candidate
    from test_fall_stall_audit import put, reseal_attempt
    raw, original = additional_fixture
    c = module('contract').load_freeze(FREEZE)
    c['sources']['pilot_critical_sha256'] = original['sources']['pilot_critical_sha256']
    c['rows'] = original['rows']
    c['manifest']['protocol_sha256'] = original['manifest']['protocol_sha256']
    c['manifest']['output_root'] = str(tmp_path / 'development-collection')
    for row in c['rows']:
        row['role'] = 'train'
        if row['stage'] == 'candidate':
            attempt = raw / row['output_relative']
            sim = attempt / 'sim-output'
            path = candidate(row['direction'], row['angle']).waypoints.tolist()
            info = read(sim / 'input-manifest.json')
            info['path_body_xy_m'] = path
            put(sim / 'input-manifest.json', info)
            activation = read(sim / 'path-activation.json')
            activation['paths_tile_xy_m'] = [path] * 4
            put(sim / 'path-activation.json', activation)
            reseal_attempt(attempt)
    root = Path(c['manifest']['output_root'])
    bound = module('runner').create_binding(c, runtime(c))
    def execute(target, **kwargs):
        shutil.copytree(raw / target.name, target)
    module('runner').run_collection(c, root, bound, executor=execute)
    out = tmp_path / 'dataset'
    result = module('targets').build_development_dataset(root, c, out, roles=('train',))
    assert result == {'state_groups': 4, 'candidate_outcomes': 16, 'reserved_test_included': False}
    with np.load(out / 'samples.npz', allow_pickle=False) as arrays:
        assert arrays['dem'].shape == (4, 4, 2, 20, 20)
        assert arrays['event_mask'][:, :, 3, :].all()
    assert read(out / 'split.json')['roles'] == {'train': [0, 1, 2, 3]}


def test_cli_refuses_unpinned_collection_before_runtime_probe(monkeypatch, capsys):
    from scripts import eat_nav_additional_batch02_collection as cli
    monkeypatch.setattr(cli, 'run_collection', lambda *a, **k: pytest.fail('must not collect'))
    monkeypatch.setattr(cli, 'probe_runtime', lambda *a: pytest.fail('must not probe'))
    assert cli.main(['--collect']) == 1
    assert json.loads(capsys.readouterr().out) == {'pass': False, 'error_code': 'BINDING_MISMATCH'}


def test_expected_binding_sha_rejects_even_resealed_preparation(fixture, tmp_path):
    _, c, _ = fixture
    prepared = tmp_path / 'prepared'
    bound = module('runner').create_binding(c, runtime(c))
    module('runner').save_preparation(c, prepared, bound)
    pinned = sha(prepared / 'execution-binding.json')
    changed = read(prepared / 'execution-binding.json')
    changed['created_unix_ns'] += 1
    (prepared / 'execution-binding.json').write_text(json.dumps(changed))
    meta = read(prepared / 'preparation.json')
    meta['execution_binding_sha256'] = sha(prepared / 'execution-binding.json')
    (prepared / 'preparation.json').write_text(json.dumps(meta))
    for name in ('sha-ledger.json', 'derived-seal.json'):
        (prepared / name).unlink()
    seal_derived(prepared)
    with pytest.raises(module('runner').BlindVerificationError):
        module('runner').load_preparation(c, prepared, expected_binding_sha256=pinned)
