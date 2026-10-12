"""Fail-closed offline continuation of the frozen three-period crossover."""

import json
from pathlib import Path

import pytest

import eat_nav_p1_g3_rosfree_crossover_prepare as prepare
import eat_nav_p1_g3_rosfree_crossover_authorize as authorize
import eat_nav_p1_g3_rosfree_crossover_host as host
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_path_contract import (
    validate_host_bind,
)


REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('period, roles', [
    (1, ['left_20', 'right_20', 'direct', 'direct']),
    (2, ['right_20', 'left_20', 'right_20', 'left_20']),
])
def test_preparation_stages_only_requested_frozen_period(tmp_path, period, roles):
    base = json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = prepare.build_assignment(base)
    manifest = prepare.build_manifest(REPO, prepare.REFERENCE, 'a'*64, assignment,
                                      period_index=period)
    root = tmp_path/'preparation'
    prepare.stage_one(REPO, prepare.REFERENCE, root, assignment, manifest,
                      period_index=period)
    batch = root/f'batch-first-differential-0-period-{period}'
    assert batch.is_dir()
    assert len(list(root.glob('batch-*'))) == 1
    assert manifest['runtime_pilot_process_id'] == batch.name
    assert [x['path_role'] for x in assignment['processes'][period]['trials']] == roles
    assert json.loads((root/'stage-complete.json').read_text())['process_id'] == batch.name


def test_host_checks_period_one_assignment_and_authorization(tmp_path):
    base = json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = prepare.build_assignment(base)
    manifest = prepare.build_manifest(REPO, prepare.REFERENCE, 'a'*64, assignment,
                                      period_index=1)
    root = tmp_path/'preparation'
    prepare.stage_one(REPO, prepare.REFERENCE, root, assignment, manifest,
                      period_index=1)
    batch = root/'batch-first-differential-0-period-1'
    contract = json.loads((batch/'input/g3-contract.json').read_text())
    contract.update(runtime_authorized=True, authorized_period_index=1,
                    authorized_process_ids=[batch.name], periods_1_2_authorized=True)
    (batch/'input/g3-contract.json').write_text(json.dumps(contract))
    ledger = json.loads((batch/'input-sha-ledger.json').read_text())
    ledger['input/g3-contract.json'] = prepare.sha(batch/'input/g3-contract.json')
    (batch/'input-sha-ledger.json').write_text(json.dumps(ledger))
    class Args:
        anchor_id='first-differential'
        batch_index=0
        execution_index=1
    assert [r['path_role'] for r in host.validate_staged(batch, Args())[2]] == \
           ['left_20', 'right_20', 'direct', 'direct']
    Args.execution_index=2
    with pytest.raises(ValueError):
        host.validate_staged(batch, Args())


def test_path_contract_only_accepts_one_authorized_period(tmp_path):
    bind=tmp_path/'bind'; bind.mkdir()
    root=bind/'.eatnav-p1-g3-rosfree-crossover-v1-attempt-test'
    batch=root/'batch-first-differential-0-period-1'
    batch.mkdir(parents=True)
    manifest={'schema':'eatnav_p1_g3_rosfree_crossover_v1',
              'runtime_authorized':True,'automatic_continuation':False,
              'authorized_period_index':1,'authorized_process_ids':[batch.name],
              'attempt_id':root.name,'host_bind_source':str(bind),
              'container_bind_destination':str(tmp_path/'container'),
              'attempt_host_root':str(root),
              'attempt_container_root':str(tmp_path/'container'/root.name)}
    inspect={'Mounts':[{'Type':'bind','RW':True,'Source':str(bind),
                       'Destination':str(tmp_path/'container')}]}
    assert validate_host_bind(inspect,manifest,batch,
        tmp_path/'container'/root.name/batch.name,
        expected_bind=(str(bind),str(tmp_path/'container')))['bind_mapping_verified']
    manifest['authorized_process_ids']=['batch-first-differential-0-period-2']
    with pytest.raises(ValueError):
        validate_host_bind(inspect,manifest,batch,
            tmp_path/'container'/root.name/batch.name,
            expected_bind=(str(bind),str(tmp_path/'container')))


def test_period_one_requires_verified_sealed_period_zero(tmp_path):
    with pytest.raises(ValueError, match='period-0'):
        authorize.verify_period_reference(tmp_path/'missing',
            expected_period=0, assignment_sha='a'*64)
