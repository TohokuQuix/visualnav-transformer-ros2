import hashlib
import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_statistical_freeze import verify_ledger, freeze_manifest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ledger_rejects_changed_file(tmp_path):
    (tmp_path/'data.bin').write_bytes(b'original')
    ledger = {'file_count':1, 'files':[{'path':'data.bin','bytes':8,'sha256':sha(tmp_path/'data.bin')}]}
    (tmp_path/'ledger.json').write_text(json.dumps(ledger))
    assert verify_ledger(tmp_path, tmp_path/'ledger.json')['file_count']==1
    (tmp_path/'data.bin').write_bytes(b'changed!')
    with pytest.raises(ValueError, match='SHA mismatch'):
        verify_ledger(tmp_path, tmp_path/'ledger.json')


def test_manifest_freezes_source_config_and_assignment_without_overwrite(tmp_path):
    project=Path(__file__).parents[1]
    contract=project/'config/eat_nav_p1_batch_source_contract.json'
    config=project/'config/eat_nav_p1_statistical_v1.json'
    result=freeze_manifest(contract,config,tmp_path/'freeze')
    assert result['source_contract_sha256']==sha(contract)
    assert result['config_sha256']==sha(config)
    assert (tmp_path/'freeze/assignment.json').is_file()
    with pytest.raises(FileExistsError):
        freeze_manifest(contract,config,tmp_path/'freeze')
