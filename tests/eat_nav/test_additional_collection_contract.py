"""Frozen schedule consumption and source tamper rejection."""
import importlib
import json
from pathlib import Path
import shutil
import pytest
from visualnav_transformer.eat_nav.fall_stall_audit import seal_derived

FREEZE=Path('docs/eat-nav/fall-stall-additional-collection-freeze-20261020/artifacts')


def api():
    name='visualnav_transformer.eat_nav.fall_stall_collection_contract'
    assert importlib.util.find_spec(name) is not None,'additional collection contract implementation missing'
    return importlib.import_module(name)


def test_frozen_schedule_and_sources_are_verified():
    contract=api().load_freeze(FREEZE)
    assert len(contract['rows'])==120
    assert [contract['rows'][i]['role'] for i in (0,59,60,79,80,119)]==['train','train','validation','validation','reserved_test','reserved_test']
    assert contract['rows'][80]['reference_relative'] is None
    assert contract['rows'][81]['reference_relative'].endswith('prepath/sim-output')
    api().verify_sources(contract)


def test_changed_even_resealed_freeze_is_rejected(tmp_path):
    target=tmp_path/'freeze';shutil.copytree(FREEZE,target)
    p=target/'assignment.json';value=json.loads(p.read_text());value['rows'][0]['output_relative']='../escape'
    p.write_text(json.dumps(value));(target/'sha-ledger.json').unlink();(target/'derived-seal.json').unlink();seal_derived(target)
    with pytest.raises(ValueError):api().load_freeze(target)


def test_contained_path_rejects_absolute_parent_and_symlink(tmp_path):
    for name in ('../escape','/tmp/escape'):
        with pytest.raises(ValueError):api().contained_path(tmp_path,name)
    (tmp_path/'link').symlink_to(tmp_path/'elsewhere',target_is_directory=True)
    with pytest.raises(ValueError):api().contained_path(tmp_path,'link/nested')
