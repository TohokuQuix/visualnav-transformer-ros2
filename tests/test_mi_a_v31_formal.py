import pytest

from scripts.eat_nav_mi_a_v31_run import load_frozen
from scripts.eat_nav_mi_a_v31_formal_run import formal_rows, baseline_for, validate_new_root


def test_formal_schedule_is_exactly_all_frozen_rows():
    manifest = load_frozen()
    rows = formal_rows(manifest)
    assert rows == manifest['assignment']
    assert len(rows) == 192
    assert [row['index'] for row in rows] == list(range(192))
    assert len({row['terrain_seed'] for row in rows}) == 12
    assert sum(row['stage'] == 'prepath' for row in rows) == 48
    assert sum(row['stage'] == 'candidate' for row in rows) == 144
    for row in rows:
        base = baseline_for(rows, row)
        assert base['stage'] == 'prepath' and base['repeat_index'] == 0
        assert base['state_group_id'] == row['state_group_id']
        assert base['flipper_initialization_mode'] == row['flipper_initialization_mode']
        assert base['index'] <= row['index']


def test_formal_root_must_be_fresh_persistent_child(tmp_path, monkeypatch):
    from scripts import eat_nav_mi_a_v31_formal_run as runner
    monkeypatch.setattr(runner, 'EVIDENCE_PARENT', tmp_path)
    allowed = tmp_path/'formal-new'
    assert validate_new_root(allowed) == allowed
    allowed.mkdir()
    with pytest.raises(FileExistsError):
        validate_new_root(allowed)
    with pytest.raises(ValueError):
        validate_new_root(tmp_path/'sub'/'formal-new')
    with pytest.raises(ValueError):
        validate_new_root(tmp_path/'..'/'escape')
    (tmp_path/'link').symlink_to(allowed, target_is_directory=True)
    with pytest.raises(ValueError):
        validate_new_root(tmp_path/'link')
