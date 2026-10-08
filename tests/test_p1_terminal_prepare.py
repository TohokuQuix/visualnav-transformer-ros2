import hashlib
from pathlib import Path

import pytest

from eat_nav_p1_terminal_prepare import compare_sha_trees, verify_frozen_code


def test_compare_prior_backup_rejects_missing_or_changed_file(tmp_path):
    source = tmp_path / 'source'
    backup = tmp_path / 'backup'
    source.mkdir()
    backup.mkdir()
    (source / 'one').write_text('same')
    (backup / 'one').write_text('same')
    assert compare_sha_trees(source, backup)['file_count'] == 1
    (backup / 'one').write_text('changed')
    with pytest.raises(ValueError, match='backup_SHA_mismatch'):
        compare_sha_trees(source, backup)


def test_frozen_monitor_source_rejects_postfreeze_drift(tmp_path):
    import json
    from eat_nav_p1_terminal_prepare import WORKTREE
    path = tmp_path / 'source.py'
    path.write_text('old')
    freeze = tmp_path / 't0-frozen-code.json'
    freeze.write_text(json.dumps({'files': {str(path): hashlib.sha256(b'old').hexdigest()}}))
    assert verify_frozen_code(tmp_path)['file_count'] == 1
    path.write_text('new')
    with pytest.raises(ValueError, match='frozen_code_SHA_drift'):
        verify_frozen_code(tmp_path)
