from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_batch_golden import GoldenIntegrityError, verify_backup


BACKUP = Path('/home/kojima/eatnav-backups/2026-10-09-p1-radial-28-analysis')


def test_p1_backup_has_28_sealed_unique_runs():
    audit = verify_backup(BACKUP)
    assert audit['run_count'] == 28
    assert audit['verified_file_count'] == 2919
    assert audit['trace_steps'] == 28 * 120
    assert audit['old_tenth_excluded'] is True
    assert len(audit['run_ids']) == len(set(audit['run_ids']))


def test_missing_seal_rejected(tmp_path):
    with pytest.raises(GoldenIntegrityError, match='backup-seal'):
        verify_backup(tmp_path)
