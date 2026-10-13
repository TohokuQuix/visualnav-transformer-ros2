from pathlib import Path

import pytest

from scripts.eat_nav_mi_a_v31_run import load_frozen, row_name
from scripts.eat_nav_mi_a_v31_formal_resume_v2 import source_for_row, verify_identical_measurements


def test_resume_uses_validated_prefix_and_designated_new_attempt_only(tmp_path):
    rows = load_frozen()['assignment']
    original, stability = tmp_path/'old', tmp_path/'stability'
    for row in rows[:43]:
        assert source_for_row(row, original, stability) == original/row_name(row)
    assert source_for_row(rows[43], original, stability) == stability/'attempt-a'
    with pytest.raises(ValueError, match='new execution'):
        source_for_row(rows[44], original, stability)


def test_repeatability_check_catches_changed_trace(tmp_path):
    a, b = tmp_path/'a'/'sim-output', tmp_path/'b'/'sim-output'
    for out in (a, b):
        out.mkdir(parents=True)
        for name in ['prepath.npz', 'motion-labels.json',
                     *(f'trace-env-{i}.jsonl' for i in range(4))]:
            (out/name).write_bytes(name.encode())
    result = verify_identical_measurements(a.parent, b.parent)
    assert len(result) == 6
    (b/'trace-env-2.jsonl').write_bytes(b'different')
    with pytest.raises(ValueError, match='trace-env-2'):
        verify_identical_measurements(a.parent, b.parent)
