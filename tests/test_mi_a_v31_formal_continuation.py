import json
from pathlib import Path

import pytest

from scripts.eat_nav_mi_a_v31_formal_continuation import classify_prelaunch_failure, link_prefix


def test_prelaunch_classifier_requires_xopen_display_and_no_sim_output(tmp_path):
    failed = tmp_path/'failed'
    failed.mkdir()
    (failed/'simulator.log').write_text('Kit startup\nXOpenDisplay\nSegmentation fault\n')
    (failed/'native-exit.json').write_text(json.dumps({'exit_code': 1}))
    assert classify_prelaunch_failure(failed) == 'kit_xopen_display_before_python_main'
    (failed/'sim-output').mkdir()
    with pytest.raises(ValueError):
        classify_prelaunch_failure(failed)


def test_prefix_is_read_only_links_and_source_bytes_unchanged(tmp_path, monkeypatch):
    from scripts import eat_nav_mi_a_v31_formal_continuation as module
    prior, target = tmp_path/'prior', tmp_path/'new'
    prior.mkdir(); target.mkdir()
    rows = [{'index': 0}, {'index': 1}]
    monkeypatch.setattr(module, 'row_name', lambda row: f"{row['index']:03d}")
    monkeypatch.setattr(module, 'verify_attempt', lambda path, row: None)
    for row in rows:
        attempt = prior/f"{row['index']:03d}"
        attempt.mkdir()
        (attempt/'outer-seal.json').write_text(f"seal-{row['index']}")
    seals = link_prefix(prior, target, rows)
    assert len(seals) == 2
    assert all((target/f"{i:03d}").is_symlink() for i in range(2))
    assert [(prior/f"{i:03d}"/'outer-seal.json').read_text() for i in range(2)] == ['seal-0', 'seal-1']
    with pytest.raises(FileExistsError):
        link_prefix(prior, target, rows)
