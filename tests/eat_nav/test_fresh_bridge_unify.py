"""Fresh-protocol analysis view excludes diagnostics and never rewrites frozen runs."""
import pathlib
import pytest

from eat_nav_fresh_bridge_unify import assemble, link_once


def test_link_once_preserves_source_and_rejects_retarget(tmp_path):
    source=tmp_path/'source.json';source.write_text('{"marker": 1}\n')
    link=tmp_path/'view'/'source.json'
    link_once(link,source)
    link_once(link,source)
    assert link.is_symlink() and link.read_text()==source.read_text()
    other=tmp_path/'other.json';other.write_text('{}\n')
    with pytest.raises(ValueError,match='link target changed'):
        link_once(link,other)
    assert source.read_text()=='{"marker": 1}\n'


def test_analysis_view_blocked_until_119_complete(monkeypatch,tmp_path):
    import eat_nav_fresh_bridge_unify as u
    root=tmp_path/'fresh';root.mkdir()
    (root/'status.json').write_text('{"stage":"R8_RUNNING","completed":101}')
    monkeypatch.setattr(u,'ROOT',root)
    with pytest.raises(RuntimeError,match='119 raw runs not complete'):
        assemble()
    assert not (root/'unified').exists()
