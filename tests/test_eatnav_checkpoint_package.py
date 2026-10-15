"""Runtime checkpoint paths and checksums are portable across worktrees."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_package_matches_manifest_and_launchers_use_repo_paths():
    verifier = _module(ROOT/'scripts/eat_nav_verify_checkpoints.py', 'checkpoint_verifier')
    manifest = verifier.verify()
    assert len(manifest['files_sha256']) == 14
    for filename in ('eat_nav_demo_launch.py', 'eat_nav_phase3_gui_launch.py'):
        launcher = _module(ROOT/'scripts'/filename, filename.removesuffix('.py'))
        assert launcher.CHECKPOINT == ROOT/'checkpoints/eat_nav/motion14/training/full-seed123/best.pt'
        assert launcher.RL == ROOT/'checkpoints/eat_nav/rl_actor/model_10000.pt'


def test_missing_or_pointer_checkpoint_fails_closed(tmp_path):
    verifier = _module(ROOT/'scripts/eat_nav_verify_checkpoints.py', 'checkpoint_verifier_bad')
    files = json.loads((verifier.PACKAGE/'manifest.json').read_text())
    (tmp_path/'manifest.json').write_text(json.dumps(files))
    with pytest.raises(ValueError, match='SHA mismatch'):
        verifier.verify(tmp_path)
    target = tmp_path/'rl_actor/model_10000.pt'
    target.parent.mkdir()
    target.write_text('version https://git-lfs.github.com/spec/v1\n')
    with pytest.raises(ValueError, match='SHA mismatch'):
        verifier.verify(tmp_path)
