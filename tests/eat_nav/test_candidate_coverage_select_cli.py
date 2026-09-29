"""The Stage 0 CLI freezes inputs before it can select or overwrite states."""

from argparse import Namespace
import json

import pytest

from scripts.eat_nav_candidate_coverage_select import atomic_once, run


def test_selection_requires_fixed_budget_before_touching_inputs(tmp_path):
    args = Namespace(approved_budget_hours=None, max_full_prefix_steps=None,
                     manifest=tmp_path / "missing.json")
    with pytest.raises(ValueError, match="budget"):
        run(args)
    assert list(tmp_path.iterdir()) == []


def test_atomic_freeze_refuses_different_existing_selection(tmp_path):
    target = tmp_path / "selection.json"
    atomic_once(target, {"selected": [1]})
    before = target.read_bytes()
    atomic_once(target, {"selected": [1]})
    with pytest.raises(FileExistsError):
        atomic_once(target, {"selected": [2]})
    assert target.read_bytes() == before
    assert json.loads(before)["selected"] == [1]
