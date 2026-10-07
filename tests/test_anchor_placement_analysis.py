"""Offline report semantics for previously terminated radial poses."""

from pathlib import Path
import importlib.util


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts/eat_nav_anchor_placement_analyze.py"
    spec = importlib.util.spec_from_file_location("anchor_placement_analyze", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminated_old_pose_has_no_settled_dem_class():
    analysis = _load()
    assert analysis.settled_class({"interaction": None, "classification": "anchor_ineligible_physical_termination"}) == "NA / anchor_ineligible_physical_termination"
    assert analysis.settled_class({"interaction": None, "classification": "anchor_ineligible_numeric"}) == "NA / anchor_ineligible_numeric"
    assert analysis.settled_class({"interaction": {"class": "graded"}}) == "graded"
