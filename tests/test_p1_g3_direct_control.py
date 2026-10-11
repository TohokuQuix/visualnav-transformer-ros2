"""Static safety contract for the direct duplicate runtime control."""

import ast
from pathlib import Path

from visualnav_transformer.eat_nav.p1_statistical_pilot import path_in_map


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_direct_control_publishes_only_two_direct_paths():
    source = (SCRIPTS / "eat_nav_p1_g3_direct_control_sim.py").read_text()
    assert "active_assigned=assigned[:2]" in source
    assert "paths=direct_paths_for_slots(active_assigned,path_lookup,initial_poses[:2])" in source
    assert "poses=initial_poses[:2]+[initial_poses[0]],paths=paths" in source
    assert "branches=3" in source
    assert "action_batch[2:]=0" in source


def test_direct_path_builder_excludes_curves_and_rejects_other_slots():
    source = (SCRIPTS / "eat_nav_p1_g3_direct_control_sim.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "direct_paths_for_slots")
    namespace = {"path_in_map": path_in_map}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<direct builder>", "exec"), namespace)
    rows = [{"slot": 0, "path_sha256": "direct"},
            {"slot": 1, "path_sha256": "direct"}]
    lookup = {"direct": {"waypoints": [[0, 0], [0.75, 0]]},
              "curve": {"waypoints": [[0, 0], [0.75, 0.2]]}}
    paths = namespace["direct_paths_for_slots"](rows, lookup, [[0, 0, 0], [0, 0, 0]])
    assert len(paths) == 3
    assert paths[0] == paths[1] == paths[2] == [[0.0, 0.0], [0.75, 0.0]]


def test_direct_control_host_validates_two_traces():
    source = (SCRIPTS / "eat_nav_p1_g3_direct_control_host.py").read_text()
    assert "--branches','3'" in source
    assert "for index in range(2):" in source
    assert "['direct_duplicate_0','direct_duplicate_1']" in source
    assert "range(4)]" not in source[source.index("def validate_g3_output"):source.index("def validate_dem_diagnostic_output")]
