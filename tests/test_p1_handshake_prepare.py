import json
from pathlib import Path

import pytest

from eat_nav_p1_handshake_prepare import freeze_overlay, verify_sha_entry


def test_freeze_overlay_changes_only_output_directory(tmp_path):
    original = {"rows": [{"order": i, "physical_run_id": f"run-{i}", "output_directory": "/old"} for i in range(28)], "seed": 123}
    overlay, diff = freeze_overlay(original, tmp_path)
    assert len(diff) == 28
    assert all(set(item["changed"]) == {"output_directory"} for item in diff)
    assert all(a["physical_run_id"] == b["physical_run_id"] for a, b in zip(original["rows"], overlay["rows"]))


def test_sha_entry_refuses_mismatch(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"x": 1}))
    with pytest.raises(ValueError, match="sha_mismatch"):
        verify_sha_entry(path, "0" * 64)
