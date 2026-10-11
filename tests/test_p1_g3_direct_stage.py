from pathlib import Path


def test_direct_stage_declares_new_protocol_and_sources():
    text = (Path(__file__).resolve().parents[1] / "scripts" /
            "eat_nav_p1_g3_direct_control_stage.py").read_text()
    assert "eatnav_p1_longhorizon_g3_direct_control_v1" in text
    assert "eat_nav_p1_g3_direct_control_sim.py" in text
    assert "eat_nav_p1_g3_direct_control_host.py" in text
    assert "direct_duplicate_only" in text
