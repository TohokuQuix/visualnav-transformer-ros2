"""Frozen crossover assignment maps actual roles to one immutable Path SHA."""

from eat_nav_p1_g3_crossover_stage import assignment_from_design
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import SEEDS
from visualnav_transformer.eat_nav.p1_path_crossover_v2 import build_crossover


def test_assignment_has_18_processes_and_each_seed_env_sees_all_three_paths():
    source = {"paths": [{"bearing_deg": deg, "path_sha256": str(deg)}
                        for deg in (0, 20, -20)]}
    frozen = assignment_from_design(build_crossover(SEEDS), source)
    assert len(frozen["processes"]) == 18
    assert frozen["process_count"] == 18 and frozen["trial_count"] == 72
    assert len(frozen["pilot_process_ids"]) == 3
    for group in SEEDS:
        for seed in SEEDS[group]:
            periods = [row for row in frozen["processes"]
                       if row["anchor_id"] == group and row["batch_seed"] == seed]
            assert [row["execution_index"] for row in periods] == [0, 1, 2]
            for env in range(4):
                roles = {row["trials"][env]["path_role"] for row in periods}
                assert roles == {"direct", "left_20", "right_20"}
                for row in periods:
                    trial = row["trials"][env]
                    assert trial["path_sha256"] == str(trial["bearing_deg"])
