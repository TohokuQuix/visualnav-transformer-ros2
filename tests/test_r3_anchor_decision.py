from visualnav_transformer.eat_nav.r3_anchor_decision import decide_pilot


CLASSES = ("transition", "differential", "graded", "flat-only")


def rows():
    return [
        {"virtual_class": cls, "classification": "settled_anchor_eligible",
         "settled_class": cls, "dem_valid_fraction": .8,
         "known_onset": cls != "flat-only"}
        for cls in CLASSES for _ in range(2)
    ]


def test_pass_requires_one_matching_eligible_pose_per_class():
    assert decide_pilot(rows(), protocol_valid=True) == "r3_candidate_free_pilot_pass_all_four_classes"
    items = rows()
    items[0]["classification"] = "anchor_ineligible_numeric"
    assert decide_pilot(items, protocol_valid=True) == "r3_candidate_free_pilot_pass_all_four_classes"


def test_mismatch_and_physical_failure_are_distinct():
    items = rows()
    for item in items[:2]:
        item["settled_class"] = "graded"
    assert decide_pilot(items, protocol_valid=True) == "r3_candidate_free_pilot_class_mismatch_review_required"
    items = rows()
    for item in items[:5]:
        item["classification"] = "anchor_ineligible_physical_termination"
        item["settled_class"] = None
    assert decide_pilot(items, protocol_valid=True) == "r3_candidate_free_pilot_physical_feasibility_fail"


def test_protocol_invalid_stops_and_flat_control_has_no_onset():
    assert decide_pilot(rows(), protocol_valid=False) == "r3_candidate_free_pilot_measurement_invalid"
    items = rows()
    for item in items[:2]:
        item["known_onset"] = False
    assert decide_pilot(items, protocol_valid=True) == "r3_candidate_free_pilot_partial_class_support"
    assert all(item["known_onset"] is False for item in items[-2:])
