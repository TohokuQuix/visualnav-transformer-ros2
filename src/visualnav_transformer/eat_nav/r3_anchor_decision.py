"""Predeclared interpretation of the eight-pose R3 settling pilot."""
from __future__ import annotations

CLASSES = ("transition", "differential", "graded", "flat-only")


def physical_supply(item: dict) -> bool:
    cls = item["virtual_class"]
    return (
        item["classification"] == "settled_anchor_eligible"
        and item.get("settled_class") == cls
        and item.get("dem_valid_fraction", 0) >= .70
        and (cls == "flat-only" or item.get("known_onset") is True)
    )


def decide_pilot(rows: list[dict], *, protocol_valid: bool) -> str:
    if not protocol_valid or len(rows) != 8:
        return "r3_candidate_free_pilot_measurement_invalid"
    if all(sum(physical_supply(x) for x in rows if x["virtual_class"] == cls) >= 1
           for cls in CLASSES):
        return "r3_candidate_free_pilot_pass_all_four_classes"
    # Strict majority is fixed before measurement; it only selects the
    # review category and never invalidates a measured termination.
    if sum(x["classification"] in (
        "anchor_ineligible_numeric", "anchor_ineligible_physical_termination") for x in rows) >= 5:
        return "r3_candidate_free_pilot_physical_feasibility_fail"
    if any(x["classification"] == "settled_anchor_eligible" and
           x.get("settled_class") not in (None, x["virtual_class"]) for x in rows):
        return "r3_candidate_free_pilot_class_mismatch_review_required"
    return "r3_candidate_free_pilot_partial_class_support"
