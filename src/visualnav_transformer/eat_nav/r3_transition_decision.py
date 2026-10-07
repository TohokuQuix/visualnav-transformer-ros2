"""Frozen first-seed transition-supply decision, independent of the simulator."""
from __future__ import annotations


def decide_transition(results: list[dict], frozen_count: int) -> str:
    if frozen_count < 2:
        return 'r3_transition_reserve_insufficient'
    if len(results) != frozen_count or any(x.get('classification') == 'measurement_invalid' for x in results):
        return 'r3_transition_pilot_measurement_invalid'
    if any(x.get('physical_supply') is True for x in results):
        return 'r3_transition_supply_confirmed_ready_for_second_seed_design'
    if any(x.get('classification') == 'settled_anchor_eligible' and
           x.get('settled_class') != 'transition' for x in results):
        return 'r3_transition_class_mismatch_review_required'
    return 'r3_transition_supply_not_confirmed_placement_redesign_required'
