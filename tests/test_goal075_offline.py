import math
import os
from pathlib import Path

import numpy as np
import pytest

from visualnav_transformer.eat_nav.goal075_offline import (
    goal075, contract075, initial_forward_tangent_valid,
)
from scripts.eat_nav_goal075_review import verify_inputs, freeze_contract, verify_g0_regression_archive


def test_goal_contract_separates_path_distance_from_teacher_horizon():
    for bearing in (-20,-10,0,10,20):
        goal=goal075(bearing)
        assert np.linalg.norm(goal)==pytest.approx(.75,abs=1e-12)
        assert math.degrees(math.atan2(goal[1],goal[0]))==pytest.approx(bearing,abs=1e-12)
    contract=contract075()
    assert contract['future_path_points']==8
    assert contract['teacher_duration_s']==pytest.approx(2.)
    assert contract['nominal_time_to_endpoint_s']==pytest.approx(2.5)
    assert contract['primary_terrain_arc_end_m']==pytest.approx(.6)


def test_nonforward_g0a_tangent_is_not_admitted_even_if_endpoint_matches():
    goal=goal075(20)
    assert not initial_forward_tangent_valid(goal/np.linalg.norm(goal))
    assert initial_forward_tangent_valid(np.array([1.,0.]))


def test_input_freeze_verifies_eight_physical_anchors_and_prior_generator():
    root=os.environ.get('EATNAV_FROZEN_ARTIFACT_ROOT')
    if not root:
        pytest.skip('requires preserved EAT-Nav backup root')
    frozen=verify_inputs(Path(root))
    assert len(frozen['anchors'])==8
    assert len(frozen['old_groups'])==40
    assert frozen['provenance']['prior_curvature_decision']=='candidate_diversity_infeasible_under_frozen_follower_contract'
    assert frozen['provenance']['generator_sha_match']


def test_contract_freeze_refuses_preexisting_output_root(tmp_path):
    with pytest.raises(FileExistsError):
        freeze_contract(tmp_path)


def test_frozen_060_g0_path_sha_regression_matches_all_selected_definitions():
    root=os.environ.get('EATNAV_FROZEN_ARTIFACT_ROOT')
    if not root:pytest.skip('requires preserved EAT-Nav backup root')
    frozen=verify_inputs(Path(root))
    result=verify_g0_regression_archive(
        frozen['old_groups'],frozen['prior']['redesign060']['selected-candidate-definitions.json'])
    assert result['matched_selected_prior_paths']==560
