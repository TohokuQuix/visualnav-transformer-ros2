"""Frozen R3 primary and reserve manifest contract."""
import copy
import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.r3_anchor_contract import (
    build_primary, build_reserve, requested_world, split_shards,
)


REVIEW = Path(__file__).resolve().parents[1] / "docs/eat-nav/anchor-placement-offline-20261008"


def fixtures():
    proposal = json.loads((REVIEW / "pilot-candidate-proposal.json").read_text())
    per_pose = json.loads((REVIEW / "per-pose-review-summary.json").read_text())
    return proposal, per_pose


def test_exact_eight_primary_and_shard_order():
    proposal, per_pose = fixtures()
    primary = build_primary(proposal, per_pose)
    a, b = split_shards(primary)
    assert len(primary) == 8 and len({x["pose_id"] for x in primary}) == 8
    assert [x["rank"] for x in a] == [1, 3, 5, 7]
    assert [x["rank"] for x in b] == [2, 4, 6, 8]
    assert len({x["virtual_class"] for x in a}) == 4
    assert len({x["virtual_class"] for x in b}) == 4
    assert all(x["env_index"] == x["row"] * 7 + x["column"] for x in primary)
    assert all(x["batch_seed"] == 20261007 for x in primary)


def test_primary_rejects_modified_pose_and_dem_sha():
    proposal, per_pose = fixtures()
    bad = copy.deepcopy(proposal)
    bad["candidates"][0]["requested_xy_yaw"][0] += .01
    with pytest.raises(ValueError, match="proposal"):
        build_primary(bad, per_pose)
    bad = copy.deepcopy(proposal)
    bad["candidates"][0]["dem_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="proposal"):
        build_primary(bad, per_pose)


def test_reserve_excludes_primary_and_preserves_r3_order():
    proposal, per_pose = fixtures()
    primary = build_primary(proposal, per_pose)
    selected = json.loads((REVIEW / "r0-r3-comparison.json").read_text())
    # The complete R3 order is supplied by the frozen selector on raw scan data.
    # Here use a synthetic ordered prefix to verify exclusion and stable rank.
    ordered = [per_pose[0], per_pose[1], per_pose[2]]
    ordered += [next(x for x in per_pose if x["pose_id"] == row["pose_id"]) for row in primary]
    reserve = build_reserve(ordered, primary)
    assert len(reserve) == 3
    assert [x["selection_rank_in_r3"] for x in reserve] == [1, 2, 3]
    assert not {x["pose_id"] for x in reserve} & {x["pose_id"] for x in primary}
    assert "R3" in selected


def test_requested_pose_matches_env_origin_and_frozen_xy_yaw():
    proposal, per_pose = fixtures()
    row = build_primary(proposal, per_pose)[0]
    candidate = next(x for x in per_pose if x["pose_id"] == row["pose_id"])
    # Reconstruct the scene origin from the exact proposal XY and nominal radius.
    import math
    theta = row["theta_rad"]
    origin = [row["requested_xy_yaw"][0] - row["radius_m"] * math.cos(theta),
              row["requested_xy_yaw"][1] - row["radius_m"] * math.sin(theta), .0625]
    pose = requested_world(row, origin)
    assert pose["world_x"] == candidate["requested_xy_yaw"][0]
    assert pose["world_yaw"] == candidate["requested_xy_yaw"][2]
    with pytest.raises(ValueError, match="origin"):
        requested_world(row, [origin[0] + .1, origin[1], origin[2]])
