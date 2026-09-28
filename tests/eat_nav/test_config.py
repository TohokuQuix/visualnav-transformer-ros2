from pathlib import Path

import pytest
import yaml

from visualnav_transformer.eat_nav.config import (
    EATConfig,
    assert_checkpoint_compatible,
    load_eat_config,
)


def test_default_config_sets_history_point_count_and_derived_horizon():
    config = load_eat_config(Path(__file__).parents[2] / "config/eat_nav.yaml")

    assert config.context_size == 3
    assert config.dem_frame_count == 4
    assert config.len_traj_pred == 8
    assert config.trajectory_interval_sec == 0.25
    assert config.trajectory_horizon_sec == 2.0
    assert config.dem_history_interval_sec == 0.25
    assert (config.dem_height_cells, config.dem_width_cells, config.dem_resolution_m) == (20, 20, 0.1)


def test_point_count_and_interval_change_horizon_independently(tmp_path):
    path = tmp_path / "eat.yaml"
    path.write_text("len_traj_pred: 12\ntrajectory_interval_sec: 0.2\n")
    config = load_eat_config(path)

    assert config.len_traj_pred == 12
    assert config.trajectory_horizon_sec == pytest.approx(2.4)
    assert config.dem_frame_count == 4


@pytest.mark.parametrize(
    "overrides",
    [
        "len_traj_pred: 0\n",
        "trajectory_interval_sec: 0\n",
        "trajectory_interval_sec: -0.1\n",
        "context_size: 4\n",
        "trajectory_horizon_sec: 1.6\n",
        "dem_width_cells: 0\n",
    ],
)
def test_invalid_trajectory_or_geometry_config_is_rejected(tmp_path, overrides):
    path = tmp_path / "bad.yaml"
    path.write_text(overrides)

    with pytest.raises(ValueError):
        load_eat_config(path)


def test_explicit_consistent_horizon_is_accepted(tmp_path):
    path = tmp_path / "eat.yaml"
    path.write_text("len_traj_pred: 12\ntrajectory_interval_sec: 0.2\ntrajectory_horizon_sec: 2.4\n")

    assert load_eat_config(path).trajectory_horizon_sec == pytest.approx(2.4)


@pytest.mark.parametrize(
    "change",
    [
        {"len_traj_pred": 12},
        {"trajectory_interval_sec": 0.2},
        {"context_size": 5},
        {"dem_width_cells": 21},
        {"dem_resolution_m": 0.2},
    ],
)
def test_checkpoint_rejects_changed_input_contract(change):
    config = EATConfig()
    metadata = config.checkpoint_metadata()
    metadata.update(change)

    with pytest.raises(ValueError, match="checkpoint"):
        assert_checkpoint_compatible(config, metadata)


def test_checkpoint_metadata_round_trips_through_yaml():
    config = EATConfig()
    metadata = yaml.safe_load(yaml.safe_dump(config.checkpoint_metadata()))

    assert_checkpoint_compatible(config, metadata)


def test_checkpoint_requires_supported_schema():
    config = EATConfig()
    metadata = config.checkpoint_metadata()
    metadata["schema_version"] = 99

    with pytest.raises(ValueError, match="schema"):
        assert_checkpoint_compatible(config, metadata)
