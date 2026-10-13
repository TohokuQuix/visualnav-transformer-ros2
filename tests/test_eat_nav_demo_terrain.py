from types import SimpleNamespace

import pytest

from visualnav_transformer.eat_nav.demo_terrain import TERRAIN_TYPES, configure_selected_terrain, terrain_band


def test_35_terrain_levels_and_distinct_tiles():
    assert len(TERRAIN_TYPES) == 7
    for terrain_type in TERRAIN_TYPES:
        for level in range(5):
            cfg = SimpleNamespace(sub_terrains={terrain_type: SimpleNamespace(proportion=1 / 7)},
                                  difficulty_range=(0.0, 1.0), use_cache=True)
            importer = SimpleNamespace()
            metadata = configure_selected_terrain(cfg, importer, terrain_type, level, 1)
            assert (cfg.num_rows, cfg.num_cols, cfg.curriculum) == (5, 1, True)
            assert importer.eat_nav_selected_level == level
            assert metadata['difficulty_band'] == pytest.approx((level / 5, (level + 1) / 5))
            assert cfg.sub_terrains[terrain_type].proportion == 1.0


def test_four_env_uses_four_columns_and_preserves_original_difficulty_range():
    cfg = SimpleNamespace(sub_terrains={'boxes': SimpleNamespace(proportion=.2)},
                          difficulty_range=(0., 1.5))
    importer = SimpleNamespace()
    result = configure_selected_terrain(cfg, importer, 'boxes', 2, 4)
    assert result['difficulty_band'] == pytest.approx((.6, .9))
    assert (cfg.num_rows, cfg.num_cols) == (5, 4)


@pytest.mark.parametrize('level', [-1, 5])
def test_reject_invalid_level(level):
    with pytest.raises(ValueError):
        terrain_band(level, (0., 1.))
