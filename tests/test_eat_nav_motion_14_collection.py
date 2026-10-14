from collections import Counter
import json
from types import SimpleNamespace

import numpy as np
import torch

from visualnav_transformer.eat_nav.motion_14_collection import (
    COLLECTION_SCHEMA, build_collection_schedule, seed_groups, split_seeds,
    validate_collection_schedule,
)


def test_unique_seed_balanced_split_and_trial_counts():
    groups = seed_groups()
    assert len(groups) == len({g['terrain_seed'] for g in groups}) == 36
    assert all(0 <= g['terrain_seed'] <= 2**32 - 1 for g in groups)
    assert set(split_seeds(groups)) == {'train', 'validation', 'test'}
    assert all(len(seeds) == 12 for seeds in split_seeds(groups).values())
    cells = Counter((g['terrain_type'], g['terrain_level']) for g in groups)
    assert len(cells) == 12 and set(cells.values()) == {3}
    split_cells = Counter((g['split'], g['terrain_type'], g['terrain_level']) for g in groups)
    assert len(split_cells) == 36 and set(split_cells.values()) == {1}
    rows = build_collection_schedule(groups)
    assert len(rows) == 1152
    assert sum(r['stage'] == 'candidate' for r in rows) == 1008
    assert sum(r['stage'] == 'prepath' for r in rows) == 144
    validate_collection_schedule(rows, groups)
    assert COLLECTION_SCHEMA
    from pathlib import Path
    old = json.loads((Path(__file__).resolve().parents[1]/'docs/eat-nav/'
                      'motion-predictor-vnext/mi_a_v31_assignment_manifest.json').read_text())
    assert not {g['terrain_seed'] for g in old['state_groups']} & \
               {g['terrain_seed'] for g in groups}


def test_raw_dem_capture_preserves_native_hits_and_valid_mask():
    # Importing the Isaac simulator script would start AppLauncher; exercise its
    # isolated source logic without launching Isaac.
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1]/'scripts/eat_nav_motion_14_collection_sim.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'capture_raw_dem')
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {'np': np, 'NUM_ENVS': 4}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    starts = torch.zeros((4, 400, 3))
    hits = starts.clone()
    hits[1, 25, 2] = float('inf')
    # An explicit indexing proxy mirrors Isaac's Scene API.
    class Scene:
        env_origins = torch.zeros((4, 3))

        def __getitem__(self, key):
            return {'height_scanner_eat': SimpleNamespace(
                        data=SimpleNamespace(ray_hits_w=hits), _ray_starts_w=starts),
                    'robot': SimpleNamespace(data=SimpleNamespace(
                        root_state_w=torch.zeros((4, 13))))}[key]

    raw = namespace['capture_raw_dem'](SimpleNamespace(scene=Scene()))
    assert raw['ray_hits_w'].shape == (4, 400, 3)
    assert raw['ray_valid'].shape == (4, 400)
    assert not raw['ray_valid'][1, 25]
    assert np.isinf(raw['ray_hits_w'][1, 25, 2])
