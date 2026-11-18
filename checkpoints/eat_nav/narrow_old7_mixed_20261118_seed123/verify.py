"""Verify transfer bytes and run synthetic CPU inference (not an accuracy test)."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'inference_src'))
import numpy as np
import torch
from visualnav_transformer.eat_nav.single_pose2s_contract import INPUT_SCHEMA, MODEL_SCHEMA, PROFILE_SHA
from visualnav_transformer.eat_nav.single_pose2s_model import predict
from visualnav_transformer.eat_nav.bidirectional_bank import bank


def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    payload = torch.load(ROOT / 'checkpoint.pt', map_location='cpu', weights_only=True)
    assert payload['schema'] == MODEL_SCHEMA
    assert payload['input_schema'] == INPUT_SCHEMA
    assert payload['profile_sha256'] == PROFILE_SHA == manifest['input_profile_sha256']
    assert payload['normalization'] == json.loads((ROOT / 'normalization.json').read_text())
    assert payload['best_epoch'] == 21
    state = np.zeros((1, 27), np.float32)
    state[0, 2] = -1
    state[0, 25] = 1
    dem = np.zeros((1, 3, 40, 40), np.float32)
    dem[:, 1:] = 1
    candidates = bank()
    sample = dict(input_schema=INPUT_SCHEMA, profile_sha256=PROFILE_SHA,
                  dem=dem, state=state, state_mask=np.ones((1, 27), np.float32),
                  state_age=np.zeros((1, 27), np.float32), dem_context=np.zeros((1, 6), np.float32),
                  paths=np.array([[c.waypoints for c in candidates]], np.float32),
                  directions=np.array([[[c.direction] for c in candidates]], np.float32),
                  t0_ns=np.array([1000000000]), dem_stamp_ns=np.array([1000000000]),
                  dem_received_ns=np.array([1000000000]))
    torch.set_num_threads(1)
    result = predict(payload, sample, device='cpu')
    assert result['position_body_m'].shape == (1, 14, 3)
    assert np.isfinite(result['position_body_m']).all()
    assert result['rotation_valid'].all()
    print(json.dumps(dict(state='passed', checkpoint_sha256=manifest['checkpoint_sha256'],
                          best_epoch=payload['best_epoch'], candidates=14,
                          torch_version=str(torch.__version__), numpy_version=np.__version__,
                          kind='byte verification and synthetic CPU inference; not an accuracy evaluation')))


if __name__ == '__main__':
    main()
