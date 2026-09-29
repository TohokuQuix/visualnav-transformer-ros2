import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_seed_prefix():
    from eat_nav_phase_c_coverage import validate_config
    cfg = {'states': {'diffusion_seeds': list(range(1001, 1033))}, 'sampling': {'baseline_k': [8, 32]}, 'reference': {'heldout_forbidden': True}, 'formal_phase_c_allowed': False, 'structured': {'basis': 'global_complete_train_residual_pca'}}
    validate_config(cfg)
    cfg['states']['diffusion_seeds'][0] = 999
    try:
        validate_config(cfg)
    except ValueError:
        pass
    else:
        raise AssertionError('changed frozen seed accepted')


def test_resume_marker_requires_matching_output(tmp_path):
    from eat_nav_phase_c_coverage import write_marker, marker_valid
    output = tmp_path / 'output.json'
    output.write_text('{"result":1}')
    marker = tmp_path / 'stage.done.json'
    write_marker(marker, [output], 'abc')
    assert marker_valid(marker, 'abc')
    output.write_text('{"result":2}')
    assert not marker_valid(marker, 'abc')


def test_train_only_source_rejected_for_validation_and_test():
    from eat_nav_phase_c_coverage import require_train_only
    require_train_only({'split': 'train'})
    for split in ('val', 'test'):
        try:
            require_train_only({'split': split})
        except ValueError:
            pass
        else:
            raise AssertionError(split)
