"""Regression tests for the adopted predictor's command input contract."""

import importlib.util
import json
from pathlib import Path

import numpy as np

from eat_nav_interfaces.msg import ActorState22
from visualnav_transformer.eat_nav.ros2_phase2.nodes import build_model_evaluation
from visualnav_transformer.eat_nav.ros2_phase2 import nodes


ROOT = Path(__file__).resolve().parents[1]


class History:
    def __init__(self):
        self.dem = np.zeros((4, 2, 20, 20), np.float32)
        self.state = np.arange(22, dtype=np.float32) / 10

    def assemble(self, actor, now_ns):
        return self.dem, self.state, {}


class Worker:
    def __init__(self):
        self.inputs = []

    def predict_adopted(self, dem, state):
        self.inputs.append(state.copy())
        return np.zeros((14, 2), np.float32), 'checkpoint-sha'


def evaluate(history, worker, **kwargs):
    return build_model_evaluation(
        history, ActorState22(), 100, 1, (0., 0., 0.), (0., 2.),
        mode='learned14_adopted', checkpoint='checkpoint', python='python',
        results='results', normalization='normalization', worker=worker,
        **kwargs,
    )


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_on_explicit_on_and_off_keep_original_actor_input():
    history = History()
    original = history.state.copy()
    worker = Worker()
    evaluate(history, worker)
    evaluate(history, worker, predictor_zero_velocity_command=True)
    evaluate(history, worker, predictor_zero_velocity_command=False)
    assert np.array_equal(worker.inputs[0], worker.inputs[1])
    assert np.array_equal(worker.inputs[0][6:9], np.zeros(3))
    assert np.array_equal(worker.inputs[0][:6], original[:6])
    assert np.array_equal(worker.inputs[0][9:], original[9:])
    assert np.array_equal(worker.inputs[2], original)
    assert np.array_equal(history.state, original)


def test_phase2_and_student_gui_cli_defaults_and_overrides():
    phase2 = load_script('eat_nav_ros2_phase2')
    gui = load_script('eat_nav_phase3_gui_launch')
    for args in (['inference'], []):
        parser = phase2.build_parser() if args else gui.build_parser()
        assert parser.parse_args(args).predictor_zero_velocity_command is True
        assert parser.parse_args(args + ['--predictor-zero-velocity-command']).predictor_zero_velocity_command is True
        assert parser.parse_args(args + ['--no-predictor-zero-velocity-command']).predictor_zero_velocity_command is False


def test_gui_launch_passes_setting_to_inference_only():
    gui = load_script('eat_nav_phase3_gui_launch')
    assert gui.predictor_command_cli_args(True) == ['--predictor-zero-velocity-command']
    assert gui.predictor_command_cli_args(False) == ['--no-predictor-zero-velocity-command']


def test_zeroing_precedes_stage_c_normalization_and_off_reproduces_old_input(tmp_path, monkeypatch):
    history = History()
    history.dem[:, 1] = 1
    original = history.state.copy()
    norm = {'fitted_from': 'train_only', 'dem_height_mean': 0., 'dem_height_std': 1.,
            'state_mean': [0.] * 22, 'state_std': [1.] * 6 + [.001] * 3 + [1.] * 13,
            'target_mean': [0., 0.], 'target_std': [1., 1.]}
    results = tmp_path / 'results.json'
    normalization = tmp_path / 'normalization.json'
    results.write_text(json.dumps({'source': {}}))
    normalization.write_text(json.dumps(norm))
    received = []

    def fake_worker(python, checkpoint, result_path, source, values, normalization):
        received.append(values[1].copy())
        return np.zeros((14, 2), np.float32), 'sha'

    monkeypatch.setattr(nodes, 'worker_request', fake_worker)
    for enabled in (True, False):
        build_model_evaluation(
            history, ActorState22(), 100, 1, (0., 0., 0.), (0., 2.),
            mode='learned14_adopted', checkpoint='checkpoint', python='python',
            results=results, normalization=normalization,
            predictor_zero_velocity_command=enabled,
        )
    assert np.array_equal(received[0][0, 6:9], np.zeros(3))
    np.testing.assert_allclose(received[1][0], original / np.asarray(norm['state_std']),
                               atol=1e-4, rtol=1e-7)
    assert np.array_equal(history.state, original)
