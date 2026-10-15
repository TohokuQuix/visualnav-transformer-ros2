import unittest

import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.wire import CANDIDATE_IDS, PATHS, WIRE_SHA
from visualnav_transformer.eat_nav.ros2_phase3_gui_contract import (
    validate_selection, validate_gated_command, validate_gated_stop,
    validate_source_frame,
)


class Phase3GuiContractTest(unittest.TestCase):
    def test_frozen_selection_and_expiry(self):
        selection = {'type': 'selection', 'stamp_ns': 1_000_000_000,
                     'generation': 1, 'candidate_id': CANDIDATE_IDS[2],
                     'wire_path_sha256': WIRE_SHA, 'body_xy_m': PATHS[2].reshape(-1).tolist(),
                     'expires_ns': 1_500_000_000}
        self.assertEqual(validate_selection(selection, 1_000_000_000), 2)
        for key, value in [('candidate_id', CANDIDATE_IDS[3]),
                           ('wire_path_sha256', 'wrong'), ('generation', 0),
                           ('expires_ns', 999_999_999)]:
            bad = {**selection, key: value}
            with self.assertRaises(ValueError, msg=key):
                validate_selection(bad, 1_000_000_000)
        with self.assertRaises(ValueError):
            validate_selection({**selection, 'body_xy_m': [0.0] * 16}, 1_000_000_000)

    def test_source_frame_and_command_are_exact(self):
        frame = {'type': 'source', 'stamp_ns': 1_000_000_000, 'episode_id': 2,
                 'pose_map': [1., 2., 0.], 'actor_pre_noise': [0.] * 22,
                 'dem': np.stack((np.zeros((20,20)), np.ones((20,20)))).tolist(),
                 'dem_stamp_ns': 1_000_000_000,
                 'dem_sensor_update_ns': 1_000_000_000}
        validate_source_frame(frame)
        with self.assertRaises(ValueError):
            validate_source_frame({**frame, 'dem_stamp_ns': 950_000_000})
        command = {'type': 'gated_command', 'stamp_ns': 1_000_000_000,
                   'generation': 1, 'gate_open': True, 'command': [.2, .1]}
        self.assertEqual(validate_gated_command(command, 1_000_000_000, 1), (.2, .1))
        with self.assertRaises(ValueError):
            validate_gated_command({**command, 'stamp_ns': 950_000_000}, 1_000_000_000, 1)
        with self.assertRaises(ValueError):
            validate_gated_command({**command, 'gate_open': False}, 1_000_000_000, 1)

    def test_closed_gate_accepts_only_exact_stamped_zero_stop(self):
        stop = {'type': 'gated_command', 'stamp_ns': 31_100_000_000,
                'generation': 2, 'gate_open': False, 'command': [0., 0.]}
        self.assertEqual(validate_gated_stop(stop, 31_100_000_000, 2), (0., 0.))
        for replacement in ({'command': [0.2, 0.]}, {'stamp_ns': 31_050_000_000},
                            {'generation': 3}, {'gate_open': True}):
            with self.assertRaises(ValueError):
                validate_gated_stop({**stop, **replacement}, 31_100_000_000, 2)


if __name__ == '__main__':
    unittest.main()
