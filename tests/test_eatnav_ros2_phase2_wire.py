"""CPU-only ROS wire contract tests; no Isaac or robot topics."""

import unittest

import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.wire import (
    actor_from_message,
    actor_to_message,
    bank_from_message,
    bank_to_message,
    dem_from_message,
    dem_to_message,
)


class WireTest(unittest.TestCase):
    def test_dem_negative_and_invalid_roundtrip(self):
        dem = np.zeros((2, 20, 20), dtype=np.float32)
        dem[0, 3, 4] = -0.375
        dem[0, 10, 8] = 0.625
        dem[1] = 1
        dem[1, 5, 6] = 0
        msg = dem_to_message(dem, 1_000_000_000, 990_000_000, 7)
        recovered, meta = dem_from_message(msg)
        self.assertEqual(dem.tobytes(), recovered.tobytes())
        self.assertEqual(meta['stamp_ns'], 1_000_000_000)
        self.assertEqual(meta['episode_id'], 7)
        self.assertEqual(recovered[0, 3, 4], -0.375)
        self.assertEqual(recovered[1, 5, 6], 0)

    def test_dem_rejects_nonzero_invalid_height(self):
        dem = np.zeros((2, 20, 20), dtype=np.float32)
        dem[0, 1, 1] = 3
        with self.assertRaises(ValueError):
            dem_to_message(dem, 1, 1, 0)

    def test_actor_roundtrip_and_shape(self):
        state = np.linspace(-1, 1, 22, dtype=np.float32)
        msg = actor_to_message(state, 1_000_000_000, 7,
                               {'imu': 1_000_000_000, 'joint': 1_000_000_000,
                                'command': 1_000_000_000})
        recovered, meta = actor_from_message(msg)
        self.assertEqual(state.tobytes(), recovered.tobytes())
        self.assertEqual(meta['contract_id'], 'onix_policy_pre_noise_22d_v1')
        self.assertEqual(meta['episode_id'], 7)

    def test_bank_roundtrip_and_direction(self):
        msg = bank_to_message(1_000_000_000)
        arrays, meta = bank_from_message(msg)
        self.assertEqual(arrays['paths'].shape, (14, 8, 2))
        self.assertEqual(arrays['direction'].reshape(-1).tolist(), [1]*7+[-1]*7)
        self.assertEqual(arrays['mask'].tolist(), [True]*14)
        self.assertEqual(meta['ids'][0], 'forward_direct')
        self.assertEqual(meta['ids'][-1], 'reverse_+90')
        msg.candidate_ids[0] = 'reverse_direct'
        with self.assertRaises(ValueError):
            bank_from_message(msg)


if __name__ == '__main__':
    unittest.main()
