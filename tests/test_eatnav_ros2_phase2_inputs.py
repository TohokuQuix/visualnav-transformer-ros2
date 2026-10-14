import unittest

import numpy as np
from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

from visualnav_transformer.eat_nav.ros2_phase2.actor_state import build_actor22
from visualnav_transformer.eat_nav.ros2_phase2.gridmap import GridMapLayout, source_grid_from_gridmap
from visualnav_transformer.eat_nav.ros2_phase2.history import DemHistory
from visualnav_transformer.eat_nav.ros2_phase2.wire import actor_to_message, dem_to_message


class InputTest(unittest.TestCase):
    @staticmethod
    def frame(value, stamp, episode=1):
        dem = np.zeros((2, 20, 20), np.float32)
        dem[0] = value
        dem[1] = 1
        return dem_to_message(dem, stamp, stamp, episode)

    def test_history_oldest_first_and_reset(self):
        h = DemHistory()
        for i, stamp in enumerate((250, 500, 750, 1000)):
            h.push(self.frame(i, stamp*1_000_000))
        actor = actor_to_message(np.zeros(22, np.float32), 1_000_000_000, 1,
                                 {'imu': 1_000_000_000, 'joint': 1_000_000_000,
                                  'command': 1_000_000_000})
        dem, state, meta = h.assemble(actor, 1_000_000_000)
        self.assertEqual(dem[:, 0, 0, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(dem.shape, (4, 2, 20, 20))
        self.assertEqual(state.shape, (22,))
        self.assertEqual(meta['episode_id'], 1)
        h.push(self.frame(0, 1_250_000_000, episode=2))
        with self.assertRaises(ValueError):
            h.assemble(actor, 1_250_000_000)

    def test_duplicate_stamp_does_not_refresh_sensor(self):
        h = DemHistory()
        for stamp in (250, 500, 750, 1000):
            h.push(self.frame(0, stamp*1_000_000))
        with self.assertRaises(ValueError):
            h.push(self.frame(2, 1_000_000_000))
        self.assertEqual(h.last_valid_sensor_ns, 1_000_000_000)

    def test_gridmap_circular_negative_invalid(self):
        msg = GridMap()
        msg.header.frame_id = 'map'
        msg.info.resolution = .1
        msg.info.length_x = msg.info.length_y = 2.
        msg.info.pose.position.x = 5.
        msg.info.pose.position.y = -3.
        msg.layers = ['elevation']
        msg.outer_start_index = 3
        msg.inner_start_index = 7
        logical = np.arange(400, dtype=np.float32).reshape(20, 20) / 100 - 2
        logical[4, 6] = np.nan
        storage = np.roll(logical, (3, 7), axis=(0, 1))
        data = Float32MultiArray()
        data.layout.dim = [MultiArrayDimension(label='row', size=20, stride=400),
                           MultiArrayDimension(label='col', size=20, stride=20)]
        data.data = storage.reshape(-1).tolist()
        msg.data = [data]
        grid = source_grid_from_gridmap(msg, GridMapLayout('elevation', 'C', 'yx'))
        self.assertAlmostEqual(grid.origin_xy[0], 4.05)
        self.assertAlmostEqual(grid.origin_xy[1], -3.95)
        self.assertEqual(grid.height[3, 7], storage[3, 7])
        self.assertFalse(grid.valid[(4+3)%20, (6+7)%20])
        self.assertLess(grid.height[3, 7], 0)

    def test_actor22_order_and_missing_effort(self):
        fields = dict(angular_velocity_body=[1, 2, 3], gravity_body=[0, 0, -1],
                      command_body=[.2, 0, -.4], fold_mode=0,
                      flipper_position={'rr': 1, 'rl': 2, 'fr': 3, 'fl': 4},
                      flipper_default={'rr': 0, 'rl': 0, 'fr': 0, 'fl': 0},
                      track_velocity={'right': [10, 20], 'left': [-10, -20]},
                      flipper_effort={'rr': 10, 'rl': 20, 'fr': 30, 'fl': 40},
                      track_effort={'right': [1, 2], 'left': [3, 4]})
        state = build_actor22(**fields)
        self.assertEqual(state.shape, (22,))
        np.testing.assert_allclose(state[:3], [.25, .5, .75])
        np.testing.assert_allclose(state[6:10], [.2, 0, -.4, 0])
        np.testing.assert_allclose(state[10:16], [1, 2, 3, 4, .75, -.75])
        np.testing.assert_allclose(state[16:], [.1, .2, .3, .4, .03, .07])
        del fields['flipper_effort']['fr']
        with self.assertRaises(ValueError):
            build_actor22(**fields)


if __name__ == '__main__':
    unittest.main()
