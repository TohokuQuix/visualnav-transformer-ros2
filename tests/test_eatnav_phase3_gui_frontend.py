import json
import unittest

import numpy as np
from eat_nav_interfaces.msg import PathSelection

from eat_nav_phase3_gui_frontend import Phase3GuiFrontend
from visualnav_transformer.eat_nav.ros2_phase2.wire import (
    CANDIDATE_IDS, PATHS, WIRE_SHA, set_stamp,
)


class FrontendSerializationTest(unittest.TestCase):
    def test_closed_supervisor_gate_acknowledges_zero_with_nonzero_request(self):
        class Sink:
            pending_command = {'stamp_ns': 31_100_000_000, 'generation': 2,
                               'command': [0.25, 0.1]}
            pending_final = (0., 0.)
            status = (3, 'TIMEOUT', False, 31_100_000_000)
            def _send(self, value):
                self.sent = value
        sink = Sink()
        Phase3GuiFrontend._finish_command_if_ready(sink)
        self.assertEqual(sink.sent['command'], [0., 0.])
        self.assertFalse(sink.sent['gate_open'])
        self.assertIsNone(sink.pending_command)

    def test_numpy_float32_path_is_json_transportable(self):
        message = PathSelection()
        set_stamp(message.header.stamp, 2_000_000_000)
        set_stamp(message.expires_at, 2_500_000_000)
        message.generation = 1
        message.candidate_id = CANDIDATE_IDS[0]
        message.wire_path_sha256 = WIRE_SHA
        message.body_xy_m = np.asarray(PATHS[0].reshape(-1), np.float32)
        class Sink:
            latest_ns = 2_000_000_000
            latest_pose = (1., 2., 0.)
            last_selection = None
            selection_count = 0
            class Pub:
                def publish(self, value):
                    self.last = value
            selected_path_pub = Pub()
            def _send(self, value):
                self.wire = json.dumps(value, allow_nan=False)
        sink = Sink()
        Phase3GuiFrontend._selection(sink, message)
        self.assertEqual(json.loads(sink.wire)['candidate_id'], CANDIDATE_IDS[0])
        self.assertEqual(sink.selection_count, 1)
        self.assertAlmostEqual(sink.selected_path_pub.last.poses[-1].pose.position.x,
                               1. + float(PATHS[0][-1, 0]), places=6)


if __name__ == '__main__':
    unittest.main()
