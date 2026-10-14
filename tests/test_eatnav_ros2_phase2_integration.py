import unittest
import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.nodes import build_mock_evaluation
from visualnav_transformer.eat_nav.ros2_phase2.history import DemHistory
from visualnav_transformer.eat_nav.ros2_phase2.wire import (
    dem_to_message, actor_to_message, CANDIDATE_IDS, WIRE_SHA,
)


class IntegrationContractTest(unittest.TestCase):
    def test_mock_evaluation_roundtrip_and_goal_generation(self):
        history = DemHistory()
        for i, stamp in enumerate((250,500,750,1000)):
            frame = np.zeros((2,20,20),np.float32)
            frame[0] = -.1*i
            frame[1] = 1
            frame[1,0,0] = 0; frame[0,0,0] = 0
            history.push(dem_to_message(frame,stamp*1_000_000,
                                        stamp*1_000_000,1))
        actor = actor_to_message(np.arange(22,dtype=np.float32),1_000_000_000,1,
                                 {k:1_000_000_000 for k in ('imu','joint','command')})
        evaluation = build_mock_evaluation(history,actor,1_000_000_000,
                                           generation=3,robot_pose_map=(0.,0.,0.),
                                           goal_map=(2.,0.))
        self.assertEqual(evaluation.generation,3)
        self.assertEqual(evaluation.predictor_mode,'mock14_transport')
        self.assertEqual(evaluation.bank_source_sha256.__len__(),64)
        self.assertEqual(len(evaluation.predicted_body_xy_m),28)
        self.assertLess(evaluation.selected_index,14)
        self.assertEqual(len(CANDIDATE_IDS),14)
        self.assertEqual(len(WIRE_SHA),64)
        with self.assertRaises(ValueError):
            build_mock_evaluation(history,actor,1_400_000_000,3,(0.,0.,0.),(2.,0.))


if __name__ == '__main__': unittest.main()
