import unittest
import rclpy
from eat_nav_interfaces.msg import PathSelection

from visualnav_transformer.eat_nav.ros2_phase2.mock_robot import MockRobotInterface, MockRobotNode
from visualnav_transformer.eat_nav.ros2_phase2.wire import (
    CANDIDATE_IDS,PATHS,WIRE_SHA,set_stamp)


MS=1_000_000


class MockRobotTest(unittest.TestCase):
    def test_cancel_clears_held_path_and_zeroes_velocity(self):
        robot=MockRobotInterface()
        robot.status(1,True,0)
        robot.path(1,0)
        robot.command((.3,.1),10*MS)
        self.assertEqual(robot.tick(20*MS),(.3,.1))
        robot.status(2,False,30*MS)
        self.assertIsNone(robot.held_path_generation)
        self.assertEqual(robot.tick(30*MS),(0.,0.))
        with self.assertRaises(ValueError):
            robot.path(1,31*MS)

    def test_supervisor_communication_loss_stops(self):
        robot=MockRobotInterface()
        robot.status(1,True,0); robot.path(1,0)
        robot.command((.3,0.),10*MS)
        self.assertEqual(robot.tick(80*MS),(0.,0.))
        robot.command((.3,0.),90*MS)
        self.assertEqual(robot.tick(120*MS),(0.,0.))

    def test_mock_node_rejects_wrong_geometry(self):
        rclpy.init(); node=MockRobotNode()
        try:
            now=node.get_clock().now().nanoseconds
            node.interface.status(1,True,now)
            msg=PathSelection();msg.header.frame_id='chassis/body'
            set_stamp(msg.header.stamp,now)
            set_stamp(msg.expires_at,now+500_000_000)
            msg.generation=1;msg.candidate_id=CANDIDATE_IDS[0]
            msg.direction=1;msg.wire_path_sha256=WIRE_SHA
            msg.body_xy_m=PATHS[0].reshape(-1).tolist()
            msg.body_xy_m[2]=3.
            node._path(msg)
            self.assertIsNone(node.interface.held_path_generation)
            msg.body_xy_m=PATHS[0].reshape(-1).tolist()
            node._path(msg)
            self.assertEqual(node.interface.held_path_generation,1)
        finally:
            node.destroy_node();rclpy.shutdown()


if __name__=='__main__': unittest.main()
