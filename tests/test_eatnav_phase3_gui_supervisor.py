import unittest

import rclpy
from geometry_msgs.msg import TwistStamped

from visualnav_transformer.eat_nav.ros2_phase2.nodes import SupervisorGatewayNode
from visualnav_transformer.eat_nav.ros2_phase2.wire import set_stamp


class Phase3SupervisorTest(unittest.TestCase):
    def test_event_driven_only_when_enabled(self):
        rclpy.init()
        try:
            for enabled in (False, True):
                node = SupervisorGatewayNode(simulator_event_driven=enabled)
                calls = []
                node._tick = lambda: calls.append(1)
                message = TwistStamped()
                message.header.frame_id = 'chassis/body'
                set_stamp(message.header.stamp, 1_000_000_000)
                node._follower_command(message)
                self.assertEqual(len(calls), int(enabled))
                node.destroy_node()
        finally:
            rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
