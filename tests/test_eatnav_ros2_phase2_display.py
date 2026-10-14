import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

from visualnav_transformer.eat_nav.ros2_phase2.display_node import DisplayNode
from visualnav_transformer.eat_nav.ros2_phase2.nodes import QOS
from visualnav_transformer.eat_nav.ros2_phase2.wire import bank_to_message


class DisplayTest(unittest.TestCase):
    def test_display_does_not_own_navigation(self):
        rclpy.init(); display=DisplayNode(); observer=Node('eatnav_display_observer')
        received=[]
        observer.create_subscription(MarkerArray,'/eat_nav/display/paths',
                                     received.append,QOS)
        ex=SingleThreadedExecutor();ex.add_node(display);ex.add_node(observer)
        try:
            display._bank(bank_to_message(1_000_000_000))
            for _ in range(10):
                ex.spin_once(timeout_sec=.03)
                if received:break
            self.assertTrue(received)
            self.assertEqual(len(received[-1].markers),14)
            self.assertTrue(all(len(m.points)==8 for m in received[-1].markers))
            self.assertIsNone(display.last_status)
        finally:
            ex.shutdown();display.destroy_node();observer.destroy_node();rclpy.shutdown()


if __name__=='__main__': unittest.main()
