import time
import unittest
from pathlib import Path

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty
from eat_nav_interfaces.msg import PathSelection, NavigationStatus
from geometry_msgs.msg import TwistStamped

from visualnav_transformer.eat_nav.ros2_phase2.nodes import (
    InferenceNode, SupervisorGatewayNode, QOS)
from visualnav_transformer.eat_nav.ros2_phase2.source_nodes import SealedReplayNode
from visualnav_transformer.eat_nav.ros2_phase2.mock_robot import MockRobotNode
from visualnav_transformer.eat_nav.ros2_phase2.wire import set_stamp


FIXTURE=Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/motion-14-collection-20261014-02/0100-pyramid_stairs-level2-seed1653838991-front_folded-forward_+30')
LEGACY=Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01/training/full-seed456/best.pt')


class PipelineTest(unittest.TestCase):
    def test_gui_less_mock_pipeline_and_stop(self):
        if not FIXTURE.exists(): self.skipTest('sealed sample unavailable')
        rclpy.init()
        source=SealedReplayNode(FIXTURE)
        inference=InferenceNode()
        supervisor=SupervisorGatewayNode()
        robot=MockRobotNode()
        observer=Node('eat_nav_phase2_cpu_observer')
        selections=[]; statuses=[]; commands=[]; applied=[]
        observer.create_subscription(PathSelection,'/eat_nav/mock/selection',
                                     selections.append,QOS)
        observer.create_subscription(NavigationStatus,'/eat_nav/status',
                                     statuses.append,QOS)
        observer.create_subscription(TwistStamped,'/eat_nav/mock/final_cmd',
                                     commands.append,QOS)
        observer.create_subscription(TwistStamped,'/eat_nav/mock/applied_cmd',
                                     applied.append,QOS)
        cancel=observer.create_publisher(Empty,'/eat_nav/cancel',QOS)
        raw=observer.create_publisher(TwistStamped,'/eat_nav/mock/follower_cmd',QOS)
        ex=SingleThreadedExecutor()
        for node in (source,inference,supervisor,robot,observer): ex.add_node(node)
        try:
            deadline=time.monotonic()+4.
            while time.monotonic()<deadline and not selections:
                ex.spin_once(timeout_sec=.05)
            self.assertTrue(selections,'no selected Path through CPU ROS graph')
            self.assertEqual(selections[-1].generation,1)
            command=TwistStamped();command.header.frame_id='chassis/body'
            set_stamp(command.header.stamp,observer.get_clock().now().nanoseconds)
            command.twist.linear.x=.3
            raw.publish(command)
            deadline=time.monotonic()+.35
            while time.monotonic()<deadline and not any(
                    abs(c.twist.linear.x-.3)<1e-5 for c in applied):
                ex.spin_once(timeout_sec=.02)
            self.assertTrue(any(abs(c.twist.linear.x-.3)<1e-5 for c in applied),
                            'mock actuator never applied live command')
            cancel.publish(Empty())
            deadline=time.monotonic()+1.
            while time.monotonic()<deadline and not any(s.state=='CANCELLED' for s in statuses):
                ex.spin_once(timeout_sec=.05)
            self.assertTrue(any(s.state=='CANCELLED' for s in statuses))
            deadline=time.monotonic()+.5
            while time.monotonic()<deadline and (not commands or
                    commands[-1].twist.linear.x != 0.):
                ex.spin_once(timeout_sec=.02)
            self.assertFalse(statuses[-1].command_gate_open)
            self.assertIsNone(robot.interface.held_path_generation)
            self.assertEqual(robot.interface.tick(observer.get_clock().now().nanoseconds),
                             (0.,0.))
            self.assertTrue(commands)
            self.assertEqual(commands[-1].twist.linear.x,0.)
            self.assertEqual(commands[-1].twist.angular.z,0.)
        finally:
            ex.shutdown()
            for node in (source,inference,supervisor,robot,observer): node.destroy_node()
            rclpy.shutdown()

    def test_legacy_mode_uses_forward_only_and_pinned_checkpoint(self):
        if not FIXTURE.exists() or not LEGACY.exists():
            self.skipTest('sealed sample/legacy checkpoint unavailable')
        rclpy.init()
        source=SealedReplayNode(FIXTURE)
        inference=InferenceNode(mode='legacy6_demo',checkpoint=LEGACY,
            python=Path('/home/kojima/anaconda3/envs/tinyvla/bin/python'))
        supervisor=SupervisorGatewayNode(mode='legacy6_demo',checkpoint=LEGACY)
        observer=Node('eat_nav_phase2_legacy_observer')
        selections=[]
        observer.create_subscription(PathSelection,'/eat_nav/mock/selection',
                                     selections.append,QOS)
        ex=SingleThreadedExecutor()
        for node in (source,inference,supervisor,observer): ex.add_node(node)
        try:
            deadline=time.monotonic()+4.
            while time.monotonic()<deadline and not selections:
                ex.spin_once(timeout_sec=.05)
            self.assertTrue(selections,'legacy Path not selected')
            self.assertEqual(selections[-1].direction,1)
            self.assertIn(selections[-1].candidate_id,
                          ('forward_direct','forward_-30','forward_+30',
                           'forward_-60','forward_+60','forward_-90','forward_+90'))
        finally:
            ex.shutdown()
            for node in (source,inference,supervisor,observer): node.destroy_node()
            rclpy.shutdown()


if __name__=='__main__': unittest.main()
