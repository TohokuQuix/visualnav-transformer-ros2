#!/usr/bin/env /usr/bin/python3
"""One-shot ROS goal smoke. RViz2 uses the same /goal_pose PoseStamped topic."""

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class GoalOnce(Node):
    def __init__(self):
        super().__init__('eat_nav_demo_goal_once')
        self.pose = None
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.create_subscription(Odometry, '/eat_nav/odom', self._on_pose, 10)

    def _on_pose(self, item):
        self.pose = item

    def publish(self, distance):
        current = self.pose
        q = current.pose.pose.orientation
        yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
        goal = PoseStamped()
        goal.header = current.header
        goal.pose.position.x = current.pose.pose.position.x + distance*math.cos(yaw)
        goal.pose.position.y = current.pose.pose.position.y + distance*math.sin(yaw)
        goal.pose.orientation.w = 1.
        self.pub.publish(goal)
        print(f'goal_map=({goal.pose.position.x:.3f},{goal.pose.position.y:.3f}) '
              f'stamp={goal.header.stamp.sec}.{goal.header.stamp.nanosec:09d}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', type=float, default=.75)
    parser.add_argument('--timeout', type=float, default=240.)
    args = parser.parse_args()
    if not .05 <= args.distance <= .8:
        raise ValueError('local goal distance must be in [.05,.8] m')
    rclpy.init()
    node = GoalOnce()
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.05)
            pose = node.pose
            if pose and pose.header.stamp.sec + pose.header.stamp.nanosec/1e9 >= 1.2 and \
                    node.pub.get_subscription_count():
                node.publish(args.distance)
                # Let the transport deliver the one message before exiting.
                for _ in range(5):
                    rclpy.spin_once(node, timeout_sec=.05)
                return
        raise TimeoutError('ROS state/goal subscriber unavailable')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
