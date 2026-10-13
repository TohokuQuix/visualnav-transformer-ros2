#!/usr/bin/env /usr/bin/python3
"""ROS2 transport for the Isaac-owned EAT demo loop. No policy or follower here."""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import struct
import time

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32MultiArray, String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


def stamp(ns):
    result = Time()
    result.sec, result.nanosec = divmod(int(ns), 1_000_000_000)
    return result


def pose_message(xy, yaw, frame, stamp_ns):
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.header.stamp = stamp(stamp_ns)
    msg.pose.position.x, msg.pose.position.y = map(float, xy)
    msg.pose.orientation.z = math.sin(yaw / 2)
    msg.pose.orientation.w = math.cos(yaw / 2)
    return msg


class DemoFrontend(Node):
    def __init__(self, sock):
        super().__init__('eat_nav_demo_frontend')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.sock = sock
        self.buffer = b''
        self.latest_ns = None
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub = self.create_publisher(Odometry, '/eat_nav/odom', 10)
        self.dem_pub = self.create_publisher(Image, '/eat_nav/dem/height', 10)
        self.mask_pub = self.create_publisher(Image, '/eat_nav/dem/valid_mask', 10)
        self.actor_pub = self.create_publisher(Float32MultiArray, '/eat_nav/actor_pre_noise', 10)
        self.candidates_pub = self.create_publisher(MarkerArray, '/eat_nav/candidates', 10)
        self.selected_pub = self.create_publisher(Path, '/eat_nav/selected_path', 10)
        self.score_pub = self.create_publisher(String, '/eat_nav/scores', 10)
        self.status_pub = self.create_publisher(String, '/eat_nav/status', 10)
        self.tf_pub = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseStamped, '/goal_pose', self._on_goal, 10)
        self.create_subscription(Empty, '/eat_nav/cancel', self._on_cancel, 10)

    def _send(self, payload):
        self.sock.sendall((json.dumps(payload, allow_nan=False, separators=(',', ':')) + '\n').encode())

    def _on_cancel(self, _):
        self._send({'type': 'cancel', 'stamp_ns': self.latest_ns})

    def _on_goal(self, goal):
        if self.latest_ns is None:
            self.get_logger().error('Goal rejected: simulator state unavailable')
            return
        given_ns = goal.header.stamp.sec * 1_000_000_000 + goal.header.stamp.nanosec
        if given_ns <= 0 or abs(given_ns - self.latest_ns) > 500_000_000:
            self.get_logger().error('Goal rejected: stale/future stamp')
            return
        try:
            if goal.header.frame_id == 'map':
                xy = (goal.pose.position.x, goal.pose.position.y)
            else:
                from tf2_geometry_msgs import do_transform_pose_stamped
                transform = self.tf_buffer.lookup_transform('map', goal.header.frame_id,
                                                           rclpy.time.Time.from_msg(goal.header.stamp))
                translated = do_transform_pose_stamped(goal, transform)
                xy = (translated.pose.position.x, translated.pose.position.y)
            if not all(math.isfinite(x) for x in xy):
                raise ValueError('nonfinite goal')
            self._send({'type': 'goal', 'frame': 'map', 'xy': xy, 'stamp_ns': given_ns})
            self.get_logger().info(f'Goal map xy={xy}, stamp_ns={given_ns}')
        except Exception as exc:
            self.get_logger().error(f'Goal rejected: TF/validation {exc}')

    def _image(self, values, encoding, stamp_ns):
        image = Image()
        image.header.frame_id = 'base_link'
        image.header.stamp = stamp(stamp_ns)
        image.width = image.height = 20
        image.encoding = encoding
        image.is_bigendian = False
        if encoding == '32FC1':
            image.step = 80
            image.data = struct.pack('<400f', *[float(v) for row in values for v in row])
        else:
            image.step = 20
            image.data = bytes(int(v) for row in values for v in row)
        return image

    def _publish_paths(self, item, stamp_ns):
        candidates = item.get('candidate_paths_map', {})
        scores = item.get('scores_m', {})
        selected = item.get('selected')
        bundle = MarkerArray()
        erase = Marker()
        erase.action = Marker.DELETEALL
        bundle.markers.append(erase)
        for idx, role in enumerate(('direct', 'left_20', 'right_20')):
            if role not in candidates:
                continue
            points = candidates[role]
            color = (0.1, 1., 0.2) if selected == role else ((.2, .6, 1.) if idx == 0 else (1., .55, .1))
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = stamp(stamp_ns)
            marker.ns, marker.id, marker.type, marker.action = 'eat_nav_paths', idx, Marker.LINE_STRIP, Marker.ADD
            marker.scale.x = .035 if role == selected else .018
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (*color, 1.)
            for xy in points:
                from geometry_msgs.msg import Point
                marker.points.append(Point(x=float(xy[0]), y=float(xy[1]), z=.1))
            bundle.markers.append(marker)
            label = Marker()
            label.header = marker.header
            label.ns, label.id, label.type, label.action = 'eat_nav_scores', idx, Marker.TEXT_VIEW_FACING, Marker.ADD
            label.scale.z = .075
            label.color.r = label.color.g = label.color.b = label.color.a = 1.
            label.pose.position.x = float(points[-1][0]) + .22
            label.pose.position.y = float(points[-1][1]) + (idx - 1) * .35
            label.pose.position.z = .25
            label.text = f'{role}={scores.get(role, float("nan")):.2f}m'
            bundle.markers.append(label)
        if item.get('goal_map') is not None:
            goal = Marker()
            goal.header.frame_id = 'map'
            goal.header.stamp = stamp(stamp_ns)
            goal.ns, goal.id, goal.type, goal.action = 'eat_nav_goal', 0, Marker.SPHERE, Marker.ADD
            goal.scale.x = goal.scale.y = goal.scale.z = .12
            goal.color.r, goal.color.g, goal.color.b, goal.color.a = 1., .1, .2, 1.
            goal.pose.position.x, goal.pose.position.y = map(float, item['goal_map'])
            goal.pose.position.z = .12
            goal.pose.orientation.w = 1.
            bundle.markers.append(goal)
        self.candidates_pub.publish(bundle)
        path = Path()
        path.header.frame_id, path.header.stamp = 'map', stamp(stamp_ns)
        if selected in candidates:
            path.poses = [pose_message(xy, 0., 'map', stamp_ns) for xy in candidates[selected]]
        self.selected_pub.publish(path)
        self.score_pub.publish(String(data=json.dumps({'selected': selected, 'predicted_common_goal_progress_m': scores},
                                                       allow_nan=False)))

    def publish_state(self, item):
        if item.get('type') != 'state':
            raise ValueError('invalid simulator message type')
        ns = int(item['stamp_ns'])
        if self.latest_ns is not None and ns <= self.latest_ns:
            raise ValueError('simulation time did not advance')
        self.latest_ns = ns
        self.clock_pub.publish(Clock(clock=stamp(ns)))
        pose = item['pose_map']
        base = pose_message(pose[:2], pose[2], 'map', ns)
        transform = TransformStamped()
        transform.header, transform.child_frame_id = base.header, 'base_link'
        transform.transform.translation.x, transform.transform.translation.y = pose[:2]
        transform.transform.translation.z = float(item['pose_z_m'])
        transform.transform.rotation = base.pose.orientation
        self.tf_pub.sendTransform(transform)
        odom = Odometry()
        odom.header, odom.child_frame_id = base.header, 'base_link'
        odom.pose.pose = base.pose
        odom.pose.pose.position.z = float(item['pose_z_m'])
        odom.twist.twist.linear.x = float(item['velocity_body'][0])
        odom.twist.twist.angular.z = float(item['velocity_body'][2])
        self.odom_pub.publish(odom)
        self.actor_pub.publish(Float32MultiArray(data=[float(x) for x in item['actor_pre_noise']]))
        if item.get('dem') is not None:
            dem = item['dem']
            dem_ns = int(item['dem_stamp_ns'])
            if dem_ns > ns or ns - dem_ns > 250_000_000:
                raise ValueError('DEM stamp stale/future')
            self.dem_pub.publish(self._image(dem[0], '32FC1', dem_ns))
            self.mask_pub.publish(self._image(dem[1], 'mono8', dem_ns))
        if 'candidate_paths_map' in item:
            self._publish_paths(item, ns)
        self.status_pub.publish(String(data=json.dumps({'stamp_ns': ns, 'status': item['status'],
                                                        'command': item['command'],
                                                        'selected': item.get('selected')}, allow_nan=False)))

    def read_available(self):
        readable, _, _ = select.select([self.sock], [], [], 0)
        if not readable:
            return True
        chunk = self.sock.recv(1 << 20)
        if not chunk:
            return False
        self.buffer += chunk
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n', 1)
            self.publish_state(json.loads(line))
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=29553)
    args = parser.parse_args()
    sock = socket.create_connection(('127.0.0.1', args.port), timeout=15)
    sock.settimeout(None)
    rclpy.init()
    node = DemoFrontend(sock)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            if not node.read_available():
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sock.close()


if __name__ == '__main__':
    main()
