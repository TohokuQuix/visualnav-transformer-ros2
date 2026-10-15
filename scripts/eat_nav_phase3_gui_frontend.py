#!/usr/bin/env python3
"""Simulation-only ROS transport between measured Isaac state and Phase 2 nodes.

This node has no model, Path selector, or follower. It publishes the exact
measured inputs, forwards the selected frozen Path, and relays the supervisor's
gated command to Isaac. It never publishes a production robot command.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import socket
import struct
import time

import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from tf2_ros import TransformBroadcaster

from eat_nav_interfaces.msg import ActorState22, DemFrame, NavigationStatus, PathSelection
from visualnav_transformer.eat_nav.ros2_phase2.nodes import QOS
from visualnav_transformer.eat_nav.ros2_phase2.wire import (
    actor_to_message, dem_to_message, set_stamp, stamp_ns,
)
from visualnav_transformer.eat_nav.ros2_phase3_gui_contract import validate_selection, validate_source_frame


def _stamp(ns):
    result = Time()
    set_stamp(result, int(ns))
    return result


class Phase3GuiFrontend(Node):
    def __init__(self, sock, *, auto_goal=None):
        super().__init__('eat_nav_phase3_gui_frontend')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.sock = sock
        self.sock.setblocking(False)
        self.buffer = b''
        self.latest_ns = None
        self.latest_pose = None
        self.episode_id = 0
        self.goal_sent = False
        self.auto_goal = auto_goal
        self.pending_command = None
        self.pending_final = None
        self.last_selection = None
        self.status = None
        self.source_count = 0
        self.selection_count = 0
        self.gated_count = 0
        self.clock_pub = self.create_publisher(Clock, '/clock', 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', QOS)
        self.dem_image_pub = self.create_publisher(Image, '/eat_nav/dem/height', QOS)
        self.selected_path_pub = self.create_publisher(Path, '/eat_nav/selected_path', QOS)
        self.dem_pub = self.create_publisher(DemFrame, '/eat_nav/dem', QOS)
        self.actor_pub = self.create_publisher(ActorState22, '/eat_nav/actor22', QOS)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', QOS)
        self.cancel_pub = self.create_publisher(Empty, '/eat_nav/cancel', QOS)
        self.raw_cmd_pub = self.create_publisher(TwistStamped, '/eat_nav/mock/follower_cmd', QOS)
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(PoseStamped, '/goal_pose', self._goal, QOS)
        self.create_subscription(Empty, '/eat_nav/cancel', self._cancel, QOS)
        self.create_subscription(PathSelection, '/eat_nav/mock/selection', self._selection, QOS)
        self.create_subscription(NavigationStatus, '/eat_nav/status', self._status, QOS)
        self.create_subscription(TwistStamped, '/eat_nav/mock/final_cmd', self._final_command, QOS)

    def _send(self, value):
        self.sock.sendall((json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n').encode())

    def _goal(self, msg):
        if self.latest_ns is None or msg.header.frame_id != 'map':
            return
        stamp = stamp_ns(msg.header.stamp)
        xy = [float(msg.pose.position.x), float(msg.pose.position.y)]
        if not 0 <= self.latest_ns - stamp <= 300_000_000 or \
                not all(math.isfinite(x) for x in xy):
            return
        self._send({'type': 'goal', 'stamp_ns': stamp, 'frame': 'map', 'xy': xy})

    def _cancel(self, _):
        self.last_selection = None
        self._send({'type': 'cancel', 'stamp_ns': self.latest_ns})

    def _selection(self, msg):
        if self.latest_ns is None:
            return
        value = {'type': 'selection', 'stamp_ns': stamp_ns(msg.header.stamp),
                 'generation': int(msg.generation), 'candidate_id': msg.candidate_id,
                 'wire_path_sha256': msg.wire_path_sha256,
                 'body_xy_m': [float(x) for x in msg.body_xy_m],
                 'expires_ns': stamp_ns(msg.expires_at)}
        try:
            validate_selection(value, self.latest_ns)
        except ValueError as error:
            self.get_logger().error(f'ROS selection rejected: {error}')
            return
        if value['stamp_ns'] != self.latest_ns or self.latest_pose is None:
            self.get_logger().error('ROS selection lacks exact-stamp visualization pose')
            return
        path = Path(); path.header.frame_id='map'; path.header.stamp=msg.header.stamp
        x,y,yaw=self.latest_pose
        c,s=math.cos(yaw),math.sin(yaw)
        for bx,by in np.asarray(value['body_xy_m'],np.float32).reshape(8,2):
            point=PoseStamped(); point.header=path.header
            point.pose.position.x=float(x+c*bx-s*by)
            point.pose.position.y=float(y+s*bx+c*by)
            point.pose.position.z=.1
            point.pose.orientation.w=1.
            path.poses.append(point)
        self.selected_path_pub.publish(path)
        self.last_selection = value
        self.selection_count += 1
        self._send(value)

    def _status(self, msg):
        self.status = (int(msg.generation), msg.state, bool(msg.command_gate_open),
                       stamp_ns(msg.header.stamp))
        if not msg.command_gate_open:
            self.last_selection = None
        self._finish_command_if_ready()

    def _final_command(self, msg):
        request = self.pending_command
        if request is None or stamp_ns(msg.header.stamp) != request['stamp_ns']:
            return
        self.pending_final = (float(msg.twist.linear.x), float(msg.twist.angular.z))
        self._finish_command_if_ready()

    def _finish_command_if_ready(self):
        request = self.pending_command
        if request is None or self.pending_final is None or self.status is None:
            return
        actual = self.pending_final
        wanted = tuple(request['command'])
        status_open = self.status[2] and \
            self.status[0] == request['generation'] and \
            self.status[3] == request['stamp_ns']
        if status_open and all(abs(a-b) <= 1e-6 for a,b in zip(actual,wanted)):
            self._send({'type': 'gated_command', 'stamp_ns': request['stamp_ns'],
                        'generation': request['generation'], 'gate_open': True,
                        'command': list(actual)})
            self.gated_count += 1
            self.pending_command = None
            self.pending_final = None
        elif self.status[3] == request['stamp_ns'] and \
                not self.status[2] and actual == (0., 0.):
            self._send({'type': 'gated_command', 'stamp_ns': request['stamp_ns'],
                        'generation': request['generation'], 'gate_open': False,
                        'command': [0., 0.]})
            self.pending_command = None
            self.pending_final = None

    def publish_source(self, item):
        validate_source_frame(item)
        ns = item['stamp_ns']
        if self.latest_ns is not None and ns <= self.latest_ns:
            raise ValueError('source clock nonmonotonic')
        self.latest_ns = ns
        self.latest_pose = tuple(item['pose_map'])
        self.episode_id = item['episode_id']
        self.clock_pub.publish(Clock(clock=_stamp(ns)))
        x,y,yaw = item['pose_map']
        tr = TransformStamped()
        tr.header.frame_id = 'map'; tr.header.stamp = _stamp(ns)
        tr.child_frame_id = 'chassis/body'
        tr.transform.translation.x = float(x)
        tr.transform.translation.y = float(y)
        tr.transform.translation.z = float(item['pose_z_m'])
        tr.transform.rotation.z = math.sin(yaw/2)
        tr.transform.rotation.w = math.cos(yaw/2)
        self.tf_pub.sendTransform(tr)
        odom = Odometry(); odom.header = tr.header
        odom.child_frame_id = 'chassis/body'
        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(item['pose_z_m'])
        odom.pose.pose.orientation = tr.transform.rotation
        self.odom_pub.publish(odom)
        if item['dem'] is not None:
            self.dem_pub.publish(dem_to_message(item['dem'], ns,
                item['dem_sensor_update_ns'], self.episode_id,
                source_id='isaac_phase3_measured'))
            image=Image(); image.header.frame_id='chassis/body'
            image.header.stamp=_stamp(ns)
            image.height=image.width=20; image.encoding='32FC1'; image.step=80
            image.data=struct.pack('<400f', *np.asarray(item['dem'][0],np.float32).reshape(-1))
            self.dem_image_pub.publish(image)
        self.actor_pub.publish(actor_to_message(item['actor_pre_noise'], ns,
            self.episode_id, {key: ns for key in ('imu','joint','command')}))
        self.source_count += 1
        if self.auto_goal is not None and not self.goal_sent and ns >= 1_000_000_000:
            goal = PoseStamped(); goal.header.frame_id='map'; goal.header.stamp=_stamp(ns)
            goal.pose.position.x = float(x + self.auto_goal[0])
            goal.pose.position.y = float(y + self.auto_goal[1])
            goal.pose.orientation.w = 1.
            self.goal_pub.publish(goal)
            self.goal_sent = True

    def _request_command(self, item):
        if self.pending_command is not None or item['stamp_ns'] != self.latest_ns:
            raise ValueError('duplicate/out-of-order command request')
        self.pending_command = item
        self.pending_final = None
        msg = TwistStamped(); msg.header.frame_id='chassis/body'
        msg.header.stamp = _stamp(item['stamp_ns'])
        msg.twist.linear.x = float(item['command'][0])
        msg.twist.angular.z = float(item['command'][1])
        self.raw_cmd_pub.publish(msg)

    def read_available(self):
        if not select.select([self.sock], [], [], 0)[0]:
            return True
        chunk = self.sock.recv(1 << 20)
        if not chunk:
            return False
        self.buffer += chunk
        while b'\n' in self.buffer:
            line, self.buffer = self.buffer.split(b'\n',1)
            item = json.loads(line)
            if item.get('type') == 'source':
                self.publish_source(item)
            elif item.get('type') == 'command_request':
                self._request_command(item)
            elif item.get('type') == 'reset':
                self.last_selection = None
                # A reset invalidates the supervisor's held Path as well.
                self.cancel_pub.publish(Empty())
            else:
                raise ValueError('invalid simulator IPC type')
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--auto-goal-dx', type=float)
    parser.add_argument('--auto-goal-dy', type=float)
    args = parser.parse_args()
    if (args.auto_goal_dx is None) != (args.auto_goal_dy is None):
        parser.error('auto goal requires both coordinates')
    sock = socket.create_connection(('127.0.0.1',args.port),timeout=30)
    auto = None if args.auto_goal_dx is None else (args.auto_goal_dx,args.auto_goal_dy)
    rclpy.init()
    node = Phase3GuiFrontend(sock, auto_goal=auto)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.005)
            if not node.read_available():
                break
    finally:
        node.get_logger().info(f'sources={node.source_count} selections={node.selection_count} gated={node.gated_count}')
        node.destroy_node(); rclpy.shutdown(); sock.close()


if __name__ == '__main__':
    main()
