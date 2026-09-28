"""Host ROS2 bridge that runs the actual simple_trajectory_follower per branch.

Protocol: one newline-delimited JSON request/response on localhost TCP. The Isaac
client supplies map poses and Paths; this process publishes TF/Path, waits for
fresh TwistStamped from each actual follower, and returns those commands.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped, TwistStamped, PoseStamped
from nav_msgs.msg import Path as RosPath
from tf2_ros import TransformBroadcaster


class BridgeNode(Node):
    def __init__(self, branches):
        super().__init__('eat_nav_phase_c_bridge')
        self.branches = branches
        self.lock = threading.Lock()
        self.commands = [None] * branches
        self.sequence = [0] * branches
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.paths = [self.create_publisher(RosPath, f'/phase_c/branch_{i}/path', qos)
                      for i in range(branches)]
        self.subs = [self.create_subscription(TwistStamped, f'/phase_c/branch_{i}/cmd_vel',
                    lambda msg, i=i: self.on_command(i, msg), 10) for i in range(branches)]
        self.tf = TransformBroadcaster(self)

    def on_command(self, index, msg):
        with self.lock:
            self.commands[index] = [float(msg.twist.linear.x), float(msg.twist.linear.y), float(msg.twist.angular.z)]
            self.sequence[index] += 1

    def publish_poses(self, poses):
        if len(poses) != self.branches:
            raise ValueError('pose count mismatch')
        now = self.get_clock().now().to_msg()
        transforms = []
        for i, pose in enumerate(poses):
            x, y, yaw = [float(v) for v in pose]
            msg = TransformStamped()
            msg.header.stamp = now
            msg.header.frame_id = 'map'
            msg.child_frame_id = f'branch_{i}/chassis/body'
            msg.transform.translation.x = x
            msg.transform.translation.y = y
            msg.transform.rotation.z = math.sin(yaw/2)
            msg.transform.rotation.w = math.cos(yaw/2)
            transforms.append(msg)
        self.tf.sendTransform(transforms)

    def publish_paths(self, paths):
        if len(paths) != self.branches:
            raise ValueError('path count mismatch')
        now = self.get_clock().now().to_msg()
        for i, xy in enumerate(paths):
            if len(xy) < 2:
                raise ValueError('Path needs at least two poses')
            msg = RosPath()
            msg.header.frame_id = 'map'; msg.header.stamp = now
            for x, y in xy:
                pose = PoseStamped()
                pose.header = msg.header
                pose.pose.position.x = float(x); pose.pose.position.y = float(y)
                pose.pose.orientation.w = 1.0
                msg.poses.append(pose)
            self.paths[i].publish(msg)


class Server:
    def __init__(self, host, port, branches, follower, params):
        self.host, self.port, self.branches = host, port, branches
        self.follower, self.params = follower, params
        rclpy.init()
        self.node = BridgeNode(branches)
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()
        self.followers = []

    def restart_followers(self, poses, paths):
        for process in self.followers:
            process.terminate()
        for process in self.followers:
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        self.followers.clear()
        with self.node.lock:
            self.node.commands = [None] * self.branches
            self.node.sequence = [0] * self.branches
        # Replace transient-local Paths before a fresh controller subscribes.
        self.node.publish_poses(poses)
        self.node.publish_paths(paths)
        for i in range(self.branches):
            command = [str(self.follower), '--ros-args', '--params-file', str(self.params),
                       '-r', f'__node:=eat_nav_follower_{i}',
                       '-p', f'path_topic:=/phase_c/branch_{i}/path',
                       '-p', f'cmd_vel_topic:=/phase_c/branch_{i}/cmd_vel',
                       '-p', f'robot_frame:=branch_{i}/chassis/body']
            self.followers.append(subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                                   stderr=subprocess.DEVNULL))
        time.sleep(0.5)
        return {'followers': [p.pid for p in self.followers]}

    def handle(self, request):
        action = request['action']
        if action == 'reset':
            return self.restart_followers(request['poses'], request['paths'])
        if action == 'publish':
            poses, paths = request['poses'], request['paths']
            self.node.publish_poses(poses)
            time.sleep(0.08)
            self.node.publish_paths(paths)
            return {'published': len(paths)}
        if action == 'step':
            poses = request['poses']
            with self.node.lock:
                before = list(self.node.sequence)
            deadline = time.monotonic() + float(request.get('timeout_sec', 1.0))
            while time.monotonic() < deadline:
                self.node.publish_poses(poses)
                with self.node.lock:
                    fresh = all(self.node.sequence[i] > before[i] for i in range(self.branches))
                    commands = [None if cmd is None else list(cmd) for cmd in self.node.commands]
                    seq = list(self.node.sequence)
                if fresh:
                    return {'commands': commands, 'sequence': seq, 'fresh': True}
                time.sleep(0.02)
            return {'commands': commands, 'sequence': seq, 'fresh': False, 'error': 'follower command timeout'}
        if action == 'status':
            with self.node.lock:
                return {'commands': self.node.commands, 'sequence': self.node.sequence,
                        'followers_alive': [p.poll() is None for p in self.followers]}
        raise ValueError(f'unknown action {action}')

    def serve(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port)); listener.listen(1)
            print(json.dumps({'stage': 'listening', 'host': self.host, 'port': self.port}), flush=True)
            while True:
                conn, _ = listener.accept()
                with conn, conn.makefile('rwb') as stream:
                    for line in stream:
                        try:
                            response = {'ok': True, **self.handle(json.loads(line))}
                        except Exception as exc:
                            response = {'ok': False, 'error': repr(exc)}
                        stream.write((json.dumps(response)+'\n').encode()); stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=47819)
    parser.add_argument('--branches', type=int, default=8)
    parser.add_argument('--follower', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    args = parser.parse_args()
    Server(args.host, args.port, args.branches, args.follower, args.params).serve()


if __name__ == '__main__':
    main()
