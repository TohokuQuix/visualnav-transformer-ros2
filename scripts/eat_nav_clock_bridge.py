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
from std_msgs.msg import String
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from eat_nav_phase_c_readiness import prepath_ready, command_delivery, current_command_stamp


class BridgeNode(Node):
    def __init__(self, branches):
        super().__init__('eat_nav_phase_c_bridge')
        self.branches = branches
        self.lock = threading.Lock()
        self.commands = [None] * branches
        self.sequence = [0] * branches
        self.command_first_at = [None] * branches
        self.status_count = [0] * branches
        self.status_first_at = [None] * branches
        self.active_path_stamp_ns = None
        self.rejected_command_count = [0] * branches
        self.command_diagnostics = [None] * branches
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.paths = [self.create_publisher(RosPath, f'/phase_c/branch_{i}/path', qos)
                      for i in range(branches)]
        self.subs = [self.create_subscription(TwistStamped, f'/phase_c/branch_{i}/cmd_vel',
                    lambda msg, i=i: self.on_command(i, msg), 10) for i in range(branches)]
        self.status_subs = [self.create_subscription(String, f'/eat_nav_follower_{i}/status',
                    lambda msg, i=i: self.on_status(i, msg), 10) for i in range(branches)]
        self.tf = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)

    def on_command(self, index, msg):
        stamp_ns = int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec)
        with self.lock:
            if not current_command_stamp(stamp_ns, self.active_path_stamp_ns):
                self.rejected_command_count[index] += 1
                if self.rejected_command_count[index] == 1:
                    print(json.dumps({'event':'rejected_prepath_or_old_command','branch':index,
                                      'wall_time':time.time(),'stamp_ns':stamp_ns,
                                      'active_path_stamp_ns':self.active_path_stamp_ns}),flush=True)
                return
            self.commands[index] = [float(msg.twist.linear.x), float(msg.twist.linear.y), float(msg.twist.angular.z)]
            self.sequence[index] += 1
            self.command_diagnostics[index] = {
                'sequence': self.sequence[index], 'stamp_ns': stamp_ns,
                'receive_system_ns': time.time_ns(), 'receive_monotonic_ns': time.monotonic_ns(),
                'bridge_node_ns': self.get_clock().now().nanoseconds,
                'command': list(self.commands[index])}
            if index == 0:
                print(json.dumps({'event':'clock_diag_command_receive', **self.command_diagnostics[index]}),flush=True)
            if self.command_first_at[index] is None:
                self.command_first_at[index] = time.time()
                print(json.dumps({'event': 'first_command', 'branch': index,
                                  'wall_time': self.command_first_at[index],
                                  'command': self.commands[index]}), flush=True)

    def on_status(self, index, msg):
        with self.lock:
            self.status_count[index] += 1
            if self.status_first_at[index] is None:
                self.status_first_at[index] = time.time()
                print(json.dumps({'event': 'first_status', 'branch': index,
                                  'wall_time': self.status_first_at[index],
                                  'status': msg.data}), flush=True)

    def tf_ready(self):
        return [bool(self.tf_buffer.can_transform('map', f'branch_{i}/chassis/body',
                                                  rclpy.time.Time())) for i in range(self.branches)]

    def publish_poses(self, poses):
        if len(poses) != self.branches:
            raise ValueError('pose count mismatch')
        now = self.get_clock().now().to_msg()
        print(json.dumps({'event':'clock_diag_tf_publish','system_ns':time.time_ns(),
            'monotonic_ns':time.monotonic_ns(),'stamp_ns':int(now.sec)*1000000000+int(now.nanosec),
            'branch_0_pose':poses[0]}),flush=True)
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

    def publish_empty_paths(self):
        now = self.get_clock().now().to_msg()
        for publisher in self.paths:
            msg = RosPath()
            msg.header.frame_id = 'map'
            msg.header.stamp = now
            publisher.publish(msg)

    def publish_paths(self, paths):
        if len(paths) != self.branches:
            raise ValueError('path count mismatch')
        now_clock = self.get_clock().now()
        now = now_clock.to_msg()
        print(json.dumps({'event':'clock_diag_path_publish','system_ns':time.time_ns(),
            'monotonic_ns':time.monotonic_ns(),'stamp_ns':now_clock.nanoseconds,
            'branch_0_point_count':len(paths[0])}),flush=True)
        with self.lock:
            self.active_path_stamp_ns = now_clock.nanoseconds
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
    def __init__(self, host, port, branches, follower, params, diagnostic_dir):
        self.host, self.port, self.branches = host, port, branches
        self.follower, self.params = follower, params
        self.diagnostic_dir = diagnostic_dir
        self.diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.follower_logs = []
        self.reset_number = 0
        rclpy.init()
        self.node = BridgeNode(branches)
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()
        self.followers = []

    def snapshot(self, poses=None):
        with self.node.lock:
            status_counts = list(self.node.status_count)
            command_sequence = list(self.node.sequence)
            command_first_at = list(self.node.command_first_at)
            rejected = list(self.node.rejected_command_count)
        return {'wall_time':time.time(),
                'followers_alive':[p.poll() is None for p in self.followers],
                'subscriber_counts':[p.get_subscription_count() for p in self.node.paths],
                'status_counts':status_counts, 'tf_ready':self.node.tf_ready(),
                'command_sequence':command_sequence,
                'first_command_wall_time':command_first_at,
                'rejected_prepath_or_old_command_count':rejected,
                'clock_now_ns':self.node.get_clock().now().nanoseconds,
                'clock_topic_publishers':self.node.count_publishers('/clock')}

    def wait_for_commands(self, baseline, poses, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.node.publish_poses(poses)
            with self.node.lock:
                seq=list(self.node.sequence)
                commands=list(self.node.commands)
            if command_delivery(baseline,seq,commands)=='fresh':
                return True
            time.sleep(.025)
        return False

    def restart_followers(self, poses, paths):
        for process in self.followers:
            process.terminate()
        for process in self.followers:
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
        for log in self.follower_logs:
            log.close()
        self.followers.clear()
        self.follower_logs.clear()
        self.reset_number += 1
        with self.node.lock:
            self.node.commands = [None] * self.branches
            self.node.sequence = [0] * self.branches
            self.node.command_first_at = [None] * self.branches
            self.node.status_count = [0] * self.branches
            self.node.status_first_at = [None] * self.branches
            self.node.rejected_command_count = [0] * self.branches
            self.node.command_diagnostics = [None] * self.branches
            self.node.active_path_stamp_ns = None
        reset_started = time.time()
        # Clear the transient-local Path from the previous run before new subscribers appear.
        self.node.publish_empty_paths()
        # TF is published first, but Path is held until all followers have
        # discovered it, observed two status-timer callbacks, and can lookup TF.
        self.node.publish_poses(poses)
        for i in range(self.branches):
            command = [str(self.follower), '--ros-args', '--params-file', str(self.params),
                       '-r', f'__node:=eat_nav_follower_{i}',
                       '-p', f'path_topic:=/phase_c/branch_{i}/path',
                       '-p', f'cmd_vel_topic:=/phase_c/branch_{i}/cmd_vel',
                       '-p', f'robot_frame:=branch_{i}/chassis/body']
            logfile=self.diagnostic_dir / f'reset-{self.reset_number:04d}-branch-{i}.log'
            stream=logfile.open('w')
            self.follower_logs.append(stream)
            self.followers.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT))
        clock_samples=[]
        deadline=time.monotonic()+8.0
        while time.monotonic()<deadline:
            self.node.publish_poses(poses)
            clock_samples.append(self.node.get_clock().now().nanoseconds)
            if len(clock_samples)>3:clock_samples=clock_samples[-3:]
            snap=self.snapshot()
            if prepath_ready(alive=snap['followers_alive'],subscribers=snap['subscriber_counts'],
                             status_counts=snap['status_counts'],tf_ready=snap['tf_ready'],
                             clock_samples_ns=clock_samples):
                break
            if not all(snap['followers_alive']):
                raise RuntimeError('infrastructure_invalid: follower process exited before Path '+repr(snap))
            time.sleep(.05)
        else:
            raise RuntimeError('infrastructure_invalid: pre-Path readiness timeout '+repr(self.snapshot()))
        prepath={**snap,'clock_samples_ns':list(clock_samples),
                 'clock_source':'ROS node clock with system-time fallback (use_sim_time=false)',
                 'clock_requirement':'two advancing node-clock intervals; /clock has no publisher',
                 'status_timer_proxy':'two follower status messages, not exact control callback count'}
        print(json.dumps({'event':'prepath_ready',**prepath}),flush=True)
        with self.node.lock:
            baseline=list(self.node.sequence)
        self.node.publish_paths(paths)
        path_first_at=time.time()
        retried=False
        if not self.wait_for_commands(baseline,poses,3.0):
            failed_first=self.snapshot()
            print(json.dumps({'event':'first_command_wait_timeout','snapshot':failed_first}),flush=True)
            # One and only one identical Path resend after rechecking TF and discovery.
            if not (all(failed_first['followers_alive']) and all(failed_first['tf_ready'])
                    and all(x>0 for x in failed_first['subscriber_counts'])):
                raise RuntimeError('infrastructure_invalid: readiness lost '+repr(failed_first))
            self.node.publish_paths(paths)
            retried=True
            if not self.wait_for_commands(baseline,poses,3.0):
                raise RuntimeError('infrastructure_invalid: fresh command timeout after one Path resend '
                                   +repr(self.snapshot()))
        postpath=self.snapshot()
        readiness={'event':'reset_complete','wall_time':time.time(),
                   'reset_started_wall_time':reset_started,'path_first_wall_time':path_first_at,
                   'prepath':prepath,'postpath':postpath,'path_resend_count':int(retried),
                   'sequence_baseline':list(postpath['command_sequence']),
                   'follower_logs':[str(self.diagnostic_dir / f'reset-{self.reset_number:04d}-branch-{i}.log')
                                    for i in range(self.branches)]}
        print(json.dumps(readiness),flush=True)
        return {'followers':[p.pid for p in self.followers], 'readiness':readiness}

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
            step_started = time.time()
            step_started_monotonic_ns = time.monotonic_ns()
            while time.monotonic() < deadline:
                self.node.publish_poses(poses)
                with self.node.lock:
                    fresh = all(self.node.sequence[i] > before[i] for i in range(self.branches))
                    commands = [None if cmd is None else list(cmd) for cmd in self.node.commands]
                    seq = list(self.node.sequence)
                if fresh:
                    return {'commands': commands, 'sequence': seq, 'fresh': True,
                            'step_started_wall_time': step_started,
                            'step_started_monotonic_ns':step_started_monotonic_ns,
                            'step_end_monotonic_ns':time.monotonic_ns(),
                            'bridge_node_clock_ns':self.node.get_clock().now().nanoseconds,
                            'command_diagnostics':list(self.node.command_diagnostics),
                            'first_command_wall_time':list(self.node.command_first_at)}
                time.sleep(0.02)
            with self.node.lock:
                statuses=list(self.node.status_count)
                first_commands=list(self.node.command_first_at)
            diagnostic={'event':'step_timeout','wall_time':time.time(),
                        'step_started_wall_time':step_started,
                        'sequence_before':before,'sequence_after':seq,
                        'first_command_wall_time':first_commands,
                        'status_counts':statuses, 'tf_ready':self.node.tf_ready(),
                        'subscriber_counts':[publisher.get_subscription_count() for publisher in self.node.paths],
                        'followers_alive':[process.poll() is None for process in self.followers]}
            print(json.dumps(diagnostic), flush=True)
            return {'commands': commands, 'sequence': seq, 'fresh': False,
                    'diagnostic':diagnostic, 'error': 'follower command timeout'}
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
    parser.add_argument('--diagnostic-dir', type=Path, default=Path('/tmp/eatnav-phase-c-follower-logs'))
    args = parser.parse_args()
    Server(args.host, args.port, args.branches, args.follower, args.params, args.diagnostic_dir).serve()


if __name__ == '__main__':
    main()
