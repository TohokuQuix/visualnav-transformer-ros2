"""Independent mock actuator watchdog. Never targets a real robot topic."""

import math
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from eat_nav_interfaces.msg import NavigationStatus, PathSelection

from .nodes import QOS
from .wire import set_stamp, stamp_ns, CANDIDATE_IDS, WIRE_SHA, PATHS


class MockRobotInterface:
    HEARTBEAT_MAX_NS=100_000_000
    COMMAND_MAX_NS=50_000_000

    def __init__(self):
        self.generation=0
        self.gate_open=False
        self.status_stamp_ns=None
        self.held_path_generation=None
        self.command_value=(0.,0.)
        self.command_stamp_ns=None

    def status(self,generation,gate_open,stamp_ns):
        if generation < self.generation or \
                (self.status_stamp_ns is not None and stamp_ns < self.status_stamp_ns):
            raise ValueError('stale mock supervisor status')
        if generation > self.generation or not gate_open:
            self.held_path_generation=None
            self.command_value=(0.,0.)
            self.command_stamp_ns=None
        self.generation=generation
        self.gate_open=bool(gate_open)
        self.status_stamp_ns=stamp_ns

    def path(self,generation,stamp_ns):
        if not self.gate_open or generation != self.generation or \
                self.status_stamp_ns is None or stamp_ns < self.status_stamp_ns-500_000_000:
            raise ValueError('old/inactive mock Path')
        self.held_path_generation=generation

    def command(self,value,stamp_ns):
        command=tuple(float(x) for x in value)
        if len(command)!=2 or not all(map(math.isfinite,command)):
            self.command_value=(0.,0.); self.command_stamp_ns=None
            return
        self.command_value=command
        self.command_stamp_ns=stamp_ns

    def tick(self,now_ns):
        if not self.gate_open or self.held_path_generation!=self.generation or \
                self.status_stamp_ns is None or \
                not 0 <= now_ns-self.status_stamp_ns <= self.HEARTBEAT_MAX_NS or \
                self.command_stamp_ns is None or \
                not 0 <= now_ns-self.command_stamp_ns <= self.COMMAND_MAX_NS:
            return (0.,0.)
        return self.command_value


class MockRobotNode(Node):
    def __init__(self):
        super().__init__('eat_nav_phase2_mock_robot')
        self.interface=MockRobotInterface()
        self.pending_path=None
        self.pub=self.create_publisher(TwistStamped,'/eat_nav/mock/applied_cmd',QOS)
        self.create_subscription(NavigationStatus,'/eat_nav/status',self._status,QOS)
        self.create_subscription(PathSelection,'/eat_nav/mock/selection',self._path,QOS)
        self.create_subscription(TwistStamped,'/eat_nav/mock/final_cmd',self._command,QOS)
        self.create_timer(.05,self._tick)

    def _status(self,msg):
        try:
            self.interface.status(msg.generation,msg.command_gate_open,
                                  stamp_ns(msg.header.stamp))
            if self.pending_path and msg.command_gate_open and \
                    msg.generation==self.pending_path.generation:
                self.interface.path(self.pending_path.generation,
                                    stamp_ns(self.pending_path.header.stamp))
                self.pending_path=None
            elif not msg.command_gate_open:
                self.pending_path=None
        except ValueError:
            pass

    def _path(self,msg):
        if msg.wire_path_sha256!=WIRE_SHA or msg.candidate_id not in CANDIDATE_IDS:
            return
        index=CANDIDATE_IDS.index(msg.candidate_id)
        points=np.asarray(msg.body_xy_m,dtype=np.float32)
        stamp=stamp_ns(msg.header.stamp)
        expires=stamp_ns(msg.expires_at)
        now=self.get_clock().now().nanoseconds
        if points.shape!=(16,) or not np.array_equal(points,PATHS[index].reshape(-1)) or \
                msg.direction != (1 if index<7 else -1) or \
                not 0 <= now-stamp <= 300_000_000 or \
                not 0 < expires-now <= 500_000_000:
            return
        try:
            self.interface.path(msg.generation,stamp)
        except ValueError:
            if msg.generation>=self.interface.generation:
                self.pending_path=msg

    def _command(self,msg):
        if msg.header.frame_id!='chassis/body': return
        self.interface.command((msg.twist.linear.x,msg.twist.angular.z),
                               stamp_ns(msg.header.stamp))

    def _tick(self):
        now=self.get_clock().now().nanoseconds
        output=self.interface.tick(now)
        msg=TwistStamped();msg.header.frame_id='chassis/body'
        set_stamp(msg.header.stamp,now)
        msg.twist.linear.x=output[0];msg.twist.angular.z=output[1]
        self.pub.publish(msg)


def main():
    rclpy.init();node=MockRobotNode()
    try:rclpy.spin(node)
    finally:node.destroy_node();rclpy.shutdown()
