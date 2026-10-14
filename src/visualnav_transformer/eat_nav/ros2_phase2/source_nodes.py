"""Sealed-evidence ROS replay source; real GridMap/Actor feeds require parity gate."""

from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from eat_nav_interfaces.msg import DemFrame, ActorState22

from .nodes import QOS
from .replay import load_sealed_prepath
from .wire import set_stamp


class SealedReplayNode(Node):
    """One-shot CPU integration fixture. Never publishes a robot command."""
    def __init__(self, root: Path, *, env_id=0, goal_map=(2.,0.)):
        super().__init__('eat_nav_phase2_sealed_source')
        self.sample=load_sealed_prepath(root,env_id=env_id,episode_id=1)
        self.goal_map=goal_map
        self.dem_pub=self.create_publisher(DemFrame,'/eat_nav/dem',QOS)
        self.actor_pub=self.create_publisher(ActorState22,'/eat_nav/actor22',QOS)
        self.goal_pub=self.create_publisher(PoseStamped,'/goal_pose',QOS)
        self.tf=TransformBroadcaster(self)
        self.phase=0
        self.timer=self.create_timer(.25,self._step)

    def _step(self):
        if self.phase == 0:
            goal=PoseStamped()
            goal.header.frame_id='map'
            set_stamp(goal.header.stamp,self.get_clock().now().nanoseconds)
            goal.pose.position.x=float(self.goal_map[0])
            goal.pose.position.y=float(self.goal_map[1])
            goal.pose.orientation.w=1.
            self.goal_pub.publish(goal)
        elif 1 <= self.phase <= 4:
            msg=self.sample['dem_messages'][self.phase-1]
            # The stored 250/500/750/1000 ms are logical capture stamps.
            # Shift them into this local ROS clock domain without altering arrays.
            if self.phase == 1:
                self.anchor_ns=self.get_clock().now().nanoseconds+750_000_000
            stamp=self.anchor_ns-(4-self.phase)*250_000_000
            set_stamp(msg.header.stamp,stamp)
            set_stamp(msg.sensor_stamp,stamp)
            self.dem_pub.publish(msg)
            if self.phase == 4:
                transform=TransformStamped()
                transform.header.frame_id='map'
                transform.child_frame_id='chassis/body'
                set_stamp(transform.header.stamp,stamp)
                root=self.sample['root_tile_local']
                transform.transform.translation.x=float(root[0])
                transform.transform.translation.y=float(root[1])
                transform.transform.translation.z=float(root[2])
                transform.transform.rotation.w=float(root[3])
                transform.transform.rotation.x=float(root[4])
                transform.transform.rotation.y=float(root[5])
                transform.transform.rotation.z=float(root[6])
                self.tf.sendTransform(transform)
        elif self.phase == 5:
            actor=self.sample['actor_message']
            set_stamp(actor.header.stamp,self.anchor_ns)
            for field in ('imu_stamp','joint_stamp','command_stamp'):
                set_stamp(getattr(actor,field),self.anchor_ns)
            self.actor_pub.publish(actor)
            self.timer.cancel()
        self.phase+=1


def main(root, env_id=0):
    rclpy.init()
    node=SealedReplayNode(Path(root),env_id=env_id)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()
