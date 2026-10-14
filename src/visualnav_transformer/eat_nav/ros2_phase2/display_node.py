"""Independent RViz-only display; it cannot publish selections or commands."""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from eat_nav_interfaces.msg import CandidateBank, PathSelection, NavigationStatus

from .nodes import QOS, BANK_QOS
from .wire import bank_from_message, PATHS, CANDIDATE_IDS, WIRE_SHA


class DisplayNode(Node):
    def __init__(self):
        super().__init__('eat_nav_phase2_display')
        self.markers=self.create_publisher(MarkerArray,'/eat_nav/display/paths',QOS)
        self.create_subscription(CandidateBank,'/eat_nav/bank',self._bank,BANK_QOS)
        self.create_subscription(PathSelection,'/eat_nav/mock/selection',self._selected,QOS)
        self.create_subscription(NavigationStatus,'/eat_nav/status',self._status,QOS)
        self.last_status=None

    def _paths(self, paths, stamp, selected=-1):
        out=MarkerArray()
        for index,path in enumerate(paths):
            marker=Marker()
            marker.header.frame_id='chassis/body'
            marker.header.stamp=stamp
            marker.ns='eat_nav_phase2_bank'
            marker.id=index
            marker.type=Marker.LINE_STRIP
            marker.action=Marker.ADD
            marker.scale.x=.015 if selected < 0 else (.04 if index==selected else .008)
            marker.color.a=1.
            marker.color.g=1. if index==selected else .3
            marker.color.r=0. if index==selected else 1.
            marker.points=[Point(x=float(x),y=float(y),z=0.) for x,y in path]
            out.markers.append(marker)
        self.markers.publish(out)

    def _bank(self,msg):
        try:
            values,_=bank_from_message(msg)
            self._paths(values['paths'],msg.header.stamp)
        except ValueError as error:
            self.get_logger().warn(f'bank display rejected: {error}')

    def _selected(self,msg):
        if msg.wire_path_sha256 != WIRE_SHA or msg.candidate_id not in CANDIDATE_IDS:
            return
        self._paths(PATHS,msg.header.stamp,CANDIDATE_IDS.index(msg.candidate_id))

    def _status(self,msg):
        self.last_status=(msg.state,msg.reason)
        if not msg.command_gate_open:
            clear=MarkerArray()
            marker=Marker(); marker.action=Marker.DELETEALL
            marker.header.frame_id='chassis/body'
            marker.header.stamp=msg.header.stamp
            clear.markers.append(marker)
            self.markers.publish(clear)


def main():
    rclpy.init(); node=DisplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()
