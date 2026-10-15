"""Independent RViz-only display; it cannot publish selections or commands."""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from eat_nav_interfaces.msg import (CandidateBank, CandidateEvaluation,
                                    PathSelection, NavigationStatus)

from .nodes import QOS, BANK_QOS
from .wire import bank_from_message, PATHS, CANDIDATE_IDS, BANK_SHA, WIRE_SHA


# Candidate-index palette shared by Path lines and predicted-position markers.
_COLORS = (
    (0.95, 0.25, 0.25), (0.20, 0.60, 1.00), (0.25, 0.85, 0.35),
    (0.90, 0.45, 0.95), (1.00, 0.60, 0.15), (0.15, 0.85, 0.85),
    (0.95, 0.85, 0.20), (0.55, 0.40, 1.00), (0.75, 0.50, 0.25),
    (0.95, 0.45, 0.65), (0.45, 0.90, 0.65), (0.75, 0.75, 0.25),
    (0.45, 0.75, 0.95), (0.95, 0.75, 0.55),
)


def candidate_color(index):
    return _COLORS[index]


def _color(marker, index):
    marker.color.r, marker.color.g, marker.color.b = candidate_color(index)
    marker.color.a = 1.0


def path_markers(paths, stamp, selected=-1):
    if len(paths) != len(CANDIDATE_IDS):
        raise ValueError('Path count differs from candidate bank')
    out = MarkerArray()
    for index, path in enumerate(paths):
        marker = Marker()
        marker.header.frame_id = 'chassis/body'
        marker.header.stamp = stamp
        marker.ns = 'eat_nav_phase2_bank'
        marker.id = index
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = .04 if index == selected else .018
        _color(marker, index)
        marker.points = [Point(x=float(x), y=float(y), z=0.) for x, y in path]
        out.markers.append(marker)
    return out


def prediction_markers(msg):
    values = np.asarray(msg.predicted_body_xy_m, np.float32)
    yaw = np.asarray(msg.predicted_delta_yaw_rad, np.float32)
    if msg.header.frame_id != 'chassis/body' or msg.bank_source_sha256 != BANK_SHA or \
            values.shape != (28,) or not np.isfinite(values).all() or \
            yaw.shape != (14,) or (msg.predicted_yaw_valid and not np.isfinite(yaw).all()) or \
            len(msg.mask) != 14 or not 0 <= msg.selected_index < 14:
        raise ValueError('candidate prediction display contract')
    points = values.reshape(14, 2)
    out = MarkerArray()
    for index, (x, y) in enumerate(points):
        if not msg.mask[index]:
            continue
        sphere = Marker()
        sphere.header = msg.header
        sphere.ns = 'eat_nav_predicted_xy'
        sphere.id = index
        sphere.type = Marker.ARROW if msg.predicted_yaw_valid else Marker.SPHERE
        sphere.action = Marker.ADD
        if msg.predicted_yaw_valid:
            sphere.scale.x = .18 if index == msg.selected_index else .13
            sphere.scale.y = .03
            sphere.scale.z = .05
            sphere.pose.orientation.z = math.sin(float(yaw[index]) / 2)
            sphere.pose.orientation.w = math.cos(float(yaw[index]) / 2)
        else:
            size = .09 if index == msg.selected_index else .055
            sphere.scale.x = sphere.scale.y = sphere.scale.z = size
            sphere.pose.orientation.w = 1.0
        sphere.pose.position.x = float(x)
        sphere.pose.position.y = float(y)
        sphere.pose.position.z = .06
        _color(sphere, index)
        out.markers.append(sphere)

        label = Marker()
        label.header = msg.header
        label.ns = 'eat_nav_predicted_ids'
        label.id = index
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = float(x)
        label.pose.position.y = float(y)
        label.pose.position.z = .18
        label.pose.orientation.w = 1.0
        label.scale.z = .08
        label.text = CANDIDATE_IDS[index]
        _color(label, index)
        out.markers.append(label)
    return out


class DisplayNode(Node):
    def __init__(self):
        super().__init__('eat_nav_phase2_display')
        self.markers=self.create_publisher(MarkerArray,'/eat_nav/display/paths',QOS)
        self.predicted=self.create_publisher(MarkerArray,
                                             '/eat_nav/display/predicted_positions',QOS)
        self.create_subscription(CandidateBank,'/eat_nav/bank',self._bank,BANK_QOS)
        self.create_subscription(CandidateEvaluation,'/eat_nav/evaluation',
                                 self._evaluation,QOS)
        self.create_subscription(PathSelection,'/eat_nav/mock/selection',self._selected,QOS)
        self.create_subscription(NavigationStatus,'/eat_nav/status',self._status,QOS)
        self.last_status=None

    def _paths(self, paths, stamp, selected=-1):
        self.markers.publish(path_markers(paths,stamp,selected))

    def _evaluation(self,msg):
        try:
            self.predicted.publish(prediction_markers(msg))
        except ValueError as error:
            self.get_logger().warn(f'prediction display rejected: {error}')

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
            self.predicted.publish(clear)


def main():
    rclpy.init(); node=DisplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node(); rclpy.shutdown()
