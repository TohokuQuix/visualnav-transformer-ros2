"""RViz candidate/forecast correspondence, without a live ROS graph."""

import numpy as np
import pytest
import math
from types import SimpleNamespace

from eat_nav_interfaces.msg import CandidateEvaluation, NavigationStatus
from visualnav_transformer.eat_nav.ros2_phase2.display_node import (
    DisplayNode, candidate_color, path_markers, prediction_markers,
)
from visualnav_transformer.eat_nav.ros2_phase2.wire import (
    BANK_SHA, CANDIDATE_IDS, PATHS, set_stamp,
)


def evaluation():
    message = CandidateEvaluation()
    message.header.frame_id = 'chassis/body'
    set_stamp(message.header.stamp, 1_000_000_000)
    message.bank_source_sha256 = BANK_SHA
    message.predicted_body_xy_m = np.arange(28, dtype=np.float32).tolist()
    message.scores_m = [0.] * 14
    message.mask = [True] * 14
    message.selected_index = 2
    return message


def rgb(marker):
    return marker.color.r, marker.color.g, marker.color.b


def test_every_path_and_predicted_position_share_stable_candidate_color():
    message = evaluation()
    paths = path_markers(PATHS, message.header.stamp, selected=2)
    predicted = prediction_markers(message)
    assert len(paths.markers) == 14
    assert len(predicted.markers) == 28
    assert len({candidate_color(i) for i in range(14)}) == 14
    for index, line in enumerate(paths.markers):
        sphere, label = predicted.markers[2 * index:2 * index + 2]
        assert rgb(line) == rgb(sphere) == rgb(label) == candidate_color(index)
        assert sphere.pose.position.x == message.predicted_body_xy_m[2 * index]
        assert sphere.pose.position.y == message.predicted_body_xy_m[2 * index + 1]
        assert sphere.header.frame_id == line.header.frame_id == 'chassis/body'
        assert sphere.header.stamp == line.header.stamp == message.header.stamp
        assert label.text == CANDIDATE_IDS[index]
    assert paths.markers[2].scale.x > paths.markers[1].scale.x
    assert predicted.markers[4].scale.x > predicted.markers[2].scale.x


@pytest.mark.parametrize('mutation', [
    lambda msg: setattr(msg, 'bank_source_sha256', 'wrong'),
    lambda msg: setattr(msg.header, 'frame_id', 'map'),
    lambda msg: setattr(msg, 'predicted_body_xy_m', [1.] * 27),
    lambda msg: setattr(msg, 'predicted_body_xy_m', [float('nan')] + [0.] * 27),
    lambda msg: setattr(msg, 'mask', [True] * 13),
])
def test_invalid_evaluation_is_rejected(mutation):
    message = evaluation()
    mutation(message)
    with pytest.raises(ValueError):
        prediction_markers(message)


def test_display_callback_publishes_predictions_and_clears_both_layers():
    path_messages, predicted_messages = [], []
    display = SimpleNamespace(
        markers=SimpleNamespace(publish=path_messages.append),
        predicted=SimpleNamespace(publish=predicted_messages.append),
        last_status=None,
    )
    DisplayNode._evaluation(display, evaluation())
    assert len(predicted_messages[-1].markers) == 28
    status = NavigationStatus()
    status.command_gate_open = False
    DisplayNode._status(display, status)
    assert path_messages[-1].markers[0].action == predicted_messages[-1].markers[0].action


def test_yaw_head_orients_matching_color_arrows_at_predicted_xy():
    message = evaluation()
    message.predicted_yaw_valid = True
    yaw = [0.] * 14
    yaw[2] = math.pi / 2
    message.predicted_delta_yaw_rad = yaw
    markers = prediction_markers(message).markers
    arrow = markers[4]
    assert arrow.type == arrow.ARROW
    assert rgb(arrow) == candidate_color(2)
    assert arrow.pose.orientation.z == pytest.approx(math.sin(math.pi / 4))
    assert arrow.pose.orientation.w == pytest.approx(math.cos(math.pi / 4))
    assert arrow.pose.position.x == message.predicted_body_xy_m[4]
    assert arrow.pose.position.y == message.predicted_body_xy_m[5]
    assert all(markers[2 * i].type == arrow.ARROW for i in range(14))


def test_nonfinite_predicted_yaw_is_rejected():
    message = evaluation()
    message.predicted_yaw_valid = True
    message.predicted_delta_yaw_rad = [float('nan')] + [0.] * 13
    with pytest.raises(ValueError):
        prediction_markers(message)
