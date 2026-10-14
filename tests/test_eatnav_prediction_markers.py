"""RViz candidate/forecast correspondence, without a live ROS graph."""

import numpy as np
import pytest
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
