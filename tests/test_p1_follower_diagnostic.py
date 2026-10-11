import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_follower_diagnostic import find_record


def test_exact_stamp_and_accepted_command_select_one(tmp_path: Path):
    path = tmp_path / 'log'
    rows = [
        {'phase': 'tracking', 'command_stamp_ns': 100, 'vx': 0.1, 'wz': 0.2},
        {'phase': 'command_pre_publish', 'command_stamp_ns': 100, 'vx': 0.1, 'wz': 0.2},
        {'phase': 'command_pre_publish', 'command_stamp_ns': 150, 'vx': 0.3, 'wz': 0.2},
    ]
    path.write_text(''.join('TF_RACE_JSON ' + json.dumps(row) + '\n' for row in rows))
    assert find_record(path, 100, [0.1, 0, 0.2]) == rows[1]
    with pytest.raises(ValueError, match='count=0'):
        find_record(path, 100, [0.3, 0, 0.2])
