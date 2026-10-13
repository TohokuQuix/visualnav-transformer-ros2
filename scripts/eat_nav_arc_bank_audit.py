"""Offline Arc Bank geometry and frozen PureFollower initial-command audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from visualnav_transformer.eat_nav.arc_bank import (
    SCHEMA, STAGE1_ANGLES, STAGE2_ANGLES, arc_waypoints, audit_arc,
)
from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, SOURCE, compile_core


def audit_bank(build_root: Path) -> dict:
    library = compile_core(build_root)
    rows = []
    for stage, angles in ((1, STAGE1_ANGLES), (2, STAGE2_ANGLES)):
        for angle in angles:
            path = arc_waypoints(angle)
            geometry = audit_arc(path, angle)
            follower = PureFollower(path, library=library)
            try:
                command, diagnostic = follower.step_diagnostic(0., 0., 0., 0)
            finally:
                follower.close()
            rows.append({**geometry, 'stage': stage,
                         'path_xy_m': path.tolist(),
                         'path_sha256': hashlib.sha256(path.astype('<f8').tobytes()).hexdigest(),
                         'initial_command': list(command),
                         'initial_heading_error_rad': diagnostic['heading_error'],
                         'initial_turn_in_place': diagnostic['turn_in_place'],
                         'initial_goal_reached': diagnostic['goal_reached'],
                         'initial_carrot_xy_m': [diagnostic['carrot_x'], diagnostic['carrot_y']]})
    return {'schema': SCHEMA, 'follower_source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'geometry_only': True, 'runtime_feasibility_verified': False, 'candidates': rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-root', type=Path, required=True)
    args = parser.parse_args()
    result = audit_bank(args.build_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
