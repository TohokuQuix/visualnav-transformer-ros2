"""Match accepted exact-stamp commands to the follower's prepublish records."""

from __future__ import annotations

import json
import math
from pathlib import Path


PREFIX = 'TF_RACE_JSON '


def find_record(path: Path, stamp_ns: int, command: list[float]) -> dict:
    """Return exactly one matching prepublish event or fail closed."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith(PREFIX):
            continue
        row = json.loads(line[len(PREFIX):])
        if row.get('phase') != 'command_pre_publish' or row.get('command_stamp_ns') != stamp_ns:
            continue
        if math.isclose(row['vx'], command[0], rel_tol=0, abs_tol=1e-6) and math.isclose(
                row['wz'], command[2], rel_tol=0, abs_tol=1e-6):
            rows.append(row)
    if len(rows) != 1:
        raise ValueError(f'exact-stamp follower diagnostic match count={len(rows)}: {path}, {stamp_ns}')
    return rows[0]
