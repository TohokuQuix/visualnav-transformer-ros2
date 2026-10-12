"""Four-slot Path assignment contract shared by prelaunch and post-Path checks."""

from __future__ import annotations


def validate_frozen_assignment(frozen_rows: list[dict], observed_rows: list[dict],
                               *, stage: str) -> None:
    if stage not in ('prelaunch', 'post_path'):
        raise ValueError('unknown assignment validation stage')
    if len(frozen_rows) != 4 or len(observed_rows) != 4:
        raise ValueError('crossover four-branch assignment invalid')
    frozen = sorted(frozen_rows, key=lambda row: row['slot'])
    if ([row['slot'] for row in frozen] != list(range(4)) or
            [row['candidate_slot'] for row in frozen] !=
            [f'crossover_{slot}' for slot in range(4)] or
            {row['condition'] for row in frozen} != {'A'} or
            any(row['path_role'] not in ('direct', 'left_20', 'right_20') or
                not isinstance(row['path_sha256'], str) or
                len(row['path_sha256']) != 64 for row in frozen)):
        raise ValueError('frozen crossover assignment invalid')
    if stage == 'prelaunch':
        if observed_rows != frozen:
            raise ValueError('crossover prelaunch assignment differs from frozen manifest')
        return
    trial_ids = [row.get('trial_id') for row in observed_rows]
    if any(not isinstance(item, str) or not item.endswith(f'-env-{slot}')
           for slot, item in enumerate(trial_ids)) or len(set(trial_ids)) != 4:
        raise ValueError('crossover publication assignment trial/env mismatch')
    if [(row.get('path_role'), row.get('path_sha256')) for row in observed_rows] != [
            (row['path_role'], row['path_sha256']) for row in frozen]:
        raise ValueError('crossover publication assignment differs from frozen manifest')
