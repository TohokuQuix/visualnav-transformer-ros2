"""Detached final aggregation, gated on the separate continuation completion marker."""
from __future__ import annotations

import json
import time

import eat_nav_goal_subgoal_v2_analyze as analyze
import eat_nav_goal_subgoal_v2_run as v


def main():
    deadline = v.read(v.OUT / 'v2-continuation-start.json')['start_unix'] + 7200 + 180
    while time.time() <= deadline:
        done = v.OUT / 'v2-continuation.done.json'
        if done.exists():
            if v.read(done)['status'] != 'pass':
                raise ValueError('continuation completion marker invalid')
            analyze.main()
            v.atomic(v.OUT / 'v2-continuation-analysis.done.json', {
                'status': 'pass', 'continuation_sha256': v.sha(done),
                'decision_sha256': v.sha(v.OUT / 'final-decision-v2-schedule-storage.json'),
                'recovery_marker_sha256': v.sha(v.OUT / 'runs/v2-train-001-near_left_lateral-seed-1002/'
                                                   'recovered_complete_after_monitor_cleanup_race.json')})
            return
        status = v.read(v.OUT / 'status.json')
        if status.get('stage') == 'V2_CONTINUATION_STOPPED':
            v.atomic(v.OUT / 'v2-continuation-analysis.skipped.json', {
                'reason': 'continuation stopped before full schedule',
                'status': status})
            return
        time.sleep(20)
    v.atomic(v.OUT / 'v2-continuation-analysis.skipped.json', {
        'reason': 'additional two-hour continuation deadline exceeded'})


if __name__ == '__main__':
    main()
