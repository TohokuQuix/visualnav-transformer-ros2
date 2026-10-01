"""120-step readiness-002 parity after admitting frozen v2 state IDs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import eat_nav_goal_subgoal_timing_parity as parity
from eat_nav_goal_subgoal_v2_select import REPO, OUT, sha, atomic

PARITY_OUT = OUT / 'v6-parity'


def setup():
    source = REPO / 'scripts/eat_nav_goal_subgoal_sim.py'
    parity.REPO = REPO
    parity.OUT = PARITY_OUT
    parity.PARITY = PARITY_OUT / 'parity'
    parity.REMOTE = str(PARITY_OUT / 'sim-parity.py')
    parity.fresh.REPO = REPO
    if json.loads((OUT / 'v5.done.json').read_text())['decision'] != 'pass':
        raise ValueError('V5 runtime gate not passed')
    return source


def run():
    source = setup()
    if PARITY_OUT.exists():
        raise FileExistsError('V6 parity already attempted')
    PARITY_OUT.mkdir(parents=True)
    provenance = json.loads((OUT / 'v2-source-provenance.json').read_text())
    atomic(PARITY_OUT / 'resource-probe-provenance.json',
           {'rl_checkpoint_sha256': provenance['rl_checkpoint']['sha256'],
            'v2_provenance_sha256': sha(OUT / 'v2-source-provenance.json'),
            'v2_schedule_sha256': sha(OUT / 'run-schedule-v2.json'),
            'current_simulator_sha256': sha(source)})
    parity.run()


def analyze():
    source = setup()
    parity.analyze()
    result = json.loads((PARITY_OUT / 'parity/parity-result.json').read_text())
    if result['status'] != 'pass' or result['source_sha256'] != sha(source):
        raise ValueError('V6 parity failed or source changed')
    atomic(OUT / 'v6.done.json', {'status': 'pass', 'source_sha256': sha(source),
                                  'parity_result_sha256': sha(PARITY_OUT / 'parity/parity-result.json'),
                                  'trace_sha256': result['new_trace_sha256'],
                                  'compared_steps': 120})
    print('V6_PARITY_PASS', result['new_trace_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('run', 'analyze'))
    args = parser.parse_args()
    {'run': run, 'analyze': analyze}[args.stage]()
