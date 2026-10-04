"""Attempt-3 entrypoint: same frozen E1 schedule, fresh ownership protocol root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import eat_nav_e1_attempt2 as flow


def configure_attempt3():
    prior=Path('/tmp/eatnav-e1-attempt2-20261004')
    flow.PRIOR=prior
    flow.ROOT=Path('/tmp/eatnav-e1-attempt3-20261004')
    flow.LABEL='attempt-3'
    if prior not in flow.HISTORICAL:flow.HISTORICAL.append(prior)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','continue'])
    args=parser.parse_args();configure_attempt3()
    if args.stage=='preflight':print(json.dumps(flow.static_preflight(),sort_keys=True),flush=True)
    else:flow.continue_runs()
