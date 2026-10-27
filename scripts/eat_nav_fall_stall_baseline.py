#!/usr/bin/env python3
"""Run or independently verify the frozen fall/stall task-3 comparison."""
import argparse
import json
from pathlib import Path
from visualnav_transformer.eat_nav.fall_stall_comparison import run_comparison,verify_comparison


def main():
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=repo/'docs/eat-nav/fall-stall-dataset-freeze-20261019/artifacts')
    parser.add_argument('--config',type=Path,default=repo/'config/eat_nav/fall_stall_baseline_v1.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    result=verify_comparison(args.output,args.dataset) if args.verify_only else run_comparison(args.dataset,args.config,args.output,repo)
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
