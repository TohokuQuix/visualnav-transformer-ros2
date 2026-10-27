"""Freeze initial paired inputs, v2 teachers and seed-grouped exploratory splits."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from visualnav_transformer.eat_nav.fall_stall_dataset import build_dataset


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--teachers',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(build_dataset(args.source,args.teachers,args.output),indent=2))
