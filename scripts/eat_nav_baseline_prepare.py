"""Build a versioned derived dataset from read-only sealed ROS-free P1 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualnav_transformer.eat_nav.evaluator.dataset import (
    SOURCE_CROSSOVER, SOURCE_TEACHER, build_crossover, build_teacher_capture, write_derived,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',choices=('crossover','teacher'),required=True)
    parser.add_argument('--root',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.source=='crossover':
        samples=build_crossover(args.root)
        source=SOURCE_CROSSOVER
    else:
        if len(args.root)!=1:
            raise ValueError('teacher capture requires one root')
        samples=build_teacher_capture(args.root[0])
        source=SOURCE_TEACHER
    metadata=write_derived(samples,args.output,source=source)
    print(json.dumps({'output':str(args.output),'samples':metadata['sample_count'],
                      'independent_groups':metadata['independent_group_count'],
                      'source':source},sort_keys=True))


if __name__=='__main__':main()
