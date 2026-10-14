#!/usr/bin/env python3
"""Launch one isolated Phase 2 CPU ROS role; no production follower topics."""

import argparse
from pathlib import Path


def build_parser():
    parser=argparse.ArgumentParser()
    parser.add_argument('role',choices=('sealed_source','inference','supervisor','supervisor_gui',
                                        'display','mock_robot'))
    parser.add_argument('--sealed-root',type=Path)
    parser.add_argument('--env-id',type=int,default=0)
    parser.add_argument('--mode',choices=('mock14_transport','legacy6_demo',
                                          'learned14_adopted'),default='mock14_transport')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--worker-python',type=Path)
    parser.add_argument('--results',type=Path)
    parser.add_argument('--normalization',type=Path)
    parser.add_argument('--sim-time', action='store_true')
    parser.add_argument('--evaluation-period-ns', type=int)
    parser.add_argument('--predictor-zero-velocity-command',
                        action=argparse.BooleanOptionalAction, default=True,
                        help='Zero Actor indices 6–8 only for the adopted Motion Predictor')
    return parser


def main():
    parser=build_parser()
    args=parser.parse_args()
    if args.role=='sealed_source':
        if args.sealed_root is None:
            parser.error('--sealed-root required for sealed_source')
        from visualnav_transformer.eat_nav.ros2_phase2.source_nodes import main as run
        run(args.sealed_root,args.env_id)
    elif args.role=='display':
        from visualnav_transformer.eat_nav.ros2_phase2.display_node import main as run
        run()
    elif args.role=='mock_robot':
        from visualnav_transformer.eat_nav.ros2_phase2.mock_robot import main as run
        run()
    else:
        from visualnav_transformer.eat_nav.ros2_phase2.nodes import main as run
        run(args.role,mode=args.mode,checkpoint=args.checkpoint,
            python=args.worker_python,results=args.results,
            normalization=args.normalization,sim_time=args.sim_time,
            evaluation_period_ns=args.evaluation_period_ns,
            predictor_zero_velocity_command=args.predictor_zero_velocity_command)


if __name__=='__main__': main()
