#!/usr/bin/env python3
"""Prepare batch02 runtime binding, or collect using that exact sealed binding."""
import argparse
import json
from pathlib import Path

from visualnav_transformer.eat_nav.fall_stall_audit import sha
from visualnav_transformer.eat_nav.fall_stall_batch02_contract import load_freeze
from visualnav_transformer.eat_nav.fall_stall_batch02_runner import (
    ADDED_SOURCES, create_binding, save_preparation, load_preparation, run_collection, verify_collection,
)
from visualnav_transformer.eat_nav.fall_stall_batch02_runtime import probe_runtime, stage_preflight, recheck_runtime
from visualnav_transformer.eat_nav.fall_stall_blind_verification import BlindVerificationError


def main(argv=None):
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,default=repo/'docs/eat-nav/fall-stall-additional-batch02-freeze-20261021/artifacts')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--check-prepared',action='store_true')
    mode.add_argument('--collect',action='store_true')
    mode.add_argument('--verify-only',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--binding-sha256')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.resume and not args.collect:raise ValueError('resume requires collect')
        c=load_freeze(args.freeze);root=Path(c['manifest']['output_root'])
        if args.prepare:
            if args.output is None or args.output.exists() or root.exists():raise ValueError('new outputs required')
            runtime=probe_runtime(c)
            runtime['stage_parity']=stage_preflight(c,{p:sha(c['repo']/p) for p in ADDED_SOURCES})
            bound=create_binding(c,runtime);save_preparation(c,args.output,bound)
            result={'preparation':'pass','collection_started':False,'reserved_test':'unopened',
                    'execution_binding_sha256':sha(args.output/'execution-binding.json'),
                    'preparation_ledger_sha256':sha(args.output/'sha-ledger.json'),'output':str(args.output)}
        elif args.verify_only:
            result=verify_collection(c,root)
        else:
            if args.prepared is None or args.binding_sha256 is None:raise ValueError('pinned preparation required')
            bound=load_preparation(c,args.prepared,expected_binding_sha256=args.binding_sha256)
            if args.check_prepared:
                recheck_runtime(c,bound)
                result={'preparation':'pass','runtime_recheck':'pass','reserved_test':'unopened',
                        'collection_started':root.exists(),'execution_binding_sha256':args.binding_sha256}
            else:
                result=run_collection(c,root,bound,resume=args.resume)
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False));return 0
    except BlindVerificationError as error:
        print(json.dumps({'pass':False,'error_code':error.code,'row_index':error.row_index}));return 1
    except Exception:
        print(json.dumps({'pass':False,'error_code':'BINDING_MISMATCH'}));return 1


if __name__=='__main__':raise SystemExit(main())
