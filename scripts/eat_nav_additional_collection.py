#!/usr/bin/env python3
"""Collect the frozen additional batch or verify it without opening test semantics."""
import argparse
import json
from pathlib import Path
from visualnav_transformer.eat_nav.fall_stall_audit import read
from visualnav_transformer.eat_nav.fall_stall_collection_contract import load_freeze
from visualnav_transformer.eat_nav.fall_stall_collection_runner import create_binding,run_collection,verify_collection
from visualnav_transformer.eat_nav.fall_stall_collection_runtime import probe_runtime
from visualnav_transformer.eat_nav.fall_stall_blind_verification import BlindVerificationError


def main():
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,default=repo/'docs/eat-nav/fall-stall-additional-collection-freeze-20261020/artifacts')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true');mode.add_argument('--collect',action='store_true');mode.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:
        contract=load_freeze(args.freeze);root=Path(contract['manifest']['output_root'])
        if args.verify_only:result=verify_collection(contract,root)
        else:
            runtime=probe_runtime(contract)
            if args.preflight:result={'preflight':'pass','binding':create_binding(contract,runtime),'collection_started':False}
            else:
                binding=read(root/'execution-binding.json') if args.resume else create_binding(contract,runtime)
                if args.resume:
                    for key in ('container_image','container_id','python','numpy','torch','isaac_version','cuda'):
                        if binding['runtime'][key]!=runtime[key]:raise BlindVerificationError('BINDING_MISMATCH')
                result=run_collection(contract,root,binding,resume=args.resume)
        print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False));return 0
    except BlindVerificationError as error:
        print(json.dumps({'pass':False,'error_code':error.code,'row_index':error.row_index}));return 1
    except Exception:
        print(json.dumps({'pass':False,'error_code':'BINDING_MISMATCH'}));return 1

if __name__=='__main__':raise SystemExit(main())
