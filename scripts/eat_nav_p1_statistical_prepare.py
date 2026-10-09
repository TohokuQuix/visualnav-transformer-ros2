"""Read-only historical audit and prospective P1 statistical assignment freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualnav_transformer.eat_nav.p1_batch_golden import verify_backup
from visualnav_transformer.eat_nav.p1_statistical_freeze import (
    freeze_manifest, sha, verify_ledger,
)


def verify_historical(golden: Path, batch: Path, minimal: Path) -> dict:
    golden_result = verify_backup(golden)
    batch_results = {}
    for name in ('eatnav-p1-batch-parity-20261009',
                 'eatnav-p1-batch-parity-20261009-v2'):
        root = batch/name
        seal = json.loads((root/'seal.json').read_text())
        ledger = root/'sha-ledger.json'
        if sha(ledger) != seal['sha_ledger_sha256']:
            raise ValueError(f'historical batch ledger SHA mismatch: {name}')
        batch_results[name] = verify_ledger(root,ledger)
    seal = json.loads((minimal/'package-seal.json').read_text())
    ledger = minimal/'sha256-ledger.json'
    if sha(ledger) != seal['ledger_sha256']:
        raise ValueError('minimal backup ledger SHA mismatch')
    files = json.loads(ledger.read_text())
    if len(files) != seal['file_count'] or any(sha(minimal/relative)!=digest
                                              for relative,digest in files.items()):
        raise ValueError('minimal backup file SHA mismatch')
    return {'golden':golden_result,'batch':batch_results,
            'minimal':{'file_count':len(files),'ledger_sha256':sha(ledger)}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-contract',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--golden',type=Path,required=True)
    p.add_argument('--batch-backup',type=Path,required=True)
    p.add_argument('--minimal-backup',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    evidence=verify_historical(args.golden,args.batch_backup,args.minimal_backup)
    freeze=freeze_manifest(args.source_contract,args.config,args.output)
    path=args.output/'historical-audit.json'
    path.write_text(json.dumps(evidence,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'freeze':freeze,'historical_audit_sha256':sha(path)},sort_keys=True))


if __name__=='__main__':
    main()
