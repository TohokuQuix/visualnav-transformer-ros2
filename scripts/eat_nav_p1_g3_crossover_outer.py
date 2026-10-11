"""External-log launcher for a separately approved G3 crossover process."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eat_nav_p1_g3_material_diag_outer import run_outer


def command_for(root: Path, remote: str, host_script: Path, router_logs: Path,
                anchor_id: str, batch_index: int, execution_index: int) -> list[str]:
    manifest=json.loads((root/'input/g3-contract.json').read_text())
    source_sha=hashlib.sha256(host_script.read_bytes()).hexdigest()
    if source_sha!=manifest['g3_host_source_sha256']:
        raise ValueError('crossover host source SHA drift')
    expected=f'batch-{anchor_id}-{batch_index}-period-{execution_index}'
    if root.name!=expected:
        raise ValueError('crossover batch/period root mismatch')
    return ['/usr/bin/python3',str(host_script),'--root',str(root),
            '--remote',remote,'--router-log-dir',str(router_logs),
            '--anchor-id',anchor_id,'--batch-index',str(batch_index),
            '--execution-index',str(execution_index)]


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--remote',required=True)
    parser.add_argument('--external-log-dir',type=Path,required=True)
    parser.add_argument('--host-script',type=Path,required=True)
    parser.add_argument('--router-log-dir',type=Path,required=True)
    parser.add_argument('--anchor-id',required=True)
    parser.add_argument('--batch-index',type=int,required=True)
    parser.add_argument('--execution-index',type=int,required=True)
    args=parser.parse_args()
    command=command_for(args.root,args.remote,args.host_script,args.router_log_dir,
                        args.anchor_id,args.batch_index,args.execution_index)
    result=run_outer(command,args.root,args.external_log_dir)
    if result['host_exit_code']!=0 or result['post_launcher_seal_error'] is not None:
        raise RuntimeError('crossover host/launcher failed: '+str(result))
    print('G3_CROSSOVER_OUTER_PASS',args.root)


if __name__=='__main__':main()
