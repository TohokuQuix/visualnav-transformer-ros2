"""Seal remains immutable when the outer launcher records late console/exit data."""

import json
import hashlib
import sys

from eat_nav_p1_g3_crossover_outer import command_for
from eat_nav_p1_g3_material_diag_outer import run_outer


def fake_command(root, mutate_after_seal=False):
    code=("from pathlib import Path\n"
          "from eat_nav_p1_g1_lifecycle import seal_after_writers,verify_seal\n"
          f"r=Path({str(root)!r})\n"
          "(r/'raw.json').write_text('{}\\n')\n"
          "seal_after_writers(r,writer_state={'simulator':True,'heartbeat':True,"
          "'resource':True,'log':True},stage='synthetic')\n"
          "assert verify_seal(r)['valid']\n"
          "print('late host console after seal')\n")
    if mutate_after_seal:
        code+="(r/'late-file.txt').write_text('changed')\n"
    return [sys.executable,'-c',code]


def test_late_console_and_exit_marker_stay_outside_sealed_batch(tmp_path):
    batch=tmp_path/'batch';batch.mkdir()
    external=tmp_path/'external'
    result=run_outer(fake_command(batch),batch,external)
    assert result['host_exit_code']==0
    assert result['post_launcher_seal_verification']['valid']
    assert result['post_launcher_seal_error'] is None
    assert sorted(path.name for path in batch.iterdir())==['host-seal.json','raw.json','sha-ledger.json']
    assert 'late host console' in (external/'host-console.log').read_text()
    assert json.loads((external/'launcher-result.json').read_text())['post_launcher_seal_verification']['valid']


def test_postseal_batch_write_is_detected(tmp_path):
    batch=tmp_path/'batch';batch.mkdir()
    result=run_outer(fake_command(batch,mutate_after_seal=True),batch,tmp_path/'external')
    assert result['host_exit_code']==0
    assert result['post_launcher_seal_verification'] is None
    assert 'post_seal_mutation' in result['post_launcher_seal_error']


def test_crossover_outer_checks_frozen_host_and_period(tmp_path):
    batch=tmp_path/'batch-first-differential-0-period-0'
    (batch/'input').mkdir(parents=True)
    host=tmp_path/'host.py';host.write_text('print("host")\n')
    (batch/'input/g3-contract.json').write_text(json.dumps({
        'g3_host_source_sha256':hashlib.sha256(host.read_bytes()).hexdigest()}))
    command=command_for(batch,'/container/batch',host,tmp_path/'router',
                        'first-differential',0,0)
    assert command[-2:]==['--execution-index','0']
    host.write_text('print("drift")\n')
    try:
        command_for(batch,'/container/batch',host,tmp_path/'router',
                    'first-differential',0,0)
    except ValueError as exc:
        assert 'source SHA drift' in str(exc)
    else:
        raise AssertionError('host drift was accepted')
