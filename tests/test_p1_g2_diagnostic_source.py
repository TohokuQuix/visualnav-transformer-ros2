"""Static checks only: importing the simulator would start Isaac Sim."""

import ast
import hashlib
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/eat_nav_p1_g2_diagnostic_runner.py'
G1_HOST = Path(__file__).resolve().parents[1] / 'scripts/eat_nav_p1_g1_gpu_host.py'
CONFIG = Path(__file__).resolve().parents[1] / 'configs/eat-nav/p1-g2-contract.json'


def test_runner_is_separate_and_compiles():
    assert SOURCE.name != 'eat_nav_p1_paired_prepare.py'
    ast.parse(SOURCE.read_text())


def test_g2_has_frozen_direct_path_and_durable_per_step_trace():
    source = SOURCE.read_text()
    assert "g2_contract['control_steps'] != 600" in source
    assert "g2_contract['episode_length_s'] != 40.0" in source
    assert "p['bearing_deg']==0" in source
    assert "conditions=['A']*4" in source
    assert "trace_streams[i].append(record)" in source
    assert "gates[i].accept(record['timing_protocol'])" in source
    assert "if reset_calls or any(ended):" in source
    assert "args.output.is_relative_to(persistent_parent)" in source
    frozen=json.loads(CONFIG.read_text())
    assert frozen['g2_simulator_source_sha256'] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert (frozen['control_steps'],frozen['episode_length_s']) == (600,40.0)


def test_existing_p1_source_is_not_modified():
    # Diagnostic source must keep its own SHA contract, not masquerade as P1.
    source = SOURCE.read_text()
    assert "g2_simulator_source_sha256" in source
    assert "freeze['pilot_simulator_source_sha256']" not in source


def test_g1_host_uses_new_root_and_live_per_pid_nvml():
    host=G1_HOST.read_text()
    ast.parse(host)
    assert "if ROOT.name == 'g1-attempt-0'" in host
    assert 'capture_live(nvml' in host
    assert 'capture_sample(nvml' in host
    assert 'gpu_bracket_count' in host
    assert "'owned_simulator_residual'" in host


def test_g1_failure_seal_preserves_sha_evidence(tmp_path, monkeypatch):
    import eat_nav_p1_g1_gpu_host as host
    monkeypatch.setattr(host, 'ROOT', tmp_path)
    (tmp_path / 'resource-series.jsonl').write_text('{"sample":1}\n')
    host.seal('G1_failed_or_inconclusive', reason='synthetic_monitor_failure')
    ledger=json.loads((tmp_path / 'sha-ledger.json').read_text())['file_sha256']
    assert 'resource-series.jsonl' in ledger
    assert ledger['resource-series.jsonl'] == hashlib.sha256(
        (tmp_path / 'resource-series.jsonl').read_bytes()).hexdigest()
    seal=json.loads((tmp_path / 'host-seal.json').read_text())
    assert seal['reason'] == 'synthetic_monitor_failure'
    assert seal['sha_ledger_sha256'] == hashlib.sha256(
        (tmp_path / 'sha-ledger.json').read_bytes()).hexdigest()
