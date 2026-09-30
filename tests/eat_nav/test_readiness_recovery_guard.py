import hashlib
import json
import subprocess
import sys
import time

import pytest

import eat_nav_evaluator_readiness_run as run
from visualnav_transformer.eat_nav.readiness_recovery import terminate_tree


def _completed(tmp_path, entry):
    folder = tmp_path / "runs" / entry["run_id"]
    folder.mkdir(parents=True)
    (folder / "run-complete.json").write_text(json.dumps({"identity": entry, "output_sha256": {}}))


def test_failed_run_is_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    assert not run.run_valid({"run_id": "readiness-100"})


def test_first_hundred_completed_runs_are_reusable_without_docker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "docker", lambda *args: pytest.fail("completed run invoked docker"))
    entries = [{"run_id": f"readiness-{i:03d}", "order": i} for i in range(100)]
    for entry in entries:
        _completed(tmp_path, entry)
    assert sum(run.run_valid(entry) for entry in entries) == 100
    assert all(run.execute(entry, time.time() + 60)["identity"] == entry for entry in entries)


def test_failed_attempt_zero_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    folder = tmp_path / "runs" / "readiness-100"
    folder.mkdir(parents=True)
    log = folder / "runner.log"
    log.write_text("RECOVERY_STEP live 0 20\nRECOVERY_FATAL timeout\n")
    before = hashlib.sha256(log.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match="partial run requires inspection"):
        run.execute({"run_id": "readiness-100"}, time.time() + 60)
    assert hashlib.sha256(log.read_bytes()).hexdigest() == before


def test_identity_mismatch_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    _completed(tmp_path, {"run_id": "readiness-000", "candidate_id": "original"})
    with pytest.raises(ValueError, match="identity marker mismatch"):
        run.run_valid({"run_id": "readiness-000", "candidate_id": "changed"})


def test_cleanup_terminates_a_running_process_within_sixty_seconds():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(100)"],
                               start_new_session=True)
    started = time.monotonic()
    terminate_tree(process, term_wait_sec=0.1, kill_wait_sec=0.1)
    assert process.poll() is not None
    assert time.monotonic() - started < 60


def test_cleanup_escalates_when_sigterm_is_ignored():
    process = subprocess.Popen([sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(100)"],
        stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert process.stdout.readline().strip() == "ready"
    started = time.monotonic()
    terminate_tree(process, term_wait_sec=0.1, kill_wait_sec=0.1)
    assert process.poll() is not None
    assert process.returncode != 0
    assert time.monotonic() - started < 60


def test_frozen_candidate_path_seed_and_order_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    entry = {"run_id": "readiness-100", "candidate_id": "bank-strong_left",
             "base_path_sha256": "frozen-digest", "seed": 123, "order": 100}
    _completed(tmp_path, entry)
    for field, replacement in (("candidate_id", "other"),
                               ("base_path_sha256", "changed-digest"),
                               ("seed", 124), ("order", 101)):
        changed = dict(entry, **{field: replacement})
        with pytest.raises(ValueError, match="identity marker mismatch"):
            run.run_valid(changed)
