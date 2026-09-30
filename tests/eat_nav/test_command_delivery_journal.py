import json
import threading

from visualnav_transformer.eat_nav.command_delivery_journal import EventJournal


def test_event_journal_flushes_each_boundary(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path)
    journal.emit("publish_entry", branch=3, stamp_ns=42, sequence=7)
    row = json.loads(path.read_text().strip())
    assert row["event"] == "publish_entry"
    assert row["branch"] == 3
    assert row["stamp_ns"] == 42
    assert row["sequence"] == 7
    assert row["thread_id"] == threading.get_ident()
    assert row["steady_ns"] > 0
    journal.close()


def test_event_journal_keeps_threads_separate(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = EventJournal(path)
    thread = threading.Thread(target=lambda: journal.emit("callback_entry"))
    thread.start()
    thread.join(timeout=1)
    journal.emit("waiter_wake")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["thread_id"] != rows[1]["thread_id"]
    journal.close()
