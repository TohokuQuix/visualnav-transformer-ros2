"""Flush-on-event JSONL journal for command-delivery diagnostics."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class EventJournal:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("x", buffering=1)
        self._lock = threading.Lock()

    def emit(self, event: str, **fields):
        row = {
            "event": event,
            "steady_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
            "thread_id": threading.get_ident(),
            **fields,
        }
        with self._lock:
            self._file.write(json.dumps(row, separators=(",", ":")) + "\n")
            self._file.flush()

    def close(self):
        with self._lock:
            self._file.close()
