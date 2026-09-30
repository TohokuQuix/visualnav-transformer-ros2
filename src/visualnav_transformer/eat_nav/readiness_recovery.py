"""Process cleanup for diagnostic command-delivery probes."""

from __future__ import annotations

import os
import signal
import subprocess


def terminate_tree(process: subprocess.Popen, *, term_wait_sec: float = 10,
                   kill_wait_sec: float = 10) -> None:
    """Stop a process group, escalating after a bounded graceful wait."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=term_wait_sec)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=kill_wait_sec)
