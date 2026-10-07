"""Host marker classification; a diagnostic cannot become a full smoke."""
from __future__ import annotations
import json
from pathlib import Path


def evaluate_run_markers(folder: Path, mode: str, native_exit_code: int, *, expected_pose_count: int = 5) -> str:
    folder=Path(folder)
    if native_exit_code!=0:
        return 'invalid'
    failure=folder/'failure.json'
    if mode=='diagnostic':
        invalid=folder/'contact-index-invalid.json'
        if invalid.exists() and failure.exists():
            data=json.loads(invalid.read_text())
            if data.get('first_invalid') and data.get('raw_sha256'):
                return 'diagnostic_index_invalid'
        finished=folder/'diagnostic-finished.json'
        if finished.exists() and not failure.exists():
            data=json.loads(finished.read_text())
            if data.get('physics_steps')==60:
                return 'diagnostic_complete_60'
        return 'invalid'
    if mode=='full':
        finished=folder/'sim-finished.json'
        if finished.exists() and not failure.exists():
            data=json.loads(finished.read_text())
            if data.get('physics_steps')==200 and data.get('pose_count')==expected_pose_count:
                return 'full_complete'
    return 'invalid'
