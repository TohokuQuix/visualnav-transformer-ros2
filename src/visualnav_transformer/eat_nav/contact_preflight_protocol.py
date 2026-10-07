"""Atomic evidence and strict host gate for the one-shot contact preflight."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import traceback


SCHEMA = "eatnav_contact_preflight_r3_v1"


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        try:
            fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass
        if path.read_bytes() != data:
            raise OSError("atomic marker read-back mismatch")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def execute_with_marker_before_close(app, inspect, output: Path) -> None:
    """Persist inspection before close, even when close terminates the process."""
    output = Path(output)
    try:
        result = inspect()
        atomic_json(output / "preflight-import-pass.json", result)
    except BaseException as exc:
        (output / "preflight-import-pass.json").unlink(missing_ok=True)
        try:
            atomic_json(output / "preflight-import-failure.json", {
                "schema": SCHEMA, "error": repr(exc),
            })
        finally:
            app.close()
        raise
    app.close()


def run_calibration_with_failure_marker_before_close(app, work, output: Path) -> None:
    """Retain a physics/setup exception before SimulationApp shutdown."""
    try:
        work()
    except BaseException as exc:
        atomic_json(Path(output) / "failure.json", {
            "error": repr(exc), "traceback": traceback.format_exc(),
            "wall_ns": time.time_ns(),
        })
        raise
    finally:
        app.close()


def normalize_view_filter_paths(raw, expected: list[str], sensor_count: int) -> list[str]:
    """Accept only an unambiguous per-sensor rendering of the same filter set."""
    paths = list(raw)
    if paths == expected:
        return list(expected)
    if len(paths) == sensor_count and all(
        isinstance(row, (list, tuple)) and list(row) == expected for row in paths
    ):
        return list(expected)
    raise ValueError(f"ambiguous filter path layout: {paths!r}; expected {expected!r}")


def verify_preflight(output: Path, host: dict) -> dict:
    """Fail closed on marker, identity, lifecycle, SHA or resource mismatch."""
    output = Path(output)
    errors = []
    passed = output / "preflight-import-pass.json"
    failed = output / "preflight-import-failure.json"
    if failed.exists():
        errors.append("failure_marker_present")
    if not passed.exists():
        errors.append("pass_marker_missing")
        return {"pass": False, "errors": errors}
    try:
        marker = json.loads(passed.read_text())
    except (OSError, ValueError):
        return {"pass": False, "errors": errors + ["pass_marker_unreadable"]}
    for key in ("run_id", "source_sha256", "config_sha256", "pid", "proc_start_ticks"):
        if marker.get(key) != host.get(key):
            errors.append(key + "_mismatch")
    if marker.get("schema") != SCHEMA:
        errors.append("schema_mismatch")
    for key in ("simulationapp_initialized", "simulationmanager_import",
                "contactsensor_import", "rigidcontactview_import"):
        if marker.get(key) is not True:
            errors.append(key + "_false")
    api = marker.get("required_api")
    if not isinstance(api, dict) or not api or not all(v is True for v in api.values()):
        errors.append("required_api_false")
    if marker.get("intended_physics_steps") != 0 or marker.get("observed_physics_steps") != 0:
        errors.append("physics_step_nonzero")
    for key in ("scene_created", "cube_created", "plane_created", "robot_created", "path_published"):
        if marker.get(key) is not False:
            errors.append(key + "_nonfalse")
    if host.get("native_exit_code") != 0:
        errors.append("native_exit_nonzero")
    if host.get("owned_residual") is not False:
        errors.append("owned_residual")
    if host.get("resources_within_limits") is not True:
        errors.append("resource_limit")
    if host.get("output_sha_match") is not True:
        errors.append("output_sha_mismatch")
    return {"pass": not errors, "errors": errors}
