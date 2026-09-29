"""Freeze inputs for at most three fresh-process maneuver smoke states."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    indices = [19, 13, 42]  # frozen before any candidate smoke outputs
    states = json.loads(a.selection.read_text())["train"]
    selected = {index: states[index] for index in indices}
    names = ["nominal", "slight_left", "slight_right", "strong_left", "strong_right"]
    with np.load(a.bank) as bank:
        paths = bank["paths"]
        if list(bank["names"]) != names:
            raise ValueError("frozen maneuver names mismatch")
        if paths.shape != (76, 5, 8, 2):
            raise ValueError("frozen maneuver shape mismatch")
    max_step = max(state["step"] + 40 for state in selected.values())
    actions = {}
    records = {}
    prefix_commands = {index: {} for index in indices}
    manifest = json.loads(a.manifest.read_text())
    for chunk in manifest["chunks"]:
        with gzip.open(a.manifest.parent / chunk["file"], "rt") as stream:
            for line in stream:
                row = json.loads(line)
                step = int(row["step"])
                if step > max_step:
                    continue
                env = int(row["env_id"])
                actions.setdefault(step, {})[env] = row["action"]
                for index, state in selected.items():
                    if env == state["episode_key"][1] and step <= state["step"]:
                        prefix_commands[index][step] = row["command"]
                    if step == state["step"] and env == state["episode_key"][1]:
                        if [row["session_id"], env, row["episode_id"]] != state["episode_key"] or row["pre_reset"]:
                            raise ValueError(f"anchor identity mismatch: {index}")
                        records[index] = row
        if len(records) == len(indices) and all(len(actions.get(step, {})) == 100 for step in range(max_step + 1)):
            break
    if len(records) != len(indices):
        raise ValueError("missing logged anchor")
    full = np.asarray([[actions[step][env] for env in range(100)] for step in range(max_step + 1)], dtype=np.float32)
    if full.shape != (max_step + 1, 100, 6):
        raise ValueError("incomplete all-env action prefix/suffix")
    base = yaml.safe_load(a.base_config.read_text())
    outputs = {}
    for index, state in selected.items():
        folder = a.output / f"state-{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        step = state["step"]
        prefix = folder / "prefix-actions.npz"
        suffix = folder / "suffix-actions.npz"
        if not prefix.exists():
            np.savez_compressed(prefix, actions=full[:step + 1],
                                selected_env_commands=np.asarray([prefix_commands[index][st] for st in range(step + 1)], dtype=np.float32))
        if not suffix.exists():
            np.savez_compressed(suffix, actions=full[step + 1:step + 41])
        logged = folder / "logged-anchor.json"
        if not logged.exists():
            atomic(logged, records[index])
        config = dict(base)
        config["source_state"] = {"split": "train", "state_index": index,
                                  "env_index": state["episode_key"][1],
                                  "episode_id": state["episode_key"][2], "anchor_step": step}
        config_path = folder / "config.yaml"
        if not config_path.exists():
            config_path.write_text(yaml.safe_dump(config, sort_keys=True))
        path_hashes = {}
        for path_index, name in enumerate(names):
            target = folder / f"{name}-base-path.json"
            points = [[0.0, 0.0]] + paths[index, path_index].astype(float).tolist()
            if target.exists() and json.loads(target.read_text()) != points:
                raise ValueError("existing frozen path changed")
            if not target.exists():
                atomic(target, points)
            path_hashes[name] = sha(target)
        outputs[index] = {"state": state, "paths_sha256": path_hashes,
                          "prefix_sha256": sha(prefix), "suffix_sha256": sha(suffix),
                          "logged_anchor_sha256": sha(logged), "config_sha256": sha(config_path)}
    atomic(a.output / "smoke-input-freeze.json", {"state_indices": indices, "source_sha256": {
        "manifest": sha(a.manifest), "selection": sha(a.selection), "bank": sha(a.bank),
        "base_config": sha(a.base_config)}, "states": outputs})


if __name__ == "__main__":
    main()
