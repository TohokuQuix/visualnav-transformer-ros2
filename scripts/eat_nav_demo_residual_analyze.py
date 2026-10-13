#!/usr/bin/env python3
"""Read-only sealed trace analysis; writes derived figures to a fresh root."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from visualnav_transformer.eat_nav.demo_residual import summarize_cancel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-step', type=int)
    args = parser.parse_args()
    original = args.root/'sim-output'
    ledger_path = original/'sha-ledger.json'
    seal = json.loads((original/'host-seal.json').read_text())
    ledger = json.loads(ledger_path.read_text())
    if sha(ledger_path) != seal['ledger_sha256'] or any(
            sha(original/name) != digest for name, digest in ledger.items()):
        raise ValueError('source SHA/seal mismatch')
    rows = [json.loads(line) for line in (original/'trace.jsonl').open()]
    steps = [row for row in rows if row['event'] == 'step']
    cancel = [row for row in rows if row['event'] == 'cancel']
    if not steps:
        raise ValueError('empty step trace')
    reference = cancel[0]['step'] if cancel else args.reference_step
    times = [row['stamp_ns']/1e9 for row in steps]
    x = [row['pose_post'][0] for row in steps]
    y = [row['pose_post'][1] for row in steps]
    lin = [math.dist(row['pose_pre'][:2], row['pose_post'][:2])/.05 for row in steps]
    angular = [(row['pose_post'][2]-row['pose_pre'][2])/.05 for row in steps]
    command = [row['command'][0] for row in steps]
    yaw_command = [row['command'][2] for row in steps]
    actor_input = [row['actor_input_command'][0] for row in steps]
    args.output.mkdir(parents=True, exist_ok=False)

    def line_marker(ax):
        if reference is not None:
            ax.axvline(reference*.05, color='crimson', linestyle='--', label='Cancel/reference')

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y, linewidth=1, label='root XY')
    ax.scatter(x[0], y[0], label='start', marker='o')
    if reference is not None and 0 <= reference < len(steps):
        ax.scatter(x[reference], y[reference], color='crimson', label='Cancel/reference', marker='x')
    ax.set(xlabel='map x [m]', ylabel='map y [m]', title='Root trajectory')
    ax.axis('equal')
    ax.grid(True); ax.legend()
    fig.tight_layout(); fig.savefig(args.output/'root-xy.png'); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    if reference is not None and 0 <= reference < len(steps):
        origin = steps[reference]['pose_pre']
        displacement = [math.dist(origin[:2], row['pose_post'][:2]) if row['step'] >= reference else math.nan
                        for row in steps]
        axes[0].plot(times, displacement, label='net displacement from reference')
    axes[0].set_ylabel('XY displacement [m]'); axes[0].grid(True)
    axes[1].plot(times, lin, label='root XY speed (pose finite difference)')
    axes[1].plot(times, command, label='follower forward command')
    axes[1].plot(times, actor_input, label='Actor delivered forward command', alpha=.7)
    axes[1].set_ylabel('forward/XY speed [m/s]'); axes[1].grid(True); axes[1].legend()
    axes[2].plot(times, angular, label='root yaw rate (pose finite difference)')
    axes[2].plot(times, yaw_command, label='follower yaw command')
    axes[2].set_ylabel('yaw rate [rad/s]'); axes[2].set_xlabel('simulation time [s]')
    axes[2].grid(True); axes[2].legend()
    for ax in axes: line_marker(ax)
    fig.tight_layout(); fig.savefig(args.output/'command-and-motion.png'); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for j in range(6):
        axes[0].plot(times, [row['actor_action'][j] for row in steps], label=f'action {j}')
    axes[0].set_ylabel('Actor action'); axes[0].legend(ncol=3); axes[0].grid(True)
    physical = [row.get('physical_post') for row in steps]
    if all(item is not None for item in physical):
        for j, name in enumerate(physical[0]['joint_names']):
            axes[1].plot(times, [item['joint_vel'][j] for item in physical], label=name)
        axes[1].set_ylabel('joint velocity [rad/s]'); axes[1].legend(fontsize=6, ncol=3); axes[1].grid(True)
        if all(item['applied_torque'] is not None for item in physical):
            for j, name in enumerate(physical[0]['joint_names']):
                axes[2].plot(times, [item['applied_torque'][j] for item in physical], label=name)
            axes[2].set_ylabel('sim applied torque [N m]'); axes[2].legend(fontsize=6, ncol=3)
    else:
        axes[1].text(.05, .5, 'joint velocity unavailable in source trace', transform=axes[1].transAxes)
        axes[2].text(.05, .5, 'applied torque unavailable in source trace', transform=axes[2].transAxes)
    axes[2].set_xlabel('simulation time [s]'); axes[2].grid(True)
    for ax in axes: line_marker(ax)
    fig.tight_layout(); fig.savefig(args.output/'actor-and-joints.png'); plt.close(fig)

    result = {'source_root': str(args.root), 'source_seal_sha256': sha(original/'host-seal.json'),
              'source_trace_sha256': sha(original/'trace.jsonl'),
              'step_count': len(steps), 'reference_step': reference,
              'available': {'root_velocity_tensor': all(item is not None for item in physical),
                            'joint_velocity': all(item is not None for item in physical),
                            'applied_torque': all(item is not None and item['applied_torque'] is not None for item in physical),
                            'contact_force': all(item is not None and item['contact_net_forces_w'] is not None for item in physical)}}
    if cancel:
        result['cancel_summary'] = summarize_cancel(rows)
    durable(args.output/'analysis.json', result)
    derived = {p.name: sha(p) for p in args.output.iterdir() if p.is_file()}
    durable(args.output/'sha-ledger.json', derived)
    durable(args.output/'host-seal.json', {'ledger_sha256': sha(args.output/'sha-ledger.json'),
                                          'file_count': len(derived)})
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
