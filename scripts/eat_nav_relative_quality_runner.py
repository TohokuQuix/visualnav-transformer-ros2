"""Resume-safe detached orchestration for the relative-quality diagnostic."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
DOCS = Path(__file__).resolve().parents[1] / 'docs/quality-identifiability-gate/relative-quality'
DOCKER = 'isaac-lab-onix'
REMOTE = '/tmp/eatnav-quality-identifiability-gate/relative-quality/output'
CONFIG = DOCS / 'relative-quality-config.yaml'
STAGES = [
    ('01_semantics', ['rl-timing-semantics.json', 'rl-timing-semantics.md'], None),
    ('02_extract', ['relative-quality-extraction.json', 'relative-quality-values.jsonl.gz'], None),
    ('03_statistics', ['relative-quality-summary.json', 'relative-quality-correlations.json'], None),
    ('04_matching', ['relative-quality-match-thresholds.json', 'relative-quality-matched-pairs.json'], None),
    ('05_train', ['relative-quality-probe-training-summary.json'], 'train'),
    ('06_evaluate', ['relative-quality-probe-results.json'], 'evaluate'),
    ('07_permutation', ['relative-quality-permutation.json'], 'permutation'),
    ('08_report', ['relative-quality-report.md', 'relative-quality-decision.json'] +
     [f'plots/{name}.png' for name in (
         'relative-quality-distributions', 'terrain-confounding', 'goal-confounding',
         'quality-vs-trajectory-difference', 'probe-comparison',
         'advantage-drawdown-recovery')], 'report'),
]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for part in iter(lambda: source.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')
    os.replace(temp, path)


def run(*argv):
    print('RUN', *argv, flush=True)
    subprocess.run(argv, check=True)


def copy_remote(name):
    target = DOCS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    run('docker', 'cp', f'{DOCKER}:{REMOTE}/{name}', str(temp))
    os.replace(temp, target)


def inputs_for(stage, previous):
    files = [CONFIG, Path(__file__),
             DOCS / 'rl-timing-semantics.json',
             Path(__file__).with_name('eat_nav_relative_quality_worker.py'),
             Path(__file__).with_name('eat_nav_relative_quality_analysis.py'),
             Path(__file__).with_name('eat_nav_relative_quality_probe.py')]
    if stage == '08_report':
        files.append(Path(__file__).with_name('eat_nav_relative_quality_report.py'))
    return {'files': {str(f): digest(f) for f in files if f.exists()},
            'previous_stage': previous}


def stage_done(name, outputs, input_hashes):
    marker = ROOT / 'markers' / f'{name}.json'
    if not marker.exists():
        return None
    old = json.loads(marker.read_text())
    if old['inputs'] != input_hashes:
        raise RuntimeError(f'{name} inputs changed since completion; refusing stale resume')
    now = {file: digest(DOCS / file) for file in outputs}
    if now != old['outputs']:
        raise RuntimeError(f'{name} output SHA mismatch; refusing stale resume')
    return digest(marker)


def remote_probe(stage):
    run('docker', 'exec', '-e',
        'PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src',
        DOCKER, '/isaac-sim/python.sh',
        '/tmp/eatnav-train/eat_nav_relative_quality_probe.py', '--stage', stage)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    (ROOT / 'markers').mkdir(exist_ok=True)
    status = {'state': 'running', 'current_stage': None, 'completed': [],
              'started_unix': time.time(), 'runner_pid': os.getpid()}
    write_json(ROOT / 'status.json', status)
    previous = None
    try:
        for name, outputs, command in STAGES:
            status['current_stage'] = name
            write_json(ROOT / 'status.json', status)
            inputs = inputs_for(name, previous)
            marker_sha = stage_done(name, outputs, inputs)
            if marker_sha is None:
                if command in ('train', 'evaluate', 'permutation'):
                    remote_probe(command)
                elif command == 'report':
                    run(sys.executable, str(Path(__file__).with_name('eat_nav_relative_quality_report.py')))
                else:
                    if name == '01_semantics':
                        for f in outputs:
                            if not (DOCS / f).exists():
                                raise FileNotFoundError(DOCS / f)
                    else:
                        for f in outputs:
                            copy_remote(f)
                if command in ('train', 'evaluate', 'permutation'):
                    for f in outputs:
                        copy_remote(f)
                hashes = {file: digest(DOCS / file) for file in outputs}
                marker = ROOT / 'markers' / f'{name}.json'
                write_json(marker, {'stage': name, 'inputs': inputs,
                                    'config_sha256': digest(CONFIG), 'outputs': hashes,
                                    'completed_unix': time.time()})
                marker_sha = digest(marker)
            previous = marker_sha
            status['completed'].append(name)
            print('STAGE_COMPLETE', name, flush=True)
            write_json(ROOT / 'status.json', status)
        artifacts = {}
        for _, outputs, _ in STAGES:
            for f in outputs:
                artifacts[f] = digest(DOCS / f)
        write_json(DOCS / 'artifact-hashes.json', artifacts)
        status.update(state='complete', current_stage=None,
                      finished_unix=time.time(), artifact_hashes=artifacts)
        write_json(ROOT / 'status.json', status)
        write_json(DOCS / 'status.json', status)
        print('RELATIVE_QUALITY_COMPLETE', flush=True)
    except Exception as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}',
                      finished_unix=time.time())
        write_json(ROOT / 'status.json', status)
        print('RELATIVE_QUALITY_FAILED', status['error'], flush=True)
        raise


if __name__ == '__main__':
    main()
