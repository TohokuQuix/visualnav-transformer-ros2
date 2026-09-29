"""Freeze physical-contact bound from on-policy train records only."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.dataset import assign_terrain_splits


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    manifest=json.loads(a.manifest.read_text())
    split=assign_terrain_splits(manifest['metadata']['terrain_seeds'],seed=123)
    values=[]
    for chunk in manifest['chunks']:
        with gzip.open(a.manifest.parent/chunk['file'],'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if split[int(row['terrain_seed'])]=='train':
                    value=float(row['contact_force_max'])
                    if np.isfinite(value):values.append(value)
        if len(values)%50000<5000:print('CONTACT_TRAIN_ROWS',len(values),flush=True)
    if len(values)<50000:raise ValueError('train-only contact support too small')
    bound=float(np.quantile(values,.995))
    atomic(a.output,{'source':'raw_train_terrain_seed_split_only','manifest_sha256':sha(a.manifest),
                     'source_sha256':sha(Path(__file__)),'split_seed':123,'sample_count':len(values),
                     'contact_force_max_train_p995_N':bound,
                     'quantiles_N':{'p50':float(np.quantile(values,.5)),
                                    'p95':float(np.quantile(values,.95)),
                                    'p99':float(np.quantile(values,.99)),
                                    'p995':bound,'max':float(np.max(values))},
                     'counterfactual_outcomes_used':False})
    print('CONTACT_THRESHOLD_FROZEN',bound,len(values),flush=True)

if __name__=='__main__':main()
