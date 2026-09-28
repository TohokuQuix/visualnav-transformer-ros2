"""Record the exact split/window/episode/terrain identities used by the gate."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','quality-config','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();config=load_eat_config(a.quality_config)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(a.output,'wt') as stream:
        for split in ('train','val','test'):
            dataset=EATDataset(a.manifest,split,config)
            for i in range(len(dataset)):
                w=dataset[i]
                row={'split':split,'window_index':i,'terrain_seed':int(w['terrain_seed']),
                     'episode_key':list(w['episode_key']),'step':int(w['step']),
                     'q_bin':min(int(w['quality']*4),3),'valid_future_points':int(w['action_mask'].sum())}
                stream.write(json.dumps(row,separators=(',',':'))+'\n')
            print('IDS_DONE',split,len(dataset),flush=True)

if __name__=='__main__':main()
