"""Reconcile raw DEM, legacy complete windows and current EAT windows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import _records, assign_terrain_splits
from visualnav_transformer.eat_nav.window_waterfall import count_window_waterfall


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    episodes=_records(a.manifest)
    split_map=assign_terrain_splits([rows[0]['terrain_seed'] for rows in episodes.values()],123)
    result=count_window_waterfall(episodes,load_eat_config(a.config),split_map)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('WATERFALL_DONE',json.dumps(result),flush=True)

if __name__=='__main__':main()
