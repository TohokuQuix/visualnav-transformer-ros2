"""Outcome-blind one-state-per-terrain selection from frozen train states."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path


def rank_state(state: dict, seed: int) -> str:
    identity=[seed,state['terrain_type'],state['episode_key'],state['view_index']]
    return hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()


def choose_states(states: list[dict], *, seed: int, minimum_goal_m: float) -> dict:
    if not 0.2 < minimum_goal_m < 1.0:
        raise ValueError('minimum goal distance must exceed 0.2 m tolerance')
    ranked={}
    for index,state in enumerate(states):
        if state['split']!='train':
            raise ValueError('selection input contains non-train state')
        distance=math.hypot(float(state['goal'][0]),float(state['goal'][1]))
        entry={key:state[key] for key in ('terrain_type','episode_key','view_index','terrain_seed','step','goal')}
        entry.update(state_index=index,goal_distance_m=distance,
                     rank_sha256=rank_state(state,seed),eligible=distance>=minimum_goal_m)
        ranked.setdefault(state['terrain_type'],[]).append(entry)
    selected=[];used_episodes=set();na=[]
    for terrain in sorted(ranked):
        ranked[terrain].sort(key=lambda x:(x['rank_sha256'],x['view_index']))
        available=[x for x in ranked[terrain] if x['eligible'] and tuple(x['episode_key']) not in used_episodes]
        if not available:
            na.append(terrain)
            continue
        chosen=available[0]
        selected.append(chosen)
        used_episodes.add(tuple(chosen['episode_key']))
    return {'seed':seed,'minimum_goal_m':minimum_goal_m,'selected':selected,
            'na_terrains':na,'ranked_candidates':ranked,
            'selection_fields':['terrain_type','episode_key','view_index','goal','split'],
            'outcome_fields_used':[]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20260929)
    parser.add_argument('--minimum-goal-m',type=float,default=.3)
    args=parser.parse_args()
    source=args.input.read_bytes();states=json.loads(source)['train']
    result=choose_states(states,seed=args.seed,minimum_goal_m=args.minimum_goal_m)
    result['source_sha256']=hashlib.sha256(source).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        if json.loads(args.output.read_text())!=result:
            raise FileExistsError('frozen selection differs; refusing overwrite')
    else:
        temp=args.output.with_suffix(args.output.suffix+'.tmp')
        temp.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
        temp.replace(args.output)
    print('SELECTED',[(x['terrain_type'],x['state_index']) for x in result['selected']],
          'NA',result['na_terrains'])

if __name__=='__main__':main()
