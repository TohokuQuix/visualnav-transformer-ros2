"""Prove whether recorded actor observations can anchor a forward evaluator."""
from __future__ import annotations
from collections import Counter
import gzip
import json
from pathlib import Path
import torch
import yaml
from visualnav_transformer.eat_nav.relative_quality_followup_actor import actor_observation_status
from eat_nav_relative_followup_probe_train import sha,write,ROOT,OUT,SPEC

RAW=Path('/tmp/eatnav-extended/output')
RL=Path('/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt')
COLLECTOR=Path('/tmp/eatnav-extended/collect_eat_nav.py')
ENV_CFG=Path('/tmp/eatnav-extended/isaac-source/onix_isaac_lab/tasks/locomotion/velocity/config/onix/rough_env_cfg.py')


def main():
    s=yaml.safe_load(SPEC.read_text());manifest=json.loads((RAW/'manifest.json').read_text())
    if sha(RAW/'manifest.json')!=s['source_manifest_sha256']:
        raise ValueError('raw manifest SHA mismatch')
    if sha(RL)!='b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d':
        raise ValueError('actor checkpoint SHA mismatch')
    torch.set_num_threads(4);checkpoint=torch.load(RL,map_location='cpu',weights_only=False)
    weights=checkpoint['model_state_dict'];dimension=int(weights['actor.0.weight'].shape[1])
    params=yaml.safe_load((RL.parent/'params/agent.yaml').read_text())
    statuses=Counter();latent_saved=0;key_sets=Counter();rows=0
    for chunk in manifest['chunks']:
        path=RAW/chunk['file']
        if sha(path)!=chunk['sha256']:raise ValueError(f'raw chunk SHA mismatch: {path.name}')
        with gzip.open(path,'rt') as source:
            for line in source:
                row=json.loads(line);rows+=1
                statuses[actor_observation_status(row,dimension)]+=1
                latent_saved+=int('actor_latent' in row)
                key_sets[tuple(sorted(row))]+=1
    if rows!=sum(c['rows'] for c in manifest['chunks']):raise ValueError('raw record count mismatch')
    cfg=ENV_CFG.read_text();collector=COLLECTOR.read_text()
    if "'action': actions_cpu[i]" not in collector or 'actions = policy(obs)' not in collector:
        raise ValueError('collector serialization source changed')
    if 'self.enable_corruption = True' not in cfg or 'self.joint_pos.noise' not in cfg:
        raise ValueError('saved actor observation noise config changed')
    availability='available' if statuses['available']==rows else 'actor_observation_missing'
    result={'status':availability,'raw_record_count':rows,'actor_observation_dim':dimension,
       'actor_observation_status_counts':dict(statuses),'raw_schema_variants':[
         {'keys':list(keys),'count':count} for keys,count in key_sets.items()],
       'actor_latent_saved_records':latent_saved,'actor_checkpoint_sha256':sha(RL),
       'actor_checkpoint_has_weights':True,
       'empirical_normalization_enabled':bool(params.get('empirical_normalization',False)),
       'policy_observation_corruption_enabled':True,
       'saved_observation_config_sha256':sha(ENV_CFG),'saved_collector_sha256':sha(COLLECTOR),
       'joint_positions_velocities_saved':False,
       'actor_output_vs_recorded_action_check':'not_testable_without_recorded_actor_observation',
       'actor_latent_deterministic_recomputation':'unavailable_without_observation',
       'window_anchor_correspondence':'not_testable_without_recorded_actor_observation',
       'forward_evaluator_gate':'stop_before_training' if availability!='available' else 'may_continue',
       'explanation':('Collector used obs[policy] to produce action, but serialized action/pose/twist/command/DEM '
                      'without the 22-D actor observation. The saved observation has corrupted joint position, '
                      'joint velocity, projected gravity and angular velocity terms; raw records have neither '
                      'joint state nor observation noise realization. Reconstructing it from twist/action would '
                      'be an approximation and cannot verify actor action alignment.'),
       'required_future_data':'record exact pre-action obs[policy] per session/env/episode/step with policy action and source checkpoint SHA'}
    write(OUT/'actor-latent-provenance.json',result)
    print('FOLLOWUP_ACTOR_AUDIT',availability,rows,flush=True)


if __name__=='__main__':main()
