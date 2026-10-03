"""Frozen G0-E numeric anchor and DEM gate, called before any E1 Path publish."""
from __future__ import annotations

import numpy as np

from eat_nav_g0e_formal_gate import compare_anchor
from eat_nav_goal_subgoal_dem_gate import verify_dem_history


def check_before_path(summary,logged,recorded_frames,selected_goal,selected_dem,*,
                      source_sha256,state_config,reference_config,limits):
    state=summary['state']
    if summary['source_sha256']['runner_source']!=source_sha256:
        raise ValueError('anchor runner source SHA differs')
    compared=compare_anchor(summary,logged,state,selected_goal,limits)
    if compared['status']!='pass':
        raise ValueError('anchor numeric/identity gate failed: '+repr(compared['reasons']))
    dem=verify_dem_history(entry={'state_id':state['state_id'],'episode_key':state['episode_key'],
            'anchor_step':state['step']},summary=summary,recorded_frames=recorded_frames,
        selected_dem=selected_dem,reference_config=reference_config,
        state_config=state_config,source_sha256=source_sha256,
        expected_source_sha256=source_sha256)
    if dem['status']!='pass':raise ValueError('anchor four-frame DEM gate failed')
    anchor=summary['anchor']
    for key in ('joint_position','joint_velocity','policy_actor_input','actor_latent',
                'critic_observation','raw_joint_effort','root_state'):
        if key in anchor and anchor[key] is not None and not np.isfinite(anchor[key]).all():
            raise ValueError('anchor nonfinite '+key)
    return {**compared,'dem_history_gate':dem,
            'historical_reference_NA':compared['historical_reference_NA'],
            'path_publication_allowed':True}
