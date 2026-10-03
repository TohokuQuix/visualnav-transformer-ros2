import copy
import gzip
import json

import numpy as np
import yaml

from eat_nav_e1_analyze import duplicate_metrics, unique_candidate_outcomes
from eat_nav_e1_prepare import G0E


def simple_trace():
    return [{'follower_command':[.2,.01,0.], 'actor_input':[0.,1.],
        'actor_action':[.1]*6,'actor_latent':[.3],'critic_observation':[0.,1.],
        'critic_value':.5,'reward':.1,'root_state':[0.,0.]+[0.]*11,
        'post_root_state':[.1,0.]+[0.]*11,'contact_force':[[0.,0.,1.]],
        'dem':{'transformed':[[[0.]],[[1.]]]},'terminal':None,
        'logical_sim_time_ns':100,'tf_stamp_ns':100,'follower_command_stamp_ns':100,
        'clock_stamp_ns':100,'path_stamp_ns':100}]


def test_identical_direct_repeats_pass_frozen_duplicate_thresholds():
    gates=yaml.safe_load((G0E/'e1-gates.yaml').read_text())
    trace=simple_trace()
    result=duplicate_metrics([trace,copy.deepcopy(trace),copy.deepcopy(trace)],gates['duplicate'])
    assert result['status']=='pass'
    assert result['xy_rms_m']==0
    assert result['first_divergence'] is None


def test_command_difference_is_reported_as_duplicate_failure():
    gates=yaml.safe_load((G0E/'e1-gates.yaml').read_text())
    traces=[simple_trace() for _ in range(3)]
    traces[1][0]['follower_command']=[-.2,.01,0.]
    result=duplicate_metrics(traces,gates['duplicate'])
    assert result['status']=='fail'
    assert result['first_divergence']['signal']=='follower_command'


def test_three_direct_repeats_aggregate_to_one_oracle_candidate():
    rows=[{'candidate_id':'direct_reference','outcome':{'delta_v':v}}
          for v in (.1,.3,.2)]
    rows.append({'candidate_id':'left_1','outcome':{'delta_v':.4}})
    unique=unique_candidate_outcomes(rows)
    assert len(unique)==2
    assert unique['direct_reference']['delta_v']==.2
