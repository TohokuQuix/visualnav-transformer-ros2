from visualnav_transformer.eat_nav.minimal_p1_r3 import select_r3, evaluate_gpu_baseline


def row(anchor_id, seed, role, onset, *, step=.07):
    return {'anchor_id':anchor_id,'seed_label':seed,'primary_role':role,
            'physical_eligible':True,'hard_valid':True,'follower_compatible':True,
            'dem_history_sha256':['a','b','c','d'],'body_state_actor':[0.]*22,
            'source_family':f'{seed}:{anchor_id}','selection_rank':1,
            'pair_count':1 if role=='R-differential' else 0,
            'max_pair_rms_m':.03 if role=='R-differential' else 0,
            'paths':[{'known':True,'known_fraction':1.0,'class_proxy':'transition' if role=='R-transition-common' else 'differential',
                      'onset_m':onset,'class_diagnostic':{'maximum_step_m':step}} for _ in range(5)]}


def test_earlier_onset_beats_larger_transition_margin():
    rows=[row('first-differential','first','R-differential',None),
          row('second-differential','second','R-differential',None),
          row('first-graded','first','R-transition-common',.2,step=.06),
          row('first-flat-only','first','R-transition-common',.6,step=.15),
          row('second-graded','second','R-transition-common',.2,step=.06),
          row('second-flat-only','second','R-transition-common',.6,step=.15)]
    result=select_r3(rows)
    assert result['pass']
    assert result['selected']==['first-differential','second-differential','first-graded','second-graded']


def test_missing_early_transition_fails_without_reclassification():
    rows=[row('first-differential','first','R-differential',None),
          row('second-differential','second','R-differential',None),
          row('first-flat-only','first','R-transition-common',.6),
          row('second-graded','second','R-transition-common',.2)]
    result=select_r3(rows)
    assert not result['pass']
    assert result['decision']=='source_or_artifact_integrity_invalid'


def test_gpu_baseline_rejects_unrelated_process_above_allowance():
    contract={'required_gpu_uuid':'GPU-test','required_gpu_name':'NVIDIA GeForce RTX 5070 Ti',
              'audited_total_memory_mib':16303,'required_total_memory_mib_min':16000,
              'gpu_device_total_hard_cap_mib':7680,'other_process_gpu_hard_cap_mib':256,
              'unresolved_gpu_bytes_hard_cap':0,'absolute_attribution_gap_hard_cap_mib':256}
    sample={'uuid':'GPU-test','name':'NVIDIA GeForce RTX 5070 Ti','mig_mode':'[N/A]',
            'total_mib':16303,'device_used_mib':442,
            'other_processes_used_mib':396,'unresolved_pid_bytes':0,
            'bracket_gap_mib':46,'temperature_c':49}
    result=evaluate_gpu_baseline(sample,contract)
    assert not result['pass']
    assert result['failures']==['other_process_gpu_over_cap']


def test_gpu_baseline_rejects_hardware_identity_drift():
    contract={'required_gpu_uuid':'GPU-test','required_gpu_name':'NVIDIA GeForce RTX 5070 Ti',
              'audited_total_memory_mib':16303,'required_total_memory_mib_min':16000,
              'gpu_device_total_hard_cap_mib':7680,'other_process_gpu_hard_cap_mib':256,
              'unresolved_gpu_bytes_hard_cap':0,'absolute_attribution_gap_hard_cap_mib':256}
    sample={'uuid':'GPU-test','name':'Different GPU','mig_mode':'Enabled',
            'total_mib':16001,'device_used_mib':100,
            'other_processes_used_mib':0,'unresolved_pid_bytes':0,
            'bracket_gap_mib':0,'temperature_c':49}
    failures=evaluate_gpu_baseline(sample,contract)['failures']
    assert 'gpu_name_changed' in failures
    assert 'gpu_memory_profile_changed' in failures
    assert 'gpu_mig_profile_changed' in failures
