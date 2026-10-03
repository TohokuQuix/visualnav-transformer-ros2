"""Formal G0-E gate tests written before the new sidecar implementation."""
import importlib

import pytest


@pytest.fixture
def formal():
    try:
        return importlib.import_module('eat_nav_g0e_formal_gate')
    except ModuleNotFoundError:
        pytest.fail('formal gate module is missing')


def test_baseline_accepts_low_attributed_device_usage(formal):
    samples=[{'device_total_used_mib':123,'unattributed_used_mib':30,
              'unresolved_pid_mib':0,'anon_shmem_bytes':2_000_000_000,
              'host_available_bytes':90_000_000_000,'residual_processes':[]} for _ in range(21)]
    assert formal.baseline_gate(samples,{'GPU_cap_mib':6656,'unattributed_cap_mib':256,
        'anon_shmem_cap_bytes':6442450944,'host_memory_floor_bytes':10095421850}) is None


@pytest.mark.parametrize(('field','value','reason'),[
    ('device_total_used_mib',6656,'external_gpu_occupancy'),
    ('unattributed_used_mib',257,'unattributed_gpu'),
    ('unresolved_pid_mib',1,'unresolved_gpu_pid'),
    ('anon_shmem_bytes',6442450945,'anon_shmem_cap'),
    ('host_available_bytes',10095421849,'host_memory_floor'),
    ('residual_processes',['old-sim'],'residual_process'),
])
def test_baseline_rejects_fixed_stop_conditions(formal,field,value,reason):
    sample={'device_total_used_mib':123,'unattributed_used_mib':30,'unresolved_pid_mib':0,
            'anon_shmem_bytes':2_000_000_000,'host_available_bytes':90_000_000_000,
            'residual_processes':[]}
    sample[field]=value
    assert formal.baseline_gate([sample]*21,{'GPU_cap_mib':6656,'unattributed_cap_mib':256,
        'anon_shmem_cap_bytes':6442450944,'host_memory_floor_bytes':10095421850})==reason


def test_anchor_comparison_checks_velocity_command_and_action(formal):
    state={'state_id':'g0e-train-000','episode_key':[0,22,0],'global_step':30,'local_step':30,
           'env_id':22,'logged_goal':[.5,.1]}
    logged={'session_id':0,'env_id':22,'episode_id':0,'step':30,
            'pose':[0,0,0,1,0,0,0],'twist':[0]*6,'action':[.1]*6,'command':[.2,.3,.4]}
    anchor={'root_state':logged['pose']+logged['twist'],'command_manager':logged['command'],
            'action_manager_action':logged['action'],'joint_position':[0]*52,
            'joint_velocity':[0]*52,'policy_actor_input':[0]*22,'actor_latent':[0]*8,
            'critic_observation':[0]*28,'critic_value':.5,'raw_joint_effort':[0]*52,
            'action_manager_previous_action':[0]*6,'dem_history':[[[0]]]*4}
    summary={'state':{**state,'step':30},'anchor':anchor,'anchor_error_vs_logged':{
        'position_m':0,'orientation_rad':0,'velocity_max_abs':0,
        'prefix_command_max_abs':0,'dem_rms_m':0,'mask_iou':1}}
    result=formal.compare_anchor(summary,logged,state,[.5,.1],{
        'position_m':.02,'orientation_rad':.03,'velocity_max_abs':1e-4,
        'command_max_abs':1e-5,'dem_rms_m':.03,'mask_iou_min':.95})
    assert result['status']=='pass'
    assert result['historical_reference_NA']==['joint_position','joint_velocity','previous_action',
        'actor_observation','actor_latent','critic_observation','raw_joint_effort']
    anchor['root_state'][7]=.01
    assert formal.compare_anchor(summary,logged,state,[.5,.1],{
        'position_m':.02,'orientation_rad':.03,'velocity_max_abs':1e-4,
        'command_max_abs':1e-5,'dem_rms_m':.03,'mask_iou_min':.95})['status']=='fail'


def test_cleanup_one_kill_is_warning_two_kills_fail(formal):
    clean={'own_residual':[],'zombie_count':0,'unrelated_signaled':False,
           'cleanup_elapsed_sec':12,'kill_used':True,'log_flushed':True,'marker_flushed':True}
    cfg={'cleanup_max_sec':30}
    assert formal.cleanup_gate([clean],cfg)==('pass','one_kill_warning')
    assert formal.cleanup_gate([clean,clean],cfg)==('fail','both_anchors_required_kill')


def test_process_accounting_twenty_with_direct_counted_once(formal):
    rows=[]
    for state in ('g0e-train-000','g0e-train-001'):
        rows += [{'state_group_id':state,'role':'anchor_only','path_id':None,'oracle_denominator':False}]
        rows += [{'state_group_id':state,'role':'nominal_duplicate','path_id':'direct_reference',
                  'oracle_denominator':i==0} for i in range(3)]
        rows += [{'state_group_id':state,'role':'candidate_outcome','path_id':f'{side}_{i}',
                  'oracle_denominator':True} for side in ('left','right') for i in range(1,4)]
    for order,row in enumerate(rows):
        row['order']=order
        row['process_id']=f'process-{order:02d}'
    assert formal.validate_process_accounting(rows)=={'total':20,'anchor':2,
        'nominal_duplicate':6,'candidate_outcome':12,'unique_path_per_group':7}


def test_device_total_is_hard_stop_even_when_simulator_tree_is_below_cap(formal):
    sample={'device_total_used_mib':6700,'simulator_process_tree_used_mib':5000,
            'other_processes_used_mib':1600,'unattributed_used_mib':100,'unresolved_pid_mib':0,
            'anon_shmem_bytes':3_000_000_000,'host_available_bytes':90_000_000_000,'oom_events_delta':0}
    assert formal.runtime_gate(sample,{'GPU_cap_mib':6656,'unattributed_cap_mib':256,
        'anon_shmem_cap_bytes':6442450944,'host_memory_floor_bytes':10095421850})=='device_GPU_cap_external_load'


def test_first_anchor_failure_blocks_second(formal):
    assert formal.allow_second_anchor({'status':'fail'}, {'status':'pass'}) is False
    assert formal.allow_second_anchor({'status':'pass'}, {'status':'fail'}) is False
    assert formal.allow_second_anchor({'status':'pass'}, {'status':'pass'}) is True


def test_formal_simulator_root_requires_source_output_and_container(formal):
    table={
        10:{'cgroup':'docker-ours','command_line':'python /tmp/eatnav-g0e-formal-anchor-20261003/formal_sim.py --output /tmp/eatnav-g0e-formal-anchor-20261003/runs/anchor-000'},
        11:{'cgroup':'docker-ours','command_line':'python /tmp/eatnav-g0e-gpu-attribution/diag_sim.py --output /tmp/eatnav-g0e-formal-anchor-20261003/runs/anchor-000'},
        12:{'cgroup':'docker-other','command_line':'python /tmp/eatnav-g0e-formal-anchor-20261003/formal_sim.py --output /tmp/eatnav-g0e-formal-anchor-20261003/runs/anchor-000'}}
    assert formal.formal_roots(table,'ours','/tmp/eatnav-g0e-formal-anchor-20261003/runs/anchor-000')=={10}
