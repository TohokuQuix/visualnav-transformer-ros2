"""Freeze inputs and clocks for the approved simulation-clock Stage B rerun."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import yaml

REPO=Path(__file__).resolve().parents[1]
OUT=Path('/tmp/eatnav-candidate-coverage/sim-clock-long-horizon')
OLD=Path('/tmp/eatnav-candidate-coverage/command-preview-long-horizon')
DIAG=Path('/tmp/eatnav-candidate-coverage/clock-domain-diagnostic')
FOLLOWER=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def save_once(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    rendered=json.dumps(data,indent=2,sort_keys=True)+'\n'
    if path.exists():
        if path.read_text()!=rendered:raise ValueError(f'frozen artifact differs: {path}')
        return
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(rendered);tmp.replace(path)


def stage_schedule(runs):
    assert len(runs)==20 and [r['order'] for r in runs]==list(range(20))
    assert [r['state_id'] for r in runs]==[s for s in ('coverage-001','coverage-003','coverage-004','coverage-006') for _ in range(5)]
    assert [r['kind'] for r in runs]==['nominal','nominal','nominal','nomad_oracle','bank_winner']*4
    fixed=[{**r,'logical_t0_ns':81_000_000_000+10_000_000_000*r['order']} for r in runs]
    return fixed


def main():
    status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO,text=True)
    if status!='?? scripts/eat_nav_sim_clock_stage0.py\n':
        raise RuntimeError('unexpected worktree content before T0: '+repr(status))
    old_ids=json.loads((OLD/'long-horizon-run-identities.json').read_text())['runs']
    runs=stage_schedule(old_ids)
    source={}
    for key,path in {
        'production_follower_cpp':FOLLOWER/'src/simple_trajectory_follower.cpp',
        'production_follower_header':FOLLOWER/'include/simple_trajectory_follower/simple_trajectory_follower.hpp',
        'production_follower_binary':Path('/home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node'),
        'production_follower_params':FOLLOWER/'config/onix.yaml',
        'bridge':REPO/'scripts/eat_nav_phase_c_ros_bridge.py',
        'simulator_runner':REPO/'scripts/eat_nav_command_preview_long_horizon_sim.py',
        'host_runner':REPO/'scripts/eat_nav_command_preview_long_horizon_run.py',
        'old_run_identities':OLD/'long-horizon-run-identities.json',
        'clock_diagnostic_decision':DIAG/'clock-hypothesis-decision.json',
        'clock_diagnostic_report':DIAG/'clock-diagnostic-report.md',
        'clock_diagnostic_alignment':DIAG/'clock-alignment.json',
        'clock_diagnostic_motion':DIAG/'follower-motion-window-trace.json',
    }.items():source[key]={'path':str(path),'sha256':sha(path)}
    for run in runs:
        idx=int(run['state_id'][-3:]);path=Path('/tmp/eatnav-candidate-coverage/pilot-inputs')/f'state-{idx:03d}'/f"{run['candidate_id']}-map-path.json"
        assert sha(path)==run['map_path_sha256']
    config={
        'schema_version':1,'authority':'simulator runner only',
        'clock_topic':'/clock','use_sim_time':True,
        'control_period_sec':0.05,'control_rate_hz':20,'sim_step_ns':50_000_000,
        'clock_start_schedule':'T4 smoke 1s; parity 11/21/31/41s; duplicate 51/61/71s; T5 run n 81+10n seconds',
        'step_order':['clock t_k','TF t_k','Path t_k when needed','fresh stamped follower command','Actor action','physics','clock t_k+1'],
        'motion_window_sec':20.0,'stuck_ratio_threshold':0.25,'duplicate_xy_rms_limit_m':0.0105,
        'smoke_state':'coverage-001','smoke_candidate':'bank-nominal','smoke_steps':40,
        'parity_states':['coverage-001','coverage-003','coverage-004','coverage-006'],'parity_steps':40,
        'duplicate_state':'coverage-003','duplicate_candidate':'bank-nominal','duplicate_repeats':3,'duplicate_steps':120,
        'stage_b_run_count':20,'stage_b_steps':120,'stage_b_wall_limit_sec':5400,'stage_b_storage_limit_bytes':250_000_000,
        'stage_b_startup_retry_limit':1,'gpu_memory_abort_mib':14000,
        'old_stage_b_root':str(OLD/'long-horizon'),'new_root':str(OUT),
        'production_follower_source_sha256':source['production_follower_cpp']['sha256'],
        'production_follower_binary_sha256':source['production_follower_binary']['sha256'],
        'production_follower_params_sha256':source['production_follower_params']['sha256'],
    }
    OUT.mkdir(parents=True,exist_ok=True)
    yaml_text=yaml.safe_dump(config,sort_keys=True)
    config_path=OUT/'sim-clock-config.yaml'
    if config_path.exists():assert config_path.read_text()==yaml_text
    else:config_path.write_text(yaml_text)
    save_once(OUT/'sim-clock-source-audit.json',{'git':{'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
        'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'status_before_implementation':'','status_at_t0':status},
        'source':source,'estimated_t4_wall_min':20,'estimated_t5_wall_min':55,
        'estimated_gpu_peak_mib':5500,'estimated_cpu_rss_peak_mib':5500,
        'estimated_t5_storage_mb':100,'old_completed_runs_retained':8,'old_runs_reused':0})
    save_once(OUT/'clock-corrected-run-identities.json',{'runs':runs,
        'old_run_identity_sha256':source['old_run_identities']['sha256'],
        't4_schedule':[{'stage':'smoke','state_id':'coverage-001','candidate_id':'bank-nominal','steps':40,'logical_t0_ns':1_000_000_000},
            *[{'stage':'parity','state_id':s,'candidate_id':'bank-nominal','steps':40,'logical_t0_ns':11_000_000_000+10_000_000_000*j}
              for j,s in enumerate(config['parity_states'])],
            *[{'stage':'duplicate','state_id':'coverage-003','candidate_id':'bank-nominal','steps':120,
                'repeat':j,'logical_t0_ns':51_000_000_000+10_000_000_000*j} for j in range(3)]]})
    save_once(OUT/'stage-t0-complete.json',{'config_sha256':sha(config_path),
        'audit_sha256':sha(OUT/'sim-clock-source-audit.json'),
        'identities_sha256':sha(OUT/'clock-corrected-run-identities.json'),
        'status':'complete'})
    print('T0_FROZEN',OUT)

if __name__=='__main__':main()
