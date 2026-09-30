"""Freeze approved same-stamp TF/command race inputs before instrumentation."""
from __future__ import annotations
import hashlib,json,subprocess
from pathlib import Path
import yaml
REPO=Path(__file__).resolve().parents[1]
OUT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')
OLD=Path('/tmp/eatnav-candidate-coverage/sim-clock-long-horizon')
FOLLOWER=Path('/home/kojima/onix/dep/onix_automation/teaching_playback/simple_trajectory_follower')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(obj,indent=2,sort_keys=True)+'\n'
 if p.exists():assert p.read_text()==text
 else:
  t=p.with_suffix(p.suffix+'.tmp');t.write_text(text);t.replace(p)
def main():
 status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=REPO,text=True)
 assert status=='?? scripts/eat_nav_tf_race_stage0.py\n',repr(status)
 state=json.loads(Path('/tmp/eatnav-candidate-coverage/pilot-inputs/selected-states.json').read_text())['train'][1]
 assert state['state_id']=='coverage-001'
 path=Path('/tmp/eatnav-candidate-coverage/pilot-inputs/state-001/bank-nominal-map-path.json')
 nomad=Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')
 paths={
  'nominal_path':path,'nomad_reference_checkpoint':nomad,
  'production_follower_cpp':FOLLOWER/'src/simple_trajectory_follower.cpp',
  'production_follower_header':FOLLOWER/'include/simple_trajectory_follower/simple_trajectory_follower.hpp',
  'production_follower_binary':Path('/home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node'),
  'production_follower_params':FOLLOWER/'config/onix.yaml',
  'diagnostic_follower_cpp':Path('/tmp/eatnav-candidate-coverage/clock-domain-diagnostic/diagnostic-follower-source/src/simple_trajectory_follower.cpp'),
  'diagnostic_follower_header':Path('/tmp/eatnav-candidate-coverage/clock-domain-diagnostic/diagnostic-follower-source/include/simple_trajectory_follower/simple_trajectory_follower.hpp'),
  'bridge':REPO/'scripts/eat_nav_sim_clock_bridge.py',
  'clock_publisher':REPO/'scripts/eat_nav_sim_clock_bridge.py',
  'tf_publisher':REPO/'scripts/eat_nav_sim_clock_bridge.py',
  'simulator_runner':REPO/'scripts/eat_nav_sim_clock_long_horizon_sim.py',
  'host_runner':REPO/'scripts/eat_nav_sim_clock_run.py',
  't4_failure_log':OLD/'t4/runs/t4-00/runner.log',
  't4_status':OLD/'status.json','t4_source_audit':OLD/'sim-clock-source-audit.json',
  't4_decision':OLD/'clock-protocol-decision.json',
 }
 sources={k:{'path':str(v),'sha256':sha(v),'size_bytes':v.stat().st_size} for k,v in paths.items()}
 rl_sha=subprocess.check_output(['docker','exec','isaac-lab-onix','sha256sum',
  '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt'],text=True).split()[0]
 old_branch4=(OLD/'follower-logs/reset-0001-branch-4.log').read_text()
 assert 'CLOCK_DIAG post' not in old_branch4 and 'stamp_nsec=600000000' not in old_branch4
 config={'schema_version':1,'state_id':'coverage-001','candidate_id':'bank-nominal',
  'repeat_count_max':3,'control_steps':20,'logical_t0_ns':[1_000_000_000,11_000_000_000,21_000_000_000],
  'branch_count':8,'branch_frame_ids':[f'branch_{i}/chassis/body' for i in range(8)],
  'control_period_sec':.05,'follower_control_rate_hz':20,'duplicate_xy_rms_limit_m':.0105,
  'same_stamp_command_abs_tolerance':1e-9,'clock_protocol_unchanged':True,
  'path_sha256':sources['nominal_path']['sha256'],'rl_checkpoint_sha256':rl_sha,
  'nomad_checkpoint_sha256':sources['nomad_reference_checkpoint']['sha256'],
  'output_root':str(OUT)}
 OUT.mkdir(parents=True,exist_ok=True)
 cf=OUT/'tf-race-config.yaml';data=yaml.safe_dump(config,sort_keys=True)
 if cf.exists():assert cf.read_text()==data
 else:cf.write_text(data)
 write(OUT/'tf-race-source-audit.json',{'git':{'branch':subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip(),
  'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
  'status_at_R0':status},'state':state,'sources':sources,'rl_checkpoint_sha256_from_collection_manifest':rl_sha,
  'prior_branch4_command_values_present':False,'prior_branch4_raw_log_sha256':sha(OLD/'follower-logs/reset-0001-branch-4.log')})
 write(OUT/'tf-race-run-identities.json',{'runs':[{'repeat':i,'state_id':'coverage-001','candidate_id':'bank-nominal',
  'logical_t0_ns':t0,'control_steps':20,'path_sha256':sha(path),'fresh_process':True}
  for i,t0 in enumerate(config['logical_t0_ns'])]})
 write(OUT/'stage-r0-complete.json',{'config_sha256':sha(cf),'source_audit_sha256':sha(OUT/'tf-race-source-audit.json'),
  'run_identities_sha256':sha(OUT/'tf-race-run-identities.json'),'status':'complete'})
 print('R0_COMPLETE',OUT)
if __name__=='__main__':main()
