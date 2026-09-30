"""Archive the first failed gate without replaying or changing the protocol."""
from __future__ import annotations
import hashlib,json,re,time
from pathlib import Path

ROOT=Path('/tmp/eatnav-candidate-coverage/sim-clock-long-horizon')
DOC=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/candidate-coverage/sim-clock-long-horizon'
DOC.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
log=ROOT/'t4/runs/t4-00/runner.log'
text=log.read_text(errors='replace')
assert 'different commands at same stamp 1600000000 branch 4' in text
assert 'RECOVERY_STEP live 0 1' in text
assert not (ROOT/'t4/runs/t4-00/run-complete.json').exists()
pattern=re.compile(r'CLOCK_DIAG post seq=(\d+) .*? node_ns=(\d+) .*? window_elapsed=([\d.]+) cmd_dist=([\d.]+) actual_dist=([\d.]+) recovery=(\d+) vx=([-\d.]+) wz=([-\d.]+)')
rows=[]
for line in (ROOT/'follower-logs/reset-0001-branch-0.log').read_text().splitlines():
 m=pattern.search(line)
 if m:rows.append({'seq':int(m[1]),'node_ns':int(m[2]),'elapsed':float(m[3]),'commanded_distance':float(m[4]),'actual_distance':float(m[5]),'recovery':int(m[6]),'vx':float(m[7]),'wz':float(m[8])})
assert rows
stamps=sorted(set(r['node_ns'] for r in rows))
summary={'first_stamp_ns':stamps[0],'last_stamp_ns':stamps[-1],'distinct_stamps':len(stamps),
 'branch0_callback_count':len(rows),'max_motion_window_elapsed_sec':max(r['elapsed'] for r in rows),
 'max_commanded_distance_m':max(r['commanded_distance'] for r in rows),
 'recovery_observed_branch0':any(r['recovery'] for r in rows),
 'max_callbacks_at_one_stamp':max(sum(r['node_ns']==s for r in rows) for s in stamps),
 'all_stamps_on_50ms_grid':all((s-stamps[0])%50_000_000==0 for s in stamps),
 'same_stamp_command_variation_branch0':any(len({(r['vx'],r['wz']) for r in rows if r['node_ns']==s})>1 for s in stamps)}
smoke_lines=(ROOT/'bridge-smoke-client.log').read_text().splitlines()
smoke=[]
for line in smoke_lines:
 if line.startswith(('reset ','step ')):
  d=json.loads(line[line.index('{'):]);smoke.append({'stamp_ns':d['stamp_ns'],'clock_now_ns':d['clock_now_ns'],
  'tf_stamp_ns':d['tf_stamp_ns'],'path_stamp_ns':d['path_stamp_ns'],'command':d['commands'][0],
  'callbacks':d['callback_count'][0],'rejected':d['rejected'][0]})
assert len(smoke)==5 and [x['stamp_ns'] for x in smoke]==[1_000_000_000+i*50_000_000 for i in range(5)]
atomic(ROOT/'clock-authority-trace.json',{'bridge_follower_smoke':smoke,'simulator_branch0_failed_run':summary,
 'follower_log_sha256':sha(ROOT/'follower-logs/reset-0001-branch-0.log')})
atomic(ROOT/'clock-step-order-trace.json',{'one_branch_smoke':smoke,'failed_simulator_run':{
 'state_id':'coverage-001','candidate_id':'bank-nominal','first_violation_stamp_ns':1_600_000_000,
 'first_violation_control_step':12,'branch':4,'rule':'same simulation stamp must have identical command',
 'values_not_captured':True},'runner_log_sha256':sha(log)})
atomic(ROOT/'clock-protocol-tests.json',{'pure_protocol_unit_tests':'2 passed',
 'live_one_branch_clock_tf_path_command_stamp':'5 of 5 matched',
 'production_use_sim_time_false_fixture':'7 of 7 exact, existing diagnostic artifact',
 'production_source_binary_params_unchanged':True,
 'actual_eight_branch_40_step_smoke':'failed at step 12, stamp 1.60 s, branch 4',
 'full_120_step_motion_window_check':'not reached','all_node_clock_equality':'not fully established',
 'protocol_overall':'fail'})
atomic(ROOT/'clock-protocol-decision.json',{'decision':'clock_protocol_fail','stage':'T4-A',
 'primary_failure':'same-stamp follower command mismatch','state_id':'coverage-001','candidate_id':'bank-nominal',
 'first_violation_control_step':12,'logical_time_sec':1.60,'branch':4,
 'gate_rule':'同一simulation stamp内のcommandが異なる場合はgate fail',
 'follow_on':'T4-B parity, T4-C duplicate, T5 20-run rerun and T6 Gate N not started',
 'secondary_cleanup_error':'Manual interruption during Isaac shutdown caused docker cp missing-output error in status.json; the first gate failure is recorded in simulator runner.log.',
 'evidence':{'simulator_runner_log_sha256':sha(log),'branch0_follower_log_sha256':sha(ROOT/'follower-logs/reset-0001-branch-0.log'),
  'branch4_follower_log_sha256':sha(ROOT/'follower-logs/reset-0001-branch-4.log')},
 'cause_candidates':['Follower uses latest TF (tf2::TimePointZero), so a callback stamped with new ROS time may use the previous TF and a later callback may use the newly delivered TF.',
  'Asynchronous /clock and TF delivery to separate follower processes leaves a callback order race.'],
 'cause_confirmed':False,'command_values_captured':False})
atomic(ROOT/'status.json',{'stage':'T4_STOPPED','primary_reason':'different commands at same stamp 1600000000 branch 4',
 'control_step':12,'candidate_runs_started':0,'t4_runs_completed':0,
 'secondary_cleanup_error':'docker cp failed after manual interrupt during simulator shutdown','time_unix':time.time()})
for name in ('sim-clock-config.yaml','sim-clock-source-audit.json','clock-corrected-run-identities.json',
 'clock-authority-trace.json','clock-step-order-trace.json','clock-protocol-tests.json','clock-protocol-decision.json'):
 target=DOC/name;source=ROOT/name
 if target.exists() and target.read_bytes()!=source.read_bytes():raise ValueError('existing committed artifact differs '+name)
 target.write_bytes(source.read_bytes())
raw={'runner.log':sha(log),'bridge.log':sha(ROOT/'bridge.log'),
 'follower-branch0.log':sha(ROOT/'follower-logs/reset-0001-branch-0.log'),
 'follower-branch4.log':sha(ROOT/'follower-logs/reset-0001-branch-4.log')}
report=f'''# Simulation-clock protocol: T4-A停止報告\n\n判定: **clock_protocol_fail**。`coverage-001`のnominal Pathで40-step smokeを開始したが、logical time 1.60 s（step 12）、branch 4で同一stampのcommandが異なった。指定された停止条件に従い、parity、coverage-003 duplicate、20-run Stage B、Gate Nは実行していない。\n\n## 確認できたこと\n\n- T0でHEAD、source/binary/config SHA、20 runの順序・Path SHA、閾値、予算を固定。旧8 runは使用していない。\n- 純粋なclock protocol単体テストは2件通過。実ROSの1-branch smokeは5 stamp（1.00–1.20 s）でbridge／TF／Path／commandが整合し、command拒否は0。\n- production followerを`use_sim_time=false`とした既存fixtureは7/7完全一致。production source／binary／parameter SHAはT0から不変。\n- 8-branchの実simulator smokeはstep 12で停止。branch 0の診断ログは{summary['distinct_stamps']} distinct stamp、{summary['branch0_callback_count']} callbackを記録し、最後のmotion-window elapsedは最大{summary['max_motion_window_elapsed_sec']:.2f} s、recoveryなし。停止条件のbranch 4についてcommandの2値そのものは保存できていない。\n- runnerログでは`different commands at same stamp 1600000000 branch 4`が最初のprotocol error。停止処理中の手動interrupt後にdocker cpが失敗したためstatus.jsonの二次エラーとは区別した。\n\n## 原因候補\n\nproduction followerはTFを`tf2::TimePointZero`で引く。`/clock`だけ先に新stampになり、TF配送前のcallbackが前step poseを使い、その後同一stampのcallbackが新poseを使った可能性がある。複数processへの`/clock`／TF配送順も未検証。branch 4のTF取得stampとcommand値は現ログにないため、因果は確定していない。control則・timer・thresholdは変更していない。\n\n## 証拠と制限\n\n- raw log: `{log}` (SHA256 `{raw['runner.log']}`)\n- bridge log: `{ROOT/'bridge.log'}` (SHA256 `{raw['bridge.log']}`)\n- branch 0/4 follower log: `{ROOT/'follower-logs'}` (SHA256 `{raw['follower-branch0.log']}` / `{raw['follower-branch4.log']}`)\n- 120-step clock motion history、全node clock一致、40-step parity、duplicate、T5/T6は未検証。\n\n次の変更は、branch 4を含む全branchでTF lookup stampとcommand値を同一callback単位で記録し、TF配送順のraceを確認したうえで別途判断する必要がある。今回のgateではprotocolを変更・再実行しない。\n'''
(DOC/'sim-clock-long-horizon-report.md').write_text(report)
print('FAILURE_SUMMARIZED',DOC)
