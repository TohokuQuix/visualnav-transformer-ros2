"""Publish compact plots, fix comparison, and report without changing the protocol."""
from __future__ import annotations
import hashlib,json,math,shutil,subprocess,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path('/tmp/eatnav-candidate-coverage/tf-race-diagnostic')
DOC=Path(__file__).resolve().parents[1]/'docs/quality-identifiability-gate/candidate-coverage/tf-race-diagnostic'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');t.replace(p)
def main():
 status=read(ROOT/'status.json');assert status['stage']=='R4_COMPLETE' and status['conflict'] and status['repeats']==1
 conflict=read(ROOT/'same-stamp-command-conflict.json');align=read(ROOT/'tf-command-alignment.json')
 first=read(ROOT/'tf-race-first-divergence.json');decision=read(ROOT/'tf-race-cause-decision.json')
 assert decision['classification']=='R1 TF delivery race supported' and conflict['preceding_count']>=20 and conflict['following_count']>=20
 A=conflict['A_follower_callback'];B=conflict['B_follower_callback'];timeline=align['timeline']
 path=read('/tmp/eatnav-candidate-coverage/pilot-inputs/state-001/bank-nominal-map-path.json')
 assert timeline['path_event']['points']==path
 assert abs(A['vx']-.3*(1-abs(A['heading_error'])/math.pi))<1e-12
 assert abs(B['vx']-.3*(1-abs(B['heading_error'])/math.pi))<1e-12
 counts=read(ROOT/'tf-race-callback-counts.json')['counts']
 branch=int(conflict['branch']);stamp=int(conflict['stamp_ns'])
 callback=[]
 import gzip
 with gzip.open(ROOT/f'repeat-00/callback-traces/branch-{branch}.jsonl.gz','rt') as f:
  for line in f:
   x=json.loads(line)
   if x['phase']=='command_pre_publish' and x['command_stamp_ns']==stamp:callback.append(x)
 base=timeline['clock_publish']['publish_lower_steady_ns']
 def ms(ns):return (ns-base)/1e6
 fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
 ax=axes[0,0]
 points=[('clock',timeline['clock_publish']['publish_lower_steady_ns'],0),('A lookup',A['tf_start_steady_ns'],1),
  ('TF publish',timeline['current_tf_publish']['publish_steady_ns'],2),('A recv',timeline['A_receive']['steady_ns'],3),
  ('B lookup',B['tf_start_steady_ns'],4),('B recv',timeline['B_receive']['steady_ns'],5)]
 for label,t,y in points:ax.scatter(ms(t),y,s=40);ax.text(ms(t),y+.1,label,fontsize=8)
 ax.set_yticks([]);ax.set_xlabel('steady time since clock publish (ms)');ax.set_title('Clock → TF → callback → command')
 ax=axes[0,1];seq=[x['callback_seq'] for x in callback]
 ax.plot(seq,[x['tf_x'] for x in callback],label='TF x');ax.plot(seq,[x['tf_y'] for x in callback],label='TF y')
 ax.set_xlabel('callback seq');ax.set_ylabel('map position (m)');ax.set_title('Returned TF pose, same ROS stamp');ax.legend()
 ax=axes[0,2];ax.plot(seq,[x['wz'] for x in callback],label='angular command')
 ax.axvline(A['callback_seq'],color='C1',ls='--',label='A');ax.axvline(B['callback_seq'],color='C2',ls='--',label='B')
 ax.set_xlabel('callback seq');ax.set_ylabel('angular velocity (rad/s)');ax.set_title('Command changes with returned TF');ax.legend()
 ax=axes[1,0];ax.bar(range(8),[counts[f'0:{j}'] for j in range(8)])
 ax.set_xlabel('branch');ax.set_ylabel('callbacks');ax.set_title('Callback count by branch')
 ax=axes[1,1];ax.plot(seq,[x['heading_error'] for x in callback],label='heading error')
 ax.plot(seq,[x['carrot_y'] for x in callback],label='carrot y')
 ax.set_xlabel('callback seq');ax.set_title('Lookahead and heading change');ax.legend()
 ax=axes[1,2];ax.bar(['TF age A','TF age B'],[A['tf_age_ns']/1e6,B['tf_age_ns']/1e6])
 ax.set_ylabel('ms');ax.set_title('ROS time − returned TF stamp')
 figure=DOC/'tf-race-timeline.svg';DOC.mkdir(parents=True,exist_ok=True);fig.savefig(figure);plt.close(fig)
 figure.write_text('\n'.join(line.rstrip() for line in figure.read_text().splitlines())+'\n')
 options='''# Simulation clock TF race: 修正案（未実装）\n\n| 案 | 根本原因への対応 | production followerとの差／制御則 | replayとfixture | 実装量・主要test |\n|---|---|---|---|---|\n| 1. simulation-only exact-stamp TF lookup **推奨** | 対象stepのTF以外をlookupで受け付けない。前step poseを使う経路を直接閉じる | simulation用copyのlookup時刻だけ変更。速度則・stuck則・20 Hz wall timerは維持 | logical stampと一対一。`use_sim_time=false`のproduction fixtureは元binaryで維持 | 小。TF遅延時のtimeout、全8 branch stamp一致、first-command、40-step parity、120-step duplicateを再検証 |\n| 2. follower TF-ready acknowledgment後の採用 | follower内のTF buffer到着を確認してからcommandを選ぶ | follower→bridgeの診断ackと採用条件が追加される。制御則は維持 | ack順序を凍結できるがmessageとcommandの紐付けが必要 | 中。ack欠落・遅延・古いack拒否と同一stamp複数callbackをtest |\n| 3. simulation step同期adapter | follower callback起動をstep境界へ限定する | wall timer経路をsimulationで置換するためproductionとの差が大きい | 一定のcallback数にできるがfixtureの実行経路が変わる | 大。ROS lifecycle・timer・TF・command全体の等価性test |\n| 4. `/clock`/TF barrier | clockを進める前にTF配送完了を保証する協調protocol | follower側ackなしではROSの2 topicを原子的に配送できない | 完全なbarrierを作れれば決定的。現構成への追加が多い | 大。全branch ack、timeout、clock単調性、deadlock test |\n\n**推奨は案1。** 今回のAはROS時刻1.70 sで1.65 sのTFを返し、Bは1.70 sのTFを返した。exact-stamp lookupならAに対応する古いposeは使用されない。TF未到着時の待ち／timeoutに伴うcommand欠損を含めて再評価する。今回の診断ではどの案も実装しない。\n\nStage B再開前には、simulation-onlyの変更に対するsource/binary SHA監査、production fixture、8-branch 20-step smoke、40-step parity、120-step duplicate、motion-window 6 s非ready、全stampのTF/Path/command整合と旧20-run入力不変をgateとして再確認する。\n'''
 (DOC/'tf-race-fix-options.md').write_text(options)
 runtime=read(ROOT/'repeat-00/stage-complete.json')
 raw=read(ROOT/'tf-race-source-log-sha.json')
 save_names=['tf-race-config.yaml','tf-race-source-audit.json','tf-race-instrumentation-diff.json',
  'tf-race-run-identities.json','tf-race-build.json','stage-r0-complete.json',
  'tf-command-alignment.json','tf-race-first-divergence.json','tf-race-cause-decision.json',
  'tf-race-source-log-sha.json','tf-race-callback-counts.json','status.json','graceful-stop-request.json','bridge-cleanup.json']
 for name in save_names:shutil.copy2(ROOT/name,DOC/name)
 compact_first={k:conflict[k] for k in ('branch','stamp_ns','A','B','A_receive','B_receive')}
 first_raw=ROOT/'same-stamp-command-conflict-first.json'
 compact_first['raw_path']=str(first_raw);compact_first['raw_sha256']=sha(first_raw)
 atomic(DOC/'same-stamp-command-conflict-first.json',compact_first)
 keys=('callback_seq','wall_steady_ns','ros_clock_ns','tf_stamp_ns','tf_x','tf_y','tf_yaw',
  'heading_error','carrot_x','carrot_y','vx','wz','path_seq','path_stamp_ns','mode','linear_recovery','motion_ready')
 compact={**compact_first,'A_follower_callback':{k:A[k] for k in keys},
  'B_follower_callback':{k:B[k] for k in keys},
  'preceding_count':conflict['preceding_count'],'following_count':conflict['following_count'],
  'preceding_callback_seq':[x['callback_seq'] for x in conflict['preceding_callback_records']],
  'following_callback_seq':[x['callback_seq'] for x in conflict['following_callback_records']],
  'raw_path':str(ROOT/'same-stamp-command-conflict.json'),
  'raw_sha256':sha(ROOT/'same-stamp-command-conflict.json')}
 atomic(DOC/'same-stamp-command-conflict.json',compact)
 for branch_id in range(8):
  source=ROOT/f'repeat-00/callback-traces/branch-{branch_id}.jsonl.gz';target=ROOT/'callback-traces'/source.name
  target.parent.mkdir(exist_ok=True)
  if not target.exists():shutil.copy2(source,target)
 for kind in ('clock-publish','tf-publish','path-publish','bridge-command'):
  source=ROOT/f'repeat-00/{kind}-trace.jsonl.gz';target=ROOT/source.name
  if not target.exists():shutil.copy2(source,target)
 source_sha={'diagnostic_bridge_sha256':sha(Path(__file__).resolve().parent/'eat_nav_tf_race_bridge.py'),
  'diagnostic_sim_runner_sha256':sha(Path(__file__).resolve().parent/'eat_nav_tf_race_sim.py'),
  'diagnostic_host_runner_sha256':sha(Path(__file__).resolve().parent/'eat_nav_tf_race_run.py'),
  'diagnostic_follower_binary_sha256':sha(ROOT/'diagnostic-follower-build-r1/simple_trajectory_follower_node'),
  'production_follower_binary_sha256':sha('/home/kojima/onix/build/simple_trajectory_follower/simple_trajectory_follower_node')}
 atomic(ROOT/'tf-race-runtime-source-audit.json',source_sha)
 shutil.copy2(ROOT/'tf-race-runtime-source-audit.json',DOC/'tf-race-runtime-source-audit.json')
 report=f'''# EAT-Nav same-stamp TF/command race diagnostic\n\n## 判定\n\n**R1: TF delivery race supported.** 凍結済み`coverage-001` nominal、8 branch、20-step上限で1 fresh-process repeatを実行し、logical time **1.70 s**にbranch **4**で同一stampのcommand差を再現した。ここで停止し、repeat 2/3、parity、duplicate、Stage B、candidate rolloutは実行していない。\n\n| | callback A | callback B |\n|---|---:|---:|\n| sequence | {A['callback_seq']} | {B['callback_seq']} |\n| ROS clock / command stamp (s) | {A['ros_clock_ns']/1e9:.2f} | {B['ros_clock_ns']/1e9:.2f} |\n| returned TF stamp (s) | {A['tf_stamp_ns']/1e9:.2f} | {B['tf_stamp_ns']/1e9:.2f} |\n| TF x/y (m) | {A['tf_x']:.6f} / {A['tf_y']:.6f} | {B['tf_x']:.6f} / {B['tf_y']:.6f} |\n| heading error (rad) | {A['heading_error']:.9f} | {B['heading_error']:.9f} |\n| command vx / wz | {A['vx']:.9f} / {A['wz']:.9f} | {B['vx']:.9f} / {B['wz']:.9f} |\n\nmotion historyのsample数とelapsedはA/B間で変わるが、両方でwindow ready=falseであり、recovery分岐は発火していない。Pathは同一SHA `6790f278…`、Path seq={A['path_seq']}、mode=`{A['mode']}`、stuck/recovery=false。TFのXY差は{align['command_recalculation']['A_to_B_tf_translation_m']:.6f} m。既存パラメータで`wz=0.5×heading_error`、`vx=0.3×(1−|heading_error|/π)`を再計算した残差は両者とも0。controller内部のmotion elapsedはA {A['motion_elapsed_sec']:.2f} s、B {B['motion_elapsed_sec']:.2f} sだが、20 s windowは両方ready=falseで制御modeに差はない。\n\n時系列は `/clock` publish開始→AのTF lookup（旧1.65 s）→新1.70 s TF publish→BのTF lookup（新1.70 s）→bridge receive。Aのlookup終了は新TF publish開始より前、Bのlookup開始はpublish終了より後。publish wall時刻だけでなくfollowerが返したTF stampで確認した。図: [timeline](tf-race-timeline.svg)。\n\n## 実行・保存\n\nR0でbranch/HEAD、stateとepisode、Path・NoMaD・RL checkpoint、production/診断follower、bridge/runner、T4 failureログと閾値を固定した。旧branch 4 raw logにはcallback単位のA/B値がなく推定で補っていない。logging-only follower copyのC++差分とbinary SHAは監査JSONに保存した。production source/binary/config、clock protocol、TF lookup、bridge command選択は変更していない。\n\n最初の起動はoutput directory未作成でsimulation開始前に停止した。directory作成だけ修正し同じ凍結入力で実行した。conflict artifactをatomic保存し、全8 follower logのflush/close後にdetached cleanupがsimulatorへSIGTERMを送った。bridgeはSIGTERM後もsocket acceptで残ったため、run完了と全follower終了を確認した後に専用cleanupで終了した。結果retryは行っていない。Docker側status.jsonはcopy前後SHA一致。\n\n実測wall {runtime['elapsed_sec']:.1f} s、GPU peak {runtime['gpu_peak_mib']} MiB、raw log/traceは`{ROOT}`。全branchのcallback JSONLとclock/TF/Path/bridge JSONLをgzip保存し、SHAを`tf-race-source-log-sha.json`に記録した。CPU RAM peakは計測していない。bridgeの最終cleanupではSIGTERMで終了しなかったためSIGKILLを使用したが、全run marker・follower logを閉じた後である。unit testは4件通過し、診断C++ build・Python syntax・原本SHA・8 branch JSON parseを最終検証した。\n\n## 制限\n\n診断ログはcallback時刻に影響しうるため、再現確率を評価するものではない。今回の実行で旧TF→新TFという因果順序は直接観測できた。TF lookup失敗時の例外文字列はcallback JSONLには含めていないが、A/Bはいずれもlookup成功。修正案は[比較表](tf-race-fix-options.md)に示し、実装していない。\n'''
 (DOC/'tf-race-diagnostic-report.md').write_text(report)
 print('TF_RACE_REPORT',DOC)
if __name__=='__main__':main()
