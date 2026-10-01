"""Record the approved four-hour stop without fabricating candidate outcomes."""
from __future__ import annotations
import hashlib,json,pathlib,time
ROOT=pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
DOC=pathlib.Path('/tmp/eatnav-vnt-goal-ood-simulator/docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def put(p,d):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n'
 t=p.with_name(p.name+'.tmp');t.write_text(text);t.replace(p)
if (ROOT/'pilot-stop.done.json').exists():
    marker=read(ROOT/'pilot-stop.done.json')
    if read(ROOT/'status.json')['stage']!='STOPPED_RUNTIME_FORECAST':raise ValueError('stop marker/status drift')
    print('RESUME_SKIP_RUNTIME_STOP',marker['status'],flush=True)
    raise SystemExit(0)
start=read(ROOT/'pilot-start.json');first=read(ROOT/'runs/train-003-anchor/final-status.json');second=read(ROOT/'runs/train-003-anchor-v2/final-status.json')
rows=read(DOC/'run-identities.json')['runs'];states=read(ROOT/'inputs/selected-states.json')['train']
assert len(rows)==44 and not list((ROOT/'runs').glob('*/live-0.json'))
assert first['elapsed_sec']>300 and second['elapsed_sec']>200
prefix_per_state={s['state_id']:s['step']+1 for s in states}
live_prefix_steps=sum(prefix_per_state[s['state_id']]*11 for s in states)
# The first observed reset occurs no later than step 399. Deduct a very generous
# 200 s for launch/initialization, so this is a deliberately optimistic forecast.
conservative_sec_per_step=max(0,(first['elapsed_sec']-200)/400)
conservative_live_prefix_sec=live_prefix_steps*conservative_sec_per_step
assert conservative_live_prefix_sec>4*3600
revision=read(ROOT/'anchor-diagnostic-revision.json')
put(ROOT/'source-revision.json',{'previous_sim_sha256':start['source_sha256'],
 'revised_sim_sha256':sha('/tmp/eatnav-vnt-goal-ood-simulator/scripts/eat_nav_goal_subgoal_sim.py'),
 'reason':revision['failure'],'candidate_runs_before_revision':0,'reset_event_or_dynamics_changed':False})
artifact={'status':'stopped_before_duplicate_and_candidate','stop_reason':'conservative full-prefix forecast exceeds fixed 4 h wall cap',
 'original_started_unix':start['started_unix'],'stop_unix':time.time(),
 'elapsed_wall_sec_including_failed_and_revised_anchor':time.time()-start['started_unix'],
 'first_anchor_elapsed_sec':first['elapsed_sec'],'first_anchor_result':'failed diagnostic assertion on an expected prior-episode reset',
 'revised_anchor_elapsed_sec':second['elapsed_sec'],'revised_anchor_result':'operator-stopped before target anchor because runtime forecast exceeded cap',
 'first_target_state':states[0]['state_id'],'first_target_episode':states[0]['episode_key'][2],
 'prefix_steps_by_state':prefix_per_state,'live_runs_per_state':11,
 'total_full_prefix_steps_live_runs':live_prefix_steps,
 'forecast_assumption':'first reset observed no later than 400 steps; subtract 200 s launch/initialization from 344.9 s elapsed as a generous allowance; ignore 120-step suffix, followers and four anchors',
 'conservative_prefix_sec_per_step':conservative_sec_per_step,
 'conservative_44_live_prefix_hours':conservative_live_prefix_sec/3600,
 'fixed_wall_cap_hours':4,
 'gpu_peak_mib':max(first['gpu_peak_mib'],second['gpu_peak_mib']),
 'cpu_anon_peak_bytes':max(first['cpu_anon_peak_bytes'],second['cpu_anon_peak_bytes']),
 'raw_storage_bytes':sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()),
 'candidate_runs_complete':0,'duplicate_runs_complete':0,'anchor_verified':False,
 'residual_simulator_bridge_follower_router_processes':0,'simulator_close_after_operator_term':'Isaac native SIGTERM ended with recursion abort; process exited and no residual process'}
put(DOC/'runtime-storage.json',artifact)
for name in ('duplicate-gates.json','goal-execution-results.json','path-command-realized-results.json',
             'relative-outcome-results.json','far-ood-diagnostic.json','input-readiness.json'):
 put(DOC/name,{'status':'NA_not_run','reason':artifact['stop_reason'],'candidate_runs_complete':0,
     'duplicate_runs_complete':0,'no_performance_claim':True})
put(DOC/'gate-decision.json',{'status':'stopped_runtime_forecast','GS1':'NA','GS2':'NA','GS3':'NA','GS4':'logging parity only; selected-state collection not run',
 'final_decision':'no candidate performance decision','pilot_reapproval_needed_after_runtime_design_review':True,
 'same_four_states_goals_seeds_paths_preserved':True,'no_scaled_collection_or_retraining':True})
report=f'''# EAT-Nav Goal-conditioned Local Subgoal simulator pilot: runtime gate stop

## 判定

**44 live runは開始せず停止した。** GS1–GS4とfar-OODの実走行性能はNAであり、local subgoal方式のpass/failは判定しない。offlineで固定した4 state、16 goal、32 Path、seed、順序は変更していない。NoMaD、Evaluator、RL、follower制御則、reset event、dynamicsも変更していない。

## S0とlogging parity

元のoffline proposal SHA、manifest/split、NoMaD/RL checkpoint、32 Pathのidentity/SHAを照合し、44 runを事前固定した。既知の`readiness-002`をlogging追加前後で120 step比較し、command、Actor pre/post-delay・latent・action、Critic、reward、root/joint、contact、DEM、realized trajectoryが完全一致した。raw joint effortは`robot.data.applied_torque`から52 jointで取得でき、4 flipper＋左右track sumの6 termへ変換した。rawを[-100,100]でclipしてから0.01倍するActor pre-noise termとの最大差は3.34e-6。実機onlineで同等の順序付きeffortを取得できるかは未確認。

最初のparity smokeはZenoh router不在でPath subscriberが0となり、Path投入前に失敗した。router起動後の同一run parityは通過した。その後に過去episode resetの診断assertionだけを修正したため、最新source全体のparity再試験は未実施である。pilot再開時には再試験が必要。pilot専用routerは停止し、残留simulator/bridge/follower/ROS processは0。

## 停止理由と資源

選択stateはepisode-local prefixが20/20/35/40 stepだが、現行の確立済みfresh-process replayは過去episodeを含むglobal full-prefixを再生する。global prefixは`{prefix_per_state}`で、44 live run合計{live_prefix_steps:,} stepとなる。

最初のtrain-003 anchorは、episode 4より前の正常resetを旧診断assertionが拒否し、{first['elapsed_sec']:.1f}秒で停止した。診断assertionだけを修正し同じstateで再試行したが、{second['elapsed_sec']:.1f}秒時点でruntime forecastにより終了した。anchorの一致は**未確認**であり、duplicateとcandidateは各0件。前回試行のログは保持した。

最初のresetは遅くとも400 stepまでに生じる。測定{first['elapsed_sec']:.1f}秒から起動固定費を過大な200秒控除しても、44 live runのprefixだけで**約{conservative_live_prefix_sec/3600:.1f}時間**となる。120-step suffix、follower、残る4 anchorはこの値に含まない。固定上限4時間を超えるため、新規runを開始しなかった。

GPU peak {artifact['gpu_peak_mib']} MiB、cgroup anonymous CPU memory peak {artifact['cpu_anon_peak_bytes']/1e9:.2f} GB、raw保存量 {artifact['raw_storage_bytes']/1e6:.1f} MB。二度目のanchorはoperator SIGTERMを受け、Isaac native closeがrecursion abortでexit 1となったが、processは終了し残留0。これは候補outcomeではない。

## 次の設計課題

同じ固定state・Pathを使用するには、対象episode開始状態からのepisode-local replayがfull-prefixと一致することを別gateで立証する必要がある。過去の同一process snapshot restoreや未検証の短縮方式へ切り替えて44 runを続行しない。短縮方式が成立しなければ、4時間上限と固定stateの組合せは実行不能である。新しいstate、goal、seed、Pathをこの結果を見て選び直していない。

成果物のNA JSONは未実行を明示する。performance図はデータがないため作成していない。raw log・failure marker・statusは`/tmp/eatnav-candidate-coverage/goal-ood-simulator/`に保存した。
'''
(DOC/'goal-subgoal-simulator-report.md').write_text(report)
source=read(DOC/'source-provenance.json');source['latest_sim_source_sha256']=sha('/tmp/eatnav-vnt-goal-ood-simulator/scripts/eat_nav_goal_subgoal_sim.py')
source['source_revision_raw_marker_sha256']=sha(ROOT/'source-revision.json');source['candidate_rollouts_started']=False
put(DOC/'source-provenance.json',source)
files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in DOC.iterdir() if p.is_file() and p.name!='artifact-sha-ledger.json'}
put(DOC/'artifact-sha-ledger.json',{'status':'partial_runtime_stop','file_count':len(files),'files':files,
 'raw_sha256':{'pilot_start':sha(ROOT/'pilot-start.json'),'first_anchor_final':sha(ROOT/'runs/train-003-anchor/final-status.json'),
 'revised_anchor_final':sha(ROOT/'runs/train-003-anchor-v2/final-status.json'),
 'source_revision':sha(ROOT/'source-revision.json')},'candidate_runs':0})
prior=ROOT/'status.json'
put(ROOT/'status-before-runtime-stop.json',read(prior))
put(prior,{'stage':'STOPPED_RUNTIME_FORECAST','reason':artifact['stop_reason'],'candidate_runs':0,'duplicate_runs':0,
 'anchor_verified':False,'conservative_44_live_prefix_hours':conservative_live_prefix_sec/3600,
 'report_sha256':sha(DOC/'goal-subgoal-simulator-report.md')})
put(ROOT/'pilot-stop.done.json',{'status':'stopped_as_instructed','report_sha256':sha(DOC/'goal-subgoal-simulator-report.md'),
 'runtime_storage_sha256':sha(DOC/'runtime-storage.json'),'artifact_ledger_sha256':sha(DOC/'artifact-sha-ledger.json')})
print('RUNTIME_STOP_RECORDED',round(conservative_live_prefix_sec/3600,2),'h',flush=True)
