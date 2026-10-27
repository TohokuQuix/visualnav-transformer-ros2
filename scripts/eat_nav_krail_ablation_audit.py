"""Reproduce the fixed development suite and publish descriptive paired outcomes."""
import json
import io,zipfile,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
from eat_nav_krail_ablation import AblationGuard,window_labels,exposure_partition,CONDITIONS,SEEDS
from eat_nav_risk_navigation_v2 import OnlineEvaluator,DEFAULT_CHECKPOINT,candidate_paths,CANDIDATE_IDS
from visualnav_transformer.eat_nav.fall_stall_targets import tilt
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,verify_derived,require,seal_derived
from eat_nav_krail_ablation_run import REPO,RUNS,verify_suite_sources


def verify_initial_pair(reference,actual):
    evidence={};numeric={}
    try:
        np.testing.assert_allclose(read(reference/'startup-ready.json')['initial_pose'],read(actual/'startup-ready.json')['initial_pose'],atol=1e-6,rtol=0)
        for filename in ['input-0000.npz','initial-state.npz']:
            with zipfile.ZipFile(reference/filename) as a,zipfile.ZipFile(actual/filename) as b:
                require(set(a.namelist())==set(b.namelist()),'NPZ members differ')
                for member in a.namelist():
                    aa=a.read(member);bb=b.read(member)
                    stream=io.BytesIO(aa);version=np.lib.format.read_magic(stream)
                    _,_,dtype=np.lib.format._read_array_header(stream,version)
                    if dtype.hasobject:
                        require(filename=='initial-state.npz' and member=='actor_observation.npy','unexpected object member')
                        evidence[member]=dict(method='exact_uncompressed_bytes_no_deserialization',reference_sha256=hashlib.sha256(aa).hexdigest(),actual_sha256=hashlib.sha256(bb).hexdigest())
                        require(aa==bb,'object member bytes differ')
                    else:
                        av=np.load(io.BytesIO(aa),allow_pickle=False);bv=np.load(io.BytesIO(bb),allow_pickle=False)
                        np.testing.assert_allclose(av,bv,atol=1e-6,rtol=0)
                        numeric[filename+'/'+member]=float(np.max(np.abs(av.astype(float)-bv.astype(float)))) if av.size else 0.
        return dict(matches=True,object_members=evidence,numeric_max_abs_error=numeric)
    except (ValueError,AssertionError) as exc:
        return dict(matches=False,error=str(exc),object_members=evidence,numeric_max_abs_error=numeric)


def goal_finish_diagnostics(rows,result):
    pending=0
    for row in reversed(rows):
        value=tilt(row['post_root_state'])
        if value is None or value<60-1e-9:break
        pending+=1
    end_tilt=tilt(rows[-1]['post_root_state'])
    return dict(end_tilt_deg=end_tilt,near_fall_pending_samples=pending if pending<5 else 0,
        high_tilt_at_goal=bool(result['goal_entered'] and end_tilt is not None and end_tilt>=45),
        goal_reached_while_stopped=bool(result['goal_entered'] and rows[-1]['guard_stopped']),
        post_goal_physics_observed=False)


def eligible_outcome(result):
    return (result['stop_reason'] in ['goal_reached','natural_termination'] or
            result['stop_reason']=='horizon_complete' and result['control_steps']==1800)


def observed_events(rows):
    events=[];start=None;tipped=False
    for row in rows:
        step=row['control_step'];value=tilt(row['post_root_state'])
        high=value is not None and value>=60-1e-9
        start=step if high and start is None else start if high else None
        if start is not None and step-start==4:
            events.append(dict(event='near_fall',onset_step=start,confirmation_step=step,
                onset_time_s=(start+1)/20,confirmation_time_s=(step+1)/20))
        if not tipped and ((value is not None and value>=85-1e-9) or 'tilt_over' in (row.get('terminal') or {}).get('termination_reasons',[])):
            events.append(dict(event='tip_over',event_step=step,event_time_s=(step+1)/20));tipped=True
    return events


def high_tilt_runs(rows):
    runs=[];start=None
    for i,row in enumerate(rows):
        value=tilt(row['post_root_state']);high=value is not None and value>=60-1e-9
        if high and start is None:start=i
        if start is not None and (not high or i==len(rows)-1):
            end=i if high else i-1;count=end-start+1
            runs.append(dict(onset_time_s=(rows[start]['control_step']+1)/20,end_sample_time_s=(rows[end]['control_step']+1)/20,sample_count=count,sample_span_s=(count-1)/20,interval_occupancy_s=count/20))
            start=None
    return runs


def stop_diagnostics(rows,folder,condition):
    active=[];latest=None;first_pose=None;cause_first={};episode_causes=[];episode_reasons=set()
    for i,row in enumerate(rows):
        if row['control_step']%40==0:latest=read(folder/'sim-output'/f"selection-{row['control_step']:04d}.json")
        if row['guard_stopped'] and (not i or not rows[i-1]['guard_stopped']):
            episode_reasons=set();episode_causes.append(episode_reasons)
        if row['guard_stopped']:episode_reasons.update(row['episode_causes'])
        if 'measured_tilt' in row['trip_causes'] and first_pose is None:first_pose=row['control_step']/20
        blockers=row['release_blockers'];pose=row['guard_stopped'] and 'tilt' in blockers and 'measured_tilt' in row['episode_causes']
        reasons=set()
        if row['guard_stopped'] and 'no_release_candidate' in blockers and latest is not None:
            pp=np.asarray(latest['probabilities']);q=np.asarray(latest['progress_m'])[:,3]
            if condition!='A0' and (((pp[:,2]>=.95)|(pp[:,3]>=.95))&(q<=.05)).any():reasons.add('stall')
            if condition=='A2':
                if (pp[:,0]>=.5).any():reasons.add('tip')
                if (pp[:,1]>=.5).any():reasons.add('near')
        for name in reasons:
            if name not in cause_first:cause_first[name]=row['control_step']/20
        active.append((pose,reasons))
    stopped=[r for r in rows if r['guard_stopped']]
    speeds=[float(np.linalg.norm(r['post_root_state'][7:9])) for r in stopped]
    return dict(physical_protection_first_time_s=first_pose,cause_first_active_time_s=cause_first,
        physical_active_blocker_seconds=sum(p for p,_ in active)/20,
        prediction_active_blocker_seconds=sum(bool(rs) for _,rs in active)/20,
        physical_prediction_overlap_seconds=sum(p and bool(rs) for p,rs in active)/20,
        cause_active_blocker_seconds={name:sum(name in rs for _,rs in active)/20 for name in ['stall','tip','near']},
        cause_episode_counts={name:sum(name in rs for rs in episode_causes) for name in ['measured_tilt','stall','tip','near']},
        stopped_nonzero_injected_steps=sum(bool(np.max(np.abs(r['follower_command']))>0) for r in stopped),
        stopped_nonzero_actor_seen_steps=sum(bool(np.max(np.abs(r['actor_command']))>1e-6) for r in stopped),
        stopped_max_planar_speed_m_s=max(speeds,default=0.),
        stopped_accumulated_planar_motion_m=sum(float(np.linalg.norm(np.asarray(r['post_root_state'][:2])-r['root_state'][:2])) for r in stopped))


def analyze(folder,item,model):
    verify_derived(folder);output=folder/'sim-output';result=read(folder/'result.json')
    rows=[json.loads(x) for x in (output/'trace.jsonl').read_text().splitlines()]
    guard=AblationGuard(item['condition']);error=0.;decisions=0;shadows=[]
    for row in rows:
        step=row['control_step'];require(guard.snapshot()==row['guard_state_before_step'],f'before {step}')
        guard.observe_tilt(row['measured_tilt_deg'],step)
        if step%40==0:
            saved=read(output/f'selection-{step:04d}.json')
            with np.load(output/saved['input_file'],allow_pickle=False) as data:p,q=model.predict(dict(data))
            error=max(error,float(np.max(np.abs(p-np.asarray(saved['probabilities'])))),float(np.max(np.abs(q-np.asarray(saved['progress_m'])))))
            d=guard.decide(saved['probabilities'],saved['progress_m'],saved['goal_body'],step,row['measured_tilt_deg'])
            for key,value in d.items():require(value==saved[key],f'decision {step} {key}')
            variants={}
            for mode in CONDITIONS:
                shadow=AblationGuard(mode).decide(saved['probabilities'],saved['progress_m'],saved['goal_body'],0,0.)
                variants[mode]={key:shadow[key] for key in ['scores','risk_penalty','risk_weights','guard_unsafe','exclusion_matrix','selected_index']}
            shadows.append(dict(control_step=step,meaning='instantaneous_score_filter_only_no_hysteresis_pose_or_counterfactual_outcome',conditions=variants))
            decisions+=1
        require(guard.snapshot()==row['guard_state_after_step'],f'after {step}')
        require(guard.stopped==row['guard_stopped'],'STOP replay')
        if guard.stopped:require(np.array_equal(row['follower_command'],[0.,0.,0.]),'nonzero STOP command')
    require(error<=1e-6,'prediction replay differs')
    windows=[]
    for start in range(0,len(rows),40):
        chunk=rows[start:start+40];selected=chunk[0]['selected_candidate']
        path=candidate_paths()[CANDIDATE_IDS.index(selected)].tolist() if selected!='STOP' else candidate_paths()[0].tolist()
        windows.append(window_labels(chunk,path))
    exposure={}
    for head in ['linear','turn']:
        labels=[w[head] for w in windows];part=exposure_partition(rows,labels)
        require(sum(part[k] for k in ['stop_steps','evaluable_running_steps','unknown_running_steps'])==len(rows),'exposure partition')
        part.update(positive_windows=sum(v is True for v in labels),negative_windows=sum(v is False for v in labels),unknown_windows=sum(v is None for v in labels),positive_seconds=2*sum(v is True for v in labels))
        exposure[head]=part
    sequence=[r['selected_candidate'] for r in rows if r['selected_candidate']!='STOP']
    tilted=[tilt(r['post_root_state']) for r in rows]
    tipped=any(v is not None and v>=85-1e-9 for v in tilted) or any('tilt_over' in (r.get('terminal') or {}).get('termination_reasons',[]) for r in rows)
    near=0;run=0;longest=0
    for value in tilted:
        run=run+1 if value is not None and value>=60-1e-9 else 0;longest=max(longest,run)
        if run==5:near+=1
    episodes=[]
    for i,r in enumerate(rows):
        if r['guard_stopped'] and (not i or not rows[i-1]['guard_stopped']):episodes.append(set())
        if r['guard_stopped']:episodes[-1].update(r['episode_causes'])
    return dict(seed=item['seed'],condition=item['condition'],folder=item['folder'],state='completed',
        initial_conditions_match=item.get('initial_conditions_match',False),eligible_outcome=eligible_outcome(result),
        goal_finish_diagnostics=goal_finish_diagnostics(rows,result),events=observed_events(rows),high_tilt_runs=high_tilt_runs(rows),stop_diagnostics=stop_diagnostics(rows,folder,item['condition']),
        normal_success=result['normal_success'],goal_entered=result['goal_entered'],goal_entry_time_s=result['goal_entry_time_s'],
        control_steps=len(rows),simulated_seconds=len(rows)/20,stop_reason=result['stop_reason'],
        goal_remaining_m=result['final_goal_distance_m'],goal_progress_m=4-result['final_goal_distance_m'],
        max_tilt_deg=result['max_tilt_deg'],tip_over_observed=tipped,near_fall_episodes=near,above60_longest_samples=longest,
        stopped_seconds=sum(r['guard_stopped'] for r in rows)/20,
        stop_episodes=len(episodes),physical_protection_episodes=sum('measured_tilt' in e for e in episodes),
        prediction_stop_episodes=sum(bool(e&{'stall','tip','near'}) for e in episodes),
        release_blocker_seconds={name:sum(r['guard_stopped'] and name in r['release_blockers'] for r in rows)/20 for name in ['tilt','no_release_candidate','minimum_stop','confirmation']},
        candidate_changes=sum(a!=b for a,b in zip(sequence,sequence[1:])),
        direction_changes=sum((a=='reverse_direct')!=(b=='reverse_direct') for a,b in zip(sequence,sequence[1:])),
        resume_episodes=result['guard_resume_episodes'],exposure=exposure,
        instantaneous_choice_diagnostics=shadows,replayed_steps=len(rows),replayed_decisions=decisions,prediction_max_abs_error=error)


def main(root=RUNS):
    root=Path(root);manifest=read(root/'manifest.json');verify_suite_sources(manifest)
    require(sha(REPO/'docs/eat-nav/krail-risk-ablation-spec-v0.1.md')==manifest['spec_sha256'],'spec changed')
    model=OnlineEvaluator(DEFAULT_CHECKPOINT);items=read(root/'suite-results.json');analyses=[]
    for item in items:
        analyses.append(analyze(root/item['folder'],item,model) if item['state']=='completed' else item)
    anchors={}
    for item in analyses:
        if item['state']!='completed':continue
        anchor=anchors.setdefault(item['seed'],item['folder'])
        pairing=verify_initial_pair(root/anchor/'sim-output',root/item['folder']/'sim-output')
        original=next(x for x in items if x['folder']==item['folder'])
        pairing['coordinator_initial_conditions_match']=original.get('initial_conditions_match')
        pairing['coordinator_error']=original.get('pair_error')
        item['independent_initial_pairing']=pairing
        item['initial_conditions_match']=pairing['matches']
    pairs=[]
    for seed in SEEDS:
        cases={x['condition']:x for x in analyses if x['seed']==seed and x['state']=='completed'}
        valid=len(cases)==3 and all(x['initial_conditions_match'] and x['eligible_outcome'] for x in cases.values())
        entry=dict(seed=seed,comparison_valid=valid)
        if valid:
            entry['differences']={f'{a}-{b}':{k:cases[a][k]-cases[b][k] for k in ['goal_progress_m','goal_remaining_m','max_tilt_deg','stopped_seconds','physical_protection_episodes','candidate_changes']} for a,b in [('A1','A0'),('A2','A1'),('A2','A0')]}
        pairs.append(entry)
    attempts=[]
    for folder in sorted(root.glob('run*-attempt*')):
        sealed=(folder/'derived-seal.json').exists()
        if sealed:verify_derived(folder)
        attempts.append(dict(folder=folder.name,sealed=sealed,host_failure=read(folder/'host-failure.json') if (folder/'host-failure.json').exists() else None))
    failures=[dict(file=p.name,**read(p)) for p in sorted(root.glob('failure-run*.json'))]
    write(root/'comparison-audit.json',dict(attempts=attempts,failed_attempts=failures,schema='eatnav_krail_ablation_audit_v1',formal_test=False,reserved_test_opened=False,
        checkpoint_sha256=sha(DEFAULT_CHECKPOINT),runs=analyses,paired=pairs))
    write_report(root,analyses,pairs,failures)
    plot(root,analyses)
    print(json.dumps(dict(completed=sum(x['state']=='completed' for x in analyses),valid_pairs=sum(x['comparison_valid'] for x in pairs),runs=analyses),indent=2))


def write_report(root,runs,pairs,failures):
    lines=['# K-Rail risk ablation comparison','', '2026-10-23 JST。承認仕様による既存学習済みモデルの開発比較。対象は **k_rail_step_field（段差ブロック型）level4**。別の **k_rail_field（レール状梁の型）は未実施**。追加収集・再学習・reserved testの開封は行っていない。','',
        '| seed | 条件 | 正常到達 | goal残距離m | 最大tilt度 | STOP秒 | 実姿勢保護episode | linear陽性window | turn陽性window |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in runs:
        if r['state']!='completed':lines.append(f"| {r['seed']} | {r['condition']} | 実行失敗 | — | — | — | — | — | — |");continue
        lines.append(f"| {r['seed']} | {r['condition']} | {int(r['normal_success'])} | {r['goal_remaining_m']:.3f} | {r['max_tilt_deg']:.2f} | {r['stopped_seconds']:.2f} | {r['physical_protection_episodes']} | {r['exposure']['linear']['positive_windows']} | {r['exposure']['turn']['positive_windows']} |")
    lines+=['','## 観測時間と判定可能時間','', '| seed | 条件 | sim秒 | STOP秒 | linear判定可秒 | linear陽性秒 | turn判定可秒 | turn陽性秒 | running判定不能秒 linear / turn |','|---|---|---:|---:|---:|---:|---:|---:|---|']
    for r in runs:
        if r['state']!='completed':continue
        lin=r['exposure']['linear'];turn=r['exposure']['turn']
        lines.append(f"| {r['seed']} | {r['condition']} | {r['simulated_seconds']:.2f} | {r['stopped_seconds']:.2f} | {lin['evaluable_running_steps']/20:.2f} | {lin['positive_seconds']:.2f} | {turn['evaluable_running_steps']/20:.2f} | {turn['positive_seconds']:.2f} | {lin['unknown_running_steps']/20:.2f} / {turn['unknown_running_steps']/20:.2f} |")
    lines+=['','比較成立seed数: '+str(sum(x['comparison_valid'] for x in pairs))+'/3。開始Actor入力・遅延・全初期root/joint状態とEvaluator入力の照合結果を比較成立判定へ使用。','',
        'A0はrisk項なし、A1はstallのみ、A2はfall/near＋stall。共通progress予測・切替・実姿勢45度保護は同一。','',
        '## 読み方と限界','',
        '各seedの全結果とpaired差分を[監査JSON](comparison-audit.json)へ保存した。未到達は到達時間の平均へ入れない。STOP混在windowはunknownであり、STOP時間とrunningの判定可能/不能時間をhead別に排他的集計した。低指令の完全観測windowは既知陰性。','',
        '実姿勢保護により転倒が抑えられ得るため、転倒0件だけでrisk学習の有効性を主張しない。停止だけでstall windowが減る場合も走行改善とはしない。全caseで候補が限られる場合、迂回可能性の不足と予測失敗を区別する。選ばなかった候補の実走破結果は未確認である。','',
        '初期Actor観測はTensorDictのnumpy変換によりobject memberとして保存され、coordinatorのallow_pickle=False検証が失敗した。過去の失敗flagとerrorは保持し、監査側では各seedの同じanchorに対し、数値fieldはatol1e-6、actor_observation.npyは未圧縮memberの完全byte一致で独立照合した。objectの復号・unpickleは一切行っていない。member SHAと独立判定を監査JSONへ保存。byte不一致の場合は比較不成立とする。','',
        '3 seedの記述的開発比較であり、安全性・統計的有意差・RLによる転倒抑制の因果効果・正式held-out性能は示さない。D0は過去に観察済みの開発seed。重みや閾値を結果に合わせて変更していない。','',
        '## 再現性','',
        '各完了走行の入力からモデル再推論とguard全状態/選択を再現し、STOP全3軸0を確認した。誤差・判定可能時間・unknown・実行失敗は監査JSONに保持。sourceはsuite manifestへ固定。','',
        '起動コマンド:','', '```bash','PYTHONPATH=src:scripts:. /home/ubuntu/venvs/isaaclab_py310/bin/python scripts/eat_nav_krail_ablation_run.py --output docs/eat-nav/krail-risk-ablation-20261023','```','',
        'これは実施済みコマンドであり、同じ出力先への上書き実行は拒否される。各runのsimulator-command.shに実際のDocker/Isaacコマンドを保持。','']
    lines+=['','## 失敗attempt','']
    lines+= [f"- [{f['file']}]({f['file']}): {f['type']}: {f['message']}" for f in failures] or ['失敗attemptなし。']
    lines+=['','## 正常到達数','']
    valid_seeds={p['seed'] for p in pairs if p['comparison_valid']}
    for mode in CONDITIONS:
        good=[r for r in runs if r.get('state')=='completed' and r['condition']==mode and r['normal_success']]
        lines.append(f"- {mode}: 全予定seed {len(good)}/3、比較成立seed {sum(r['seed'] in valid_seeds for r in good)}/{len(valid_seeds)}。")
    for r in runs:
        if r['state']=='completed':lines+= [f"- {r['seed']} / {r['condition']}: [結果]({r['folder']}/result.json)、[コマンド]({r['folder']}/simulator-command.sh)、[GUI]({r['folder']}/{'gui-running.png' if (root/r['folder']/'gui-running.png').exists() else 'gui-live.png'})"]
    lines+=['','![比較時間推移](comparison.png)','', '![地形サンプルと実軌跡](trajectories.png)','', '## 区切り','',
        '予定9走行と失敗attemptを報告して本比較を終了する。陽性不足を理由とする自動増枠は行わない。paired結果と失敗箇所reviewに基づく次段階の判断は、下記の考察に記す。']
    (root/'comparison-report.md').write_text('\n'.join(lines)+'\n')


def plot(root,runs):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/eatnav-krail-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,10))
    for r in runs:
        if r['state']!='completed':continue
        rows=[json.loads(x) for x in (root/r['folder']/'sim-output/trace.jsonl').read_text().splitlines()]
        i=SEEDS.index(r['seed']);times=[x['control_step']/20 for x in rows]
        axes[i,0].plot(times,[x['goal_distance_m'] for x in rows],label=r['condition'],color={'A0':'tab:blue','A1':'tab:orange','A2':'tab:green'}[r['condition']])
        axes[i,1].plot(times,[tilt(x['post_root_state']) for x in rows],label=r['condition'],color={'A0':'tab:blue','A1':'tab:orange','A2':'tab:green'}[r['condition']])
    for i,seed in enumerate(SEEDS):
        for j in range(2):axes[i,j].set_title(f'{seed}: '+('Goal remaining (m)' if j==0 else 'Tilt (deg)'));axes[i,j].grid(alpha=.2);axes[i,j].legend();axes[i,j].set_xlabel('Simulated seconds')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for r in runs:
        if r['state']!='completed':continue
        folder=root/r['folder']/'sim-output';i=SEEDS.index(r['seed'])
        rows=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
        for file in sorted(folder.glob('raw-dem-*.npz'))[::4]:
            with np.load(file,allow_pickle=False) as data:
                points=data['ray_hits_w'].reshape(-1,3);points=points[np.isfinite(points).all(axis=1)]
            axes[i].scatter(points[:,0],points[:,1],c=points[:,2],cmap='Greys',s=2,alpha=.1,vmin=0,vmax=1)
        xy=np.asarray([row['root_state'][:2] for row in rows]+[rows[-1]['post_root_state'][:2]])
        axes[i].plot(xy[:,0],xy[:,1],label=r['condition'],color={'A0':'tab:blue','A1':'tab:orange','A2':'tab:green'}[r['condition']])
        start=read(folder/'startup-ready.json');goal=start['goal_world_xy']
        axes[i].scatter(*goal,marker='*',s=100,c='gold',edgecolors='black')
    for i,seed in enumerate(SEEDS):axes[i].set_title(str(seed));axes[i].axis('equal');axes[i].set_xlabel('world x (m)');axes[i].set_ylabel('world y (m)');axes[i].legend();axes[i].grid(alpha=.2)
    fig.suptitle('Observed routes and sampled terrain; unchosen paths remain unverified')
    fig.tight_layout();fig.savefig(root/'trajectories.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
