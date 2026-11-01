"""Descriptive validation statistics only; no model/threshold fitting or test/bag reads."""
from pathlib import Path
import json,hashlib,collections,math
import numpy as np
from visualnav_transformer.eat_nav.r6_policy import wilson
from eat_nav_r6_run import read,write,sha
r=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030');source=r/'final-dataset-audit/validation-records.json';records=read(source)
assert all(x['split']=='validation' for x in records)
complete=[x for x in records if x['complete']];assert len(complete)==6440
speed=lambda x:float(np.linalg.norm(np.asarray(x['initial_body_state'])[3:5]))
cuts=np.quantile([speed(x) for x in complete],[.25,.5,.75]);assert len(set(cuts))==3
metrics=('pitch_min_deg','pitch_max_deg','roll_min_deg','roll_max_deg','pitch_positive_duration_sec','pitch_negative_duration_sec','roll_positive_duration_sec','roll_negative_duration_sec','max_pitch_rate_degps','max_roll_rate_degps','step_cm')
quantiles=(0,.25,.5,.75,.95,1)
def describe(rows):
 n=len(rows);fail=sum(x['failed'] for x in rows);groups=collections.defaultdict(list)
 for x in rows:groups[x['group_id']].append(x)
 gfail=sum(any(x['failed'] for x in members) for members in groups.values())
 distributions={}
 for label, subset in [('all',rows),('event_free_2s',[x for x in rows if not x['failed']]),('event_positive_2s',[x for x in rows if x['failed']])]:
  distributions[label]=dict(count=len(subset),quantiles={key:{str(q):float(v) for q,v in zip(quantiles,np.quantile([x[key] for x in subset],quantiles))} for key in metrics} if subset else {})
 heads={}
 for i,name in enumerate(('tip_over','near_fall','linear_stall','turn_stall')):
  known=[x for x in rows if x['event_mask'][i]];positive=sum(x['event_targets'][i]==1 for x in known)
  heads[name]=dict(known=len(known),positive=positive,wilson95=wilson(positive,len(known)) if known else None)
 return dict(branch_count=n,episode_group_count=len(groups),positive_branch_count=fail,branch_event_rate=fail/n if n else None,branch_event_wilson95=wilson(fail,n) if n else None,groups_with_any_event=gfail,group_any_event_wilson95=wilson(gfail,len(groups)) if groups else None,events=heads,signed_posture_and_rate_distributions=distributions)
field=collections.defaultdict(list);speeds=collections.defaultdict(list);postures=collections.defaultdict(list)
for x in complete:
 field[f"{x['field']}/level{x['level']}"].append(x)
 speeds[str(int(np.searchsorted(cuts,speed(x),side='right')))].append(x)
 tilt=math.degrees(math.acos(float(np.clip(-x['initial_body_state'][2],-1,1))))
 lo=next((b for b in [85,60,45,30,15,0] if tilt>=b),0);postures[str(lo)].append(x)
result=dict(schema='eatnav_r6_validation_posture_statistics_v1',source=str(source),source_sha256=sha(source),scope='actual observed2s validation branches, not2mgoal climbing outcomes',speed_definition='initial body-frame planar speed sqrt(vx^2+vy^2), m/s; body_state indices3:5',speed_bin_cutpoints_mps=cuts.tolist(),speed_bins='0belowq25;1q25..q50;2q50..q75;3q75up; boundaries assigned to higher bin',initial_tilt_bins_lower_deg=[0,15,30,45,60,85],posture_definition='initial tilt acos(-bodygravity_z); signedfuturegravityroll/pitch from20Hzrawpose where available',overall=describe(complete),by_field_level={k:describe(v) for k,v in sorted(field.items())},by_speed_bin={k:describe(v) for k,v in sorted(speeds.items())},by_initial_tilt_bin={k:describe(v) for k,v in sorted(postures.items())},limitations=['Candidate branches and repeated anchors share sourcegroups; branchWilson intervals are descriptive, not independent-trial safety guarantees','Group-any-event CI estimates a different event: anypositive candidate among observed members, not per-run risk','No validationtip/near positives; zeroobservations do not establish those predictive heads','Tread/height metrics are centreline DEM observations; width/support and actual riserclimbing are not inferred from mesh z gaps','No thresholds, model weights, runtime sources or held-out data are modified'])
out=r/'validation-posture-statistics';out.mkdir()
write(out/'statistics.json',result)
lines=['# Validationの姿勢・速度別統計（2秒の観測候補）','','学習用の成功／失敗は2秒イベントの有無です。2mゴールへの到達・段差走破の成功率とは異なります。複数候補・anchorはepisode groupを共有しており、候補単位のWilson区間は独立事故の安全性証明ではありません。','','|field/level|観測候補数|episode group数|2秒イベント陽性候補数|陽性率|Wilson95%|','|---|---:|---:|---:|---:|---|']
for k,v in result['by_field_level'].items():lines.append(f"|{k}|{v['branch_count']}|{v['episode_group_count']}|{v['positive_branch_count']}|{v['branch_event_rate']:.4f}|{v['branch_event_wilson95']}|")
lines+=['','符号付きpitch/rollの最小・最大、正負の持続秒、最大変化速度、段差cmの分位点を、全体・field/level・初期速度四分位・初期傾斜区分ごとにJSONへ保存しました。各区分内で2秒イベント陽性／非陽性を分けています。速度の区切り[m/s]: '+str(cuts.tolist()),'','転倒・near-fallのvalidation陽性はありません。そのheadの予測成功・走破限界をこの統計から結論しません。testや実機bagは読み込んでいません。']
(out/'report.md').write_text('\n'.join(lines)+'\n');(out/'report.txt').write_text('\n'.join(lines)+'\n')
write(out/'provenance.json',dict(source_sha256=sha(source),script_sha256=sha(Path(__file__)),rows=len(records),complete_rows=len(complete),thresholds_changed=False,test_or_bag_read=False))
print(json.dumps(dict(output=str(out),records=len(records),cells=len(field),speed_cutpoints=cuts.tolist(),positive=result['overall']['positive_branch_count'],groups=result['overall']['episode_group_count'])))
