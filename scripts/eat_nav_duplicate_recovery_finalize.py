"""Render fixed diagnostic plots, immutable hash manifest, and preflight report."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape

COLORS={'nominal':'#444444','slight_left':'#1f77b4','slight_right':'#ff7f0e','strong_left':'#2ca02c','strong_right':'#d62728'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def atomic(path,contents):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(contents);tmp.replace(path)


def svg(title,body,w=850,h=300):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><text x="30" y="32" font-size="18" font-family="sans-serif">{escape(title)}</text>{body}</svg>\n'


def bars(path,title,labels,values,colors=None,ymax=None):
    w=850;h=300;left=70;right=35;top=60;bottom=58
    ymax=ymax or max(max(values)*1.18,0.001)
    body=f'<line x1="{left}" y1="{h-bottom}" x2="{w-right}" y2="{h-bottom}" stroke="#333"/>'
    for i,(label,value) in enumerate(zip(labels,values)):
        x=left+(i+.5)*(w-left-right)/len(values)
        height=value/ymax*(h-top-bottom)
        color=colors[i] if colors else '#277da1'
        body+=f'<rect x="{x-30:.1f}" y="{h-bottom-height:.1f}" width="60" height="{height:.1f}" fill="{color}"/>'
        body+=f'<text x="{x:.1f}" y="{h-bottom-height-7:.1f}" text-anchor="middle" font-size="13" font-family="sans-serif">{value:.5f}</text>'
        body+=f'<text x="{x:.1f}" y="{h-bottom+20}" text-anchor="middle" font-size="12" font-family="sans-serif">{escape(label)}</text>'
    body+=f'<text x="12" y="{top+10}" font-size="12" font-family="sans-serif">m RMS</text>'
    atomic(path,svg(title,body,w,h))


def text_panel(path,title,lines):
    body=''.join(f'<text x="35" y="{80+i*28}" font-size="15" font-family="sans-serif">{escape(line)}</text>' for i,line in enumerate(lines))
    atomic(path,svg(title,body))


def command_plot(path,rows):
    w=1600;h=760;body=''
    for ri,row in enumerate(rows):
        for ci,(field,ix) in enumerate((('linear x',0),('linear y',1),('angular z',2))):
            x0=70+ci*500;y0=85+ri*220;pw=420;ph=150
            body+=f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="#fafafa" stroke="#bbb"/>'
            body+=f'<text x="{x0}" y="{y0-8}" font-size="13" font-family="sans-serif">state {row["state_index"]}: {field}</text>'
            samples=[case['command_series'][k][ix] for case in row['cases'].values() for k in range(40)]
            low=min(samples);high=max(samples);span=max(high-low,0.01);low-=.08*span;high+=.08*span
            for name,case in row['cases'].items():
                values=[r[ix] for r in case['command_series']]
                points=' '.join(f'{x0+k*pw/39:.1f},{y0+ph-(v-low)/(high-low)*ph:.1f}' for k,v in enumerate(values))
                body+=f'<polyline points="{points}" fill="none" stroke="{COLORS[name]}" stroke-width="1.6"/>'
            body+=f'<text x="{x0+5}" y="{y0+13}" font-size="10" font-family="sans-serif">{high:.2f}</text>'
            body+=f'<text x="{x0+5}" y="{y0+ph-5}" font-size="10" font-family="sans-serif">{low:.2f}</text>'
            if len(row['cases'])<5:
                body+=f'<text x="{x0+90}" y="{y0+ph-12}" font-size="12" fill="#b22222" font-family="sans-serif">candidate runs unavailable: follower timeout</text>'
    for i,(name,color) in enumerate(COLORS.items()):
        x=70+i*200
        body+=f'<line x1="{x}" y1="740" x2="{x+25}" y2="740" stroke="{color}" stroke-width="3"/><text x="{x+31}" y="744" font-size="12" font-family="sans-serif">{name}</text>'
    atomic(path,svg('40-step follower commands on three frozen states',body,w,h))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    smoke=json.loads((r/'three-state-maneuver-smoke.json').read_text())
    if smoke['state_count']!=3:raise ValueError('three-state smoke incomplete')
    baseline=json.loads((r/'baseline-metric-audit.json').read_text())
    gate=json.loads((r/'duplicate-gate.json').read_text())
    divergence=json.loads((r/'first-divergence-report.json').read_text())
    good=json.loads((r/'known-good-comparison.json').read_text())
    charts=r/'plots';charts.mkdir(exist_ok=True)
    bars(charts/'known-good-current-duplicate-rms.svg','Duplicate XY RMS: prior / failed / recovered',
         ['prior good','failed snapshot','fresh prefix'],[.003500814,baseline['recomputed_true_xy_rms_m'],gate['fresh_process_duplicate_xy_rms_m']],['#2ca02c','#d62728','#1f77b4'])
    bars(charts/'snapshot-vs-fresh.svg','Failed snapshot restore vs fresh-process prefix',
         ['snapshot restore','fresh prefix'],[baseline['recomputed_true_xy_rms_m'],gate['fresh_process_duplicate_xy_rms_m']],['#d62728','#2ca02c'])
    if divergence['earliest']:
        text_panel(charts/'first-divergence.svg','First divergence', [f'{x[1]}: control step {x[0]}' for x in divergence['earliest']])
    else:
        text_panel(charts/'first-divergence.svg','First divergence', ['None across 40 control steps in three fresh-process nominal repeats.','Command, observations, Actor, dynamics, contact, DEM, Critic and reward matched.'])
    control=json.loads((r/'recorded-command-control.json').read_text())
    text_panel(charts/'live-vs-recorded.svg','Live follower vs recorded-command control',
               ['Live follower duplicate XY RMS: %.5f m'%gate['fresh_process_duplicate_xy_rms_m'],
                'Recorded-command control: %s'%control['status'],
                'Conditional control is required only when fresh live duplicate exceeds 0.0105 m.'])
    labels=[];vals=[];colors=[]
    for row in smoke['rows']:
        if row.get('status')=='measurement_invalid':continue
        labels.append(f'{row["state_index"]} dup');vals.append(row['duplicate_xy_rms_m']);colors.append('#444444')
        for name in ('slight_left','slight_right','strong_left','strong_right'):
            labels.append(f'{row["state_index"]} {name.replace("slight_","s-").replace("strong_","S-")}')
            vals.append(row['cases'][name]['xy_effect_rms_m']);colors.append(COLORS[name])
    bars(charts/'duplicate-vs-candidate.svg','Duplicate vs candidate effect (state 42: NA after timeout)',labels,vals,colors)
    command_plot(charts/'three-state-commands.svg',smoke['rows'])
    stop=[]
    for row in smoke['rows']:
        if row.get('status')=='measurement_invalid':
            stop.append(f'state {row["state_index"]}: follower first-command timeout; third nominal repeat and candidates unavailable')
            continue
        if row['all_commands_substantially_identical']:stop.append(f'state {row["state_index"]}: all command series identical')
        if row['all_candidate_effect_at_or_below_duplicate']:stop.append(f'state {row["state_index"]}: candidate effect <= duplicate')
        for name,case in row['cases'].items():
            if case['effect_comparable_steps']<40:stop.append(f'state {row["state_index"]} {name}: terminated before 40 steps; effect uses {case["effect_comparable_steps"]} common steps')
    marker={'state':'stopped_after_follower_timeout','duplicate_gate':gate['decision'],'smoke_states':smoke['completed_state_indices'],
            'stop_conditions':stop,'formal_rollout_started':False,'next_step':'report_and_request_approval',
            'recorded_command_control':control['status']}
    atomic(r/'status.json',json.dumps(marker,indent=2,sort_keys=True)+'\n')
    lines=['# EAT-Nav feasible-set duplicate recovery','',
        '## 判定','',
        f'- Fresh-process duplicate: **{gate["fresh_process_duplicate_xy_rms_m"]:.6f} m XY RMS**, gate 0.0105 m: **{gate["decision"]}**。',
        f'- 旧snapshot restore失敗: 報告値0.023661はx/y/yaw混合。訂正XY RMSは**{baseline["recomputed_true_xy_rms_m"]:.6f} m**で、gate不合格は変わらない。',
        '- 3 runでanchorのcritical fieldと固定Path、40-step command・観測・Actor・物理・接触・DEM・Critic・rewardが一致。first-divergenceは検出されなかった。',
        '- recorded-command controlは条件未成立のため実施していない。',
        '- 旧snapshot失敗traceではpre-noise Actor observationがstep 0、commandがstep 1で初めて分岐。fresh-processでは全signalで分岐なし。差異の領域は同一processの不完全なsnapshot restoreに絞られるが、個別の隠れ状態は未特定。',
        '- 旧runのfull anchor snapshotとmap-frame Path JSONは元runで保存されておらず復元不可。元trace・base Path・source/logは凍結済み。',
        '- 収集器のcommand rowは再サンプル時にpost-step値となる。state 13の照合をpost-stepへ修正し、記録command/DEM/root一致を確認した。',
        '- state 42のnominal 3回目は初回follower command取得でfresh=false。8 followerは生存・全Path subscriber=1だが2秒間command sequenceは全0。候補は未実行で測定無効。',
        '- 56 train + 20 validationのformal rolloutは開始していない。',
        '', '## 固定3-state maneuver smoke','',
        '| train state | terrain | duplicate XY RMS m | max candidate XY effect RMS m | first command divergence | max rho |',
        '|---:|---|---:|---:|---:|---:|']
    for row in smoke['rows']:
        if row.get('status')=='measurement_invalid':
            lines.append(f'| {row["state_index"]} | {row["state"]["terrain_type"]} | NA (2成功+1失敗) | NA | NA | NA |')
            continue
        c=[v for n,v in row['cases'].items() if n!='nominal']
        first=min((v['command_divergence']['first_step'] for v in c if v['command_divergence']['first_step'] is not None),default=None)
        rho=max((v['rho'] for v in c if v['rho'] is not None),default=None)
        lines.append(f'| {row["state_index"]} | {row["state"]["terrain_type"]} | {row["duplicate_xy_rms_m"]:.6f} | {max(v["xy_effect_rms_m"] for v in c):.6f} | {first if first is not None else "none"} | {rho if rho is not None else "NA"} |')
    lines+=['','state 19/13はnominal 3回と固定5 Pathをfresh processで測定。state 42はnominal 2回成功（相互差0）、3回目が初回command timeoutとなり候補を開始していない。ΔV、A_H、D、R_recover、termination、progress、body tilt、contact、proposal OOD、利用可能な40-step commandとXY軌跡は[JSON](three-state-maneuver-smoke.json)に保存。',
            '候補間の統計的優位性や他terrainへの一般化はこの少数stateから主張しない。',
            '','## 成果物','',
            '- [凍結失敗case](failure-baseline.json)、[既知良好との差分](known-good-comparison.json)、[RMS定義監査](baseline-metric-audit.json)',
            '- [anchor比較](prefix-replay-anchor-comparison.json)、[duplicate](prefix-replay-duplicates.json)、[first-divergence](first-divergence-report.json)',
            '- [gate](duplicate-gate.json)、[maneuver smoke](three-state-maneuver-smoke.json)、[state 42 interruption](state42-interruption.json)、[hashes](artifact-hashes.json)',
            '- [図](plots/)：prior/current RMS、first-divergence、snapshot/fresh、live/recorded、duplicate/candidate、3-state commands',
            '','## 次の判断','',
            'state 42のfollower first-command timeoutを未解決として停止。formal 76 state rolloutも開始していない。再診断・続行方法は結果レビュー後に判断する。']
    if stop:lines+=['','## Stop conditions','']+[f'- {x}' for x in stop]
    atomic(r/'duplicate-recovery-report.md','\n'.join(lines)+'\n')
    files=[]
    for path in r.rglob('*'):
        if path.is_file() and path.name not in ('artifact-hashes.json','runner.log') and not path.name.endswith('.tmp'):
            files.append({'path':str(path.relative_to(r)),'sha256':sha(path),'bytes':path.stat().st_size})
    atomic(r/'artifact-hashes.json',json.dumps({'files':sorted(files,key=lambda x:x['path']),
          'runner_log_sha256':sha(r/'runner.log'),'source_hashes':json.loads((r/'source-provenance.json').read_text())},indent=2,sort_keys=True)+'\n')
    print('RECOVERY_FINALIZED',len(files),'artifacts',flush=True)

if __name__=='__main__':main()
