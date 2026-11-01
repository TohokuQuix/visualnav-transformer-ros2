"""Render saved R6 evidence; never fit thresholds or execute a robot.

Default mode requires complete validation/test/forecast/bag artifacts. --interim
labels every page as incomplete. Run from the repository root with PYTHONPATH=src.
"""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def number(value, digits=3):
    return '未確立' if value is None else f'{value:.{digits}f}'


def table(headers, rows):
    def line(row):
        return '| ' + ' | '.join(str(x).replace('|', '/') for x in row) + ' |'
    return '\n'.join([line(headers), line(['---'] * len(headers)), *map(line, rows)])


def diagnostic_rows(summary, ledger, jobs):
    """Use exactly the queue's common initial-valid denominator.

    all_candidates_excluded_decisions is a flag at each 2s decision boundary,
    not a duration. Actual protective STOP seconds are reported separately.
    """
    if not summary.get('all_jobs_completed'):
        raise ValueError('complete queue required for final diagnostic table')
    by_condition = {j['condition']['id']: {} for j in jobs}
    if not by_condition:
        raise ValueError('nonempty queue required')
    for job in jobs:
        record = ledger[job['job_id']]
        if record['state'] != 'completed':
            raise ValueError('incomplete job in final diagnostic table')
        target = by_condition[job['condition']['id']]
        for row in record['results']:
            if row['group_id'] in target:
                raise ValueError('duplicate cohort within condition')
            target[row['group_id']] = row
    common = set.intersection(*(set(rows) for rows in by_condition.values()))
    common = {gid for gid in common if all(rows[gid]['initial_valid'] for rows in by_condition.values())}
    if not common or len(common) != summary['matched_initial_valid_trials']:
        raise ValueError('diagnostic denominator differs from frozen queue summary')
    output = []
    for name, values in sorted(by_condition.items()):
        rows = [values[gid] for gid in sorted(common)]
        n = len(rows)
        processing = sum(r['processing_sec'] for r in rows)
        decisions = sum(r['decision_count'] for r in rows)
        excluded = sum(r['all_candidates_excluded_decisions'] for r in rows)
        output.append([name, n, sum(r['near_fall_count'] for r in rows),
            sum(bool(r['intervention']) for r in rows), sum(bool(r['timeout']) for r in rows),
            number(processing/n), number(processing/decisions if decisions else None),
            excluded, number(excluded/decisions if decisions else None),
            sum(r['unknown_stall_windows'] for r in rows)])
    return output


def render(root, output, interim=False):
    required = [
        'scientific-evaluation-complete.json', 'validation-evaluation/summary.json',
        'frozen-top3.json', 'test-evaluation/summary.json',
        'validation-forecast-metrics/report.json', 'test-forecast-metrics/report.json',
        'top3-bag-evaluation/report.json', 'top3-bag-inpainted-sensitivity/report.json',
        'validation-posture-statistics/statistics.json',
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing and not interim:
        raise ValueError('Full scientific results required: ' + ', '.join(missing))
    output.mkdir(parents=True, exist_ok=True)
    sections = ['# R6 2秒イベント・未来姿勢予測の結果' + ('（途中経過）' if interim else ''),
                '実ロボットへの指令送信は行っていません。' +
                (' 本書は未完了の途中報告です。未実施の結果を0件や成功として補っていません。' if interim else '')]
    if missing:
        sections += ['未完了の必須成果物:\n\n' + '\n'.join('- ' + name for name in missing)]
    profiles = read(root / 'production-profiles/index.json')['profiles']
    actual_conditions = set()
    if (root / 'validation-evaluation/frozen-binding.json').exists():
        actual_conditions = {x['id'] for x in read(root / 'validation-evaluation/frozen-binding.json')['conditions']}
    sections += ['## 入力・出力と学習',
        '入力は4時刻×2層×20×20のDEM、機体状態28次元、前後7本ずつの14候補です。'
        'DEMの物理範囲は2m四方、解像度0.1mです。出力は候補ごとの今後2秒以内の4イベントと、'
        '0.1～2.0秒の20時刻の相対XYZ・yaw差・重力基準roll/pitchです。角度はsin/cos表現です。'
        '旧10秒イベントや旧進捗headを、新しい2秒予測として扱っていません。',
        '979組（train519、validation460）、13,706本の実行済み候補走行を使用しました。'
        '同じ初期状態の14候補や同じparentを共有するため、独立した事故件数ではありません。'
        '収集はheadlessで88ジョブ、学習はGUI OFF・CUDA mini-batchの1プロセスです。'
        '既存の低レベルRL Actorは固定しています。']
    accuracy = []
    for name, directory in [('DEM＋状態', 'final-model'), ('状態のみ', 'state-only-diagnostic')]:
        metrics = read(root / directory / 'metrics.json')['validation']['trajectory']
        accuracy.append([name, number(metrics['ade_3d_m']), number(metrics['fde_3d_m']),
                         number(metrics['rotation_geodesic_deg']), number(metrics['max_tilt_mae_deg'])])
    sections += [table(['モデル', 'ADE m', 'FDE m', '回転誤差°', '最大傾斜誤差°'], accuracy),
        '状態のみの比較に対するDEMの精度改善は確立していません。以下はvalidationのイベント指標です。'
        '陽性のないheadのPR-AUC・再現率は未確立です。class-weight付き学習のsigmoidを、校正済みの実発生確率とは断定しません。']
    events = read(root / 'final-model/metrics.json')['validation']['events']
    sections += [table(['head', '既知候補数', '陽性数', 'AP', '再現率', 'FPR', 'Brier', 'ECE'],
        [[name, e['count'], e['positive'], number(e['average_precision']), number(e['recall']),
          number(e['false_positive_rate']), number(e['brier']), number(e['ece'])]
         for name, e in zip(['tip_over', 'near_fall', 'linear_stall', 'turn_stall'], events)])]
    if (root / 'event-group-support.json').exists():
        support = read(root / 'event-group-support.json')['roles']
        sections += [table(['split', 'head', '陽性候補', '陽性anchor組', '陽性episode組', '陽性parent seed数'],
            [[role, name, x['positive_candidates'], x['positive_anchor_groups'],
              x['positive_episode_groups'], x['positive_parent_seeds']]
             for role, heads in support.items() for name, x in heads.items()]),
            '転倒のtrain陽性19候補は4つのanchor組・3つのepisode組に集中しています。'
            '陽性候補数を独立した事故数や十分な汎化能力の根拠として扱いません。']
    if (root / 'validation-posture-statistics/statistics.json').exists():
        posture = read(root / 'validation-posture-statistics/statistics.json')
        observed = root / 'final-dataset-audit/validation-records.json'
        if hashlib.sha256(observed.read_bytes()).hexdigest() != posture['source_sha256']:
            raise ValueError('posture statistics differ from actual validation source')
        distributions = posture['overall']['signed_posture_and_rate_distributions']
        sections += ['## 符号付き姿勢・速度別の観測統計',
            '以下は実際の2秒候補走行の記述統計で、転倒予測の成功や2m段差走破の保証ではありません。'
            '全35 field/levelの分布、正負の傾斜持続時間、角度変化速度と区間はvalidation-posture-statistics/に保存しています。',
            table(['2秒の観測結果', '候補数', 'pitch最小の中央値°', 'pitch最大の中央値°',
                   'roll最小の中央値°', 'roll最大の中央値°', '最大pitch変化速度の中央値°/s', '最大roll変化速度の中央値°/s'],
                [[label, values['count'], *[number(values['quantiles'][key]['0.5']) if values['count'] else '未確立'
                  for key in ('pitch_min_deg','pitch_max_deg','roll_min_deg','roll_max_deg','max_pitch_rate_degps','max_roll_rate_degps')]]
                 for label, values in distributions.items() if label != 'all']),
            table(['初期平面速度bin', '候補数', 'イベント陽性数', '陽性率', 'Wilson95%'],
                [[key, values['branch_count'], values['positive_branch_count'], number(values['branch_event_rate'],4),
                  '–'.join(number(x,4) for x in values['branch_event_wilson95'])]
                 for key, values in sorted(posture['by_speed_bin'].items())]),
            '初期平面速度はbody座標のsqrt(vx²+vy²)[m/s]です。四分位の区切りは' +
            ', '.join(number(x,6) for x in posture['speed_bin_cutpoints_mps']) +
            'で、境界値は上側のbinに含めます。候補・anchorはepisode groupを共有し、区間は独立事故の安全性証明ではありません。'
            '速度と失敗の因果関係もこの分布だけでは確定しません。']
    if (root / 'anchor-leadtime/report.json').exists():
        lead = read(root / 'anchor-leadtime/report.json')
        before = [v for k, v in lead['summaries'].items() if k.endswith('/before_observed_contact')]
        sections += ['## 接触前の検知余裕',
            table(['head', '接触前の陽性候補', '警告した陽性候補'],
                  [[h, sum(v['heads'][h]['positive_candidates'] for v in before),
                    sum(v['heads'][h]['alerted_positive_candidates'] for v in before)]
                   for h in ['tip_over', 'near_fall', 'linear_stall', 'turn_stall']]),
            'スタックの2秒という検知余裕は2秒windowの成立確認までの時間であり、物理的なスタック開始までの時間ではありません。'
            '接触前の直線スタック陽性がなく、接触前の直線スタック予測の成功は示せません。'
            '接触の有無が完全観測されていない既存traceは不明として分離しています。'
            '場面別の秒・距離分位点はanchor-leadtime/report.jsonに記録しています。']
    sections += ['## validation由来の閾値候補',
        '角度・段差の成功／失敗混合分位点から12設定を作り、走行結果を見る前に10設定を選びました。'
        '角度誤差のマージンもvalidation由来です。段差は短時間event-freeの統計に基づく候補であり、'
        'この高さまで2m走破できると保証する値ではありません。踏面0.2m・同方向の段差2段以上を階段として区別します。'
        '停止余裕0.06mは幾何学的paddingで、実機の制動距離を実測した値ではありません。',
        table(['設定', 'pitch°', 'roll°', '基本段差cm', '階段cm', '壁cm', '角度マージン°', '比較対象'],
            [[p['id'], number(p['pitch_deg']), number(p['roll_deg']), number(p['step_cm']),
              number(p['conditional_step_cm'].get('stairs')), number(p['conditional_step_cm'].get('wall')),
              number(p['tilt_margin_deg']), '対象' if p['id'] in actual_conditions else '対象外'] for p in profiles])]
    sections += ['## 閉ループ評価の共通条件',
        '全7field×level0～4で、開始時yawの前方2mに固定goal、制限60秒、到達半径0.2mです。'
        '通常30試行を計画し、開始前に姿勢が崩れやすかったK-Rail level4だけ90試行を事前固定しました。'
        '開始無効は全条件で同じgroupを除き、実際の共通分母を示します。headless・30環境並列です。'
        '完全に同じ候補を全判断で選ぶ設定だけ、全batchで同一性を検証して実走行を共有します。'
        'これは独立した再実験ではありません。経路が分かれた設定は実走行します。',
        'route進捗は開始点→固定goalの線への射影を0～2mに制限した符号付き変化の合計です。'
        '旧R4/R5のwaypoint更新型route進捗と直接比較できません。'
        'スタックは2秒window・stride2秒で重複なし、陽性秒=陽性window×2です。'
        '実際の保護STOPとgoal到達後のholdは分離します。'
        '途中打切り後の残り秒は順位付けでSTOP上限へ加えます。転倒未観測の打切りは転倒結果不明です。'
        '順位は転倒上限（観測転倒＋不明試行）、STOP上限、validation固定尺度で等重み正規化した進捗・両スタックwindow/秒の順です。']
    for role in ['validation', 'test']:
        p = root / f'{role}-evaluation/summary.json'
        if not p.exists():
            continue
        summary = read(p)
        if not summary['all_jobs_completed']:
            sections += [f'## {role}：未完了', f"完了ジョブ {summary['completed_jobs']} / {summary['planned_jobs']}。最終順位は未確定です。"]
            continue
        sections += [f'## {role}の走行結果',
            f"各条件の計画試行数{summary['planned_cohort_trials']}、共通の開始有効試行数{summary['matched_initial_valid_trials']}。",
            table(['条件', '有効数', '転倒', '転倒結果不明', '転倒率CI95%', '保護STOP秒/走行', '順位STOP上限秒/走行',
                   'route進捗m', '正味接近m', '移動距離m', '到達数', '直線window計', '旋回window計', '直線秒/走行', '旋回秒/走行'],
                [[name, a['trials'], a['tip_count'], a['fall_outcome_unknown_trials'],
                  '–'.join(number(x) for x in a['tip_rate_wilson95']), number(a['mean_protective_stop_sec']),
                  number(a['mean_ranking_STOP_upper_bound_sec']), number(a['mean_route_progress_m']),
                  number(a['mean_net_goal_approach_m']), number(a['mean_physical_distance_m']), a['goal_reached_count'],
                  round(a['mean_linear_windows']*a['trials']), round(a['mean_turn_windows']*a['trials']),
                  number(a['mean_linear_stall_sec']), number(a['mean_turn_stall_sec'])]
                 for name, a in sorted(summary['aggregates'].items())]),
            '転倒率区間は不明転倒結果を含む保守的な上下限です。標本数が小さいため、転倒0件でも確定的な安全性を意味しません。',
            'field/level別の全指標とmacro平均は別表およびsummary.jsonに保存しています。']
        cellrows = []
        for name, a in sorted(summary['aggregates'].items()):
            for cell, x in sorted(a['cells'].items()):
                cellrows.append([name, cell, x['trials'], x['tip_count'], x['fall_outcome_unknown_trials'],
                    number(x['mean_protective_stop_sec']), number(x['mean_route_progress_m']),
                    number(x['mean_linear_stall_sec']), number(x['mean_turn_stall_sec'])])
        (output / f'{role}-field-level.md').write_text(table(
            ['条件', 'field/level', '有効数', '転倒', '不明', 'STOP秒', '進捗m', '直線秒', '旋回秒'], cellrows)+'\n')
        sections += ['### field/levelを等重みにしたmacro平均',
            table(['条件', 'cell数', '転倒率macro', '保護STOP秒', '順位STOP上限秒', 'route進捗m',
                   '正味接近m', '移動距離m', '直線window/走行', '旋回window/走行', '直線秒/走行', '旋回秒/走行'],
                [[name, len(a['cells']), number(a['macro_tip_rate']),
                  *[number(a['macro_mean_'+key]) for key in ('protective_stop_sec',
                    'ranking_STOP_upper_bound_sec', 'route_progress_m', 'net_goal_approach_m',
                    'physical_distance_m', 'linear_windows', 'turn_windows', 'linear_stall_sec', 'turn_stall_sec')]]
                 for name, a in sorted(summary['aggregates'].items())]),
            'macroはfield/levelの各cellを等重みにした値です。転倒結果不明の件数・上限とCIは前表およびsummary.jsonで分けています。',
            '### 診断指標', table(['条件', '有効数', 'near-fall観測回数', '介入走行数', 'timeout走行数',
                '判断処理秒/走行', '判断処理秒/判断', '全候補除外判断数計', '全判断に対する除外比率', 'スタック不明window計'],
                diagnostic_rows(summary, read(root / f'{role}-evaluation/progress.json'),
                                read(root / f'{role}-evaluation/jobs.json'))),
            '処理時間は実シミュレーション出力で計測した判断処理時間です。同一policyの共有出力は元試行の計測値を共有し、独立した再計測ではありません。'
            '全候補除外は2秒ごとの判断時点の件数です。goal到達後のholdには判断を行いません。STOPの実時間は主表の保護STOP秒を参照してください。']
        forecast_path = root / f'{role}-forecast-metrics/report.json'
        if forecast_path.exists():
            forecasts = read(forecast_path)
            sections += ['### 実行された候補に対する未来予測精度',
                'この表は全て新しいR6予測器の精度です。旧参照の行も、旧plannerが実行した走行上でR6を採点しています。旧モデルに未来XYZ・姿勢headがあるという意味ではありません。'
                'STOPには実行されたbank候補がなく、教師から除きます。未実行の13候補へ真値を複製していません。'
                'policy同一性による共有出力も重複計上しません。選択された候補の分布に依存する診断で、全14候補の精度とは異なります。',
                table(['実走行の方式', 'window数', 'ADE m', 'FDE m', 'yaw誤差°', 'roll誤差°', 'pitch誤差°', '回転誤差°', '最大傾斜誤差°'],
                    [[name, x['windows'], *[number(x['trajectory'][key]) for key in
                      ('ade_3d_m', 'fde_3d_m', 'yaw_mae_deg', 'roll_mae_deg', 'pitch_mae_deg', 'rotation_geodesic_deg', 'max_tilt_mae_deg')]]
                     for name, x in sorted(forecasts['by_condition'].items())]),
                table(['実走行の方式', 'head', '既知数', '陽性数', 'AP', '再現率', 'FPR', 'Brier', 'ECE'],
                    [[name, head, e['count'], e['positive'], number(e['average_precision']), number(e['recall']),
                      number(e['false_positive_rate']), number(e['brier']), number(e['ece'])]
                     for name, x in sorted(forecasts['by_condition'].items())
                     for head, e in zip(('tip_over','near_fall','linear_stall','turn_stall'), x['events'])])]
    if (root / 'frozen-top3.json').exists():
        frozen = read(root / 'frozen-top3.json')
        sections += ['## validationで凍結した上位3設定',
            table(['順位', '設定', 'pitch°', 'roll°', '段差cm', '角度マージン°'],
                [[i, p['id'], number(p['pitch_deg'], 8), number(p['roll_deg'], 8),
                  number(p['step_cm'], 8), number(p['tilt_margin_deg'], 8)]
                 for i, p in enumerate(frozen['profiles'], 1)]),
            '条件付きの正確な値・選定尺度・順位・SHAはfrozen-top3.jsonに保存しています。test/bagに合わせた再調整は行いません。']
    if (root / 'top3-bag-evaluation/report.json').exists():
        bag = read(root / 'top3-bag-evaluation/report.json')
        sections += ['## 指定実機bagのoffline評価',
            '20261029_210926のセンサー列を使用し、初期TFの約2m前方に全設定共通の仮goalを置きました。'
            '生elevationが未観測のセルは、inpainted値があっても既知として扱いません。'
            '補間値を既知と仮定する別解析は参考値として分離しています。',
            table(['順位', '設定', '判定', '観測数', 'STOP', '全候補地図不足', '通過余裕あり候補', '継続pair'],
                [[s['rank'], s['profile_id'], s['classification'], s['decisions'], s['STOP_decisions'],
                  s['all_candidates_map_unknown_decisions'], s['clear_footprint_candidate_decisions'],
                  s['stable_clear_candidate_pairs']] for s in bag['summaries']]),
            '車体とフリッパの現在姿勢の占有範囲、validation位置誤差q95、観測範囲を調べています。'
            '将来のフリッパ動作は予測していません。実機encoder原点や製造寸法の独立実測は未完了です。'
            'bagの後続観測は選択経路に応じて変わらないため、閉ループ実機走行の成功とは言えません。']
    sections += ['## 再現・保管と検証',
        '学習済みcheckpointはcheckpoints/eat_nav/r6_2s_pose_20261030/、取得・load手順はverified-training/README.mdに保存しています。'
        '生走行・dataset・bagはGitに含めず、元PCの成果物ルートに保管します。'
        'dataset SHA256: `955a5ab0b5e2bfc3b49c620ab33c24c0f7f9824485e793ff5951e536f229fc67`。',
        '独立レビューのP1/P2を再現テストで修正し、修正後87テストがpassしました。'
        'test/bagはvalidationで凍結したモデル・コード・設定と一致しない場合に拒否します。'
        'Git pushとremote SHAの確認は別途最終転送記録で示します。途中報告の存在をpush完了とは扱いません。']
    text = '\n\n'.join(sections)+'\n'
    (output / 'report.md').write_text(text)
    (output / 'report.txt').write_text(text)
    sources = [root/name for name in required if (root/name).exists()]
    sources += [root/'final-model/metrics.json', root/'production-profiles/index.json']
    (output/'report-sources.json').write_text(json.dumps(dict(
        interim=interim, missing=missing,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--interim', action='store_true')
    a=p.parse_args()
    render(a.root,a.output,a.interim)
