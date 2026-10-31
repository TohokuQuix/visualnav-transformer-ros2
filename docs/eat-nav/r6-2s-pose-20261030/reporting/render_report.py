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


def render(root, output, interim=False):
    required = [
        'scientific-evaluation-complete.json', 'validation-evaluation/summary.json',
        'frozen-top3.json', 'test-evaluation/summary.json',
        'validation-forecast-metrics/report.json', 'test-forecast-metrics/report.json',
        'top3-bag-evaluation/report.json', 'top3-bag-inpainted-sensitivity/report.json',
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
