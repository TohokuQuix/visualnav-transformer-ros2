# model2000による2秒後状態予測：実行結果

独立親試行 1330/1330、有効pose区間 6631。
学習 75epoch、採用epoch 50、strict合格 78/490。
strict精度達成: False。未達の場合は暫定モデルであり、目標達成とは扱いません。

## 予測誤差（全体平均、cm / deg）

- longitudinal_cm: 10.677230805690147
- lateral_cm: 5.078316732935731
- height_cm: 2.270656289731555
- roll_deg: 3.9127743127255084
- pitch_deg: 3.310252344107458
- yaw_deg: 8.146319716687522
- distance_3d_cm: 13.11754291107554

490組の平均・中央値・P95・最大値、欠測・基準未達は prediction-evaluation/ を参照。旧との比較は comparison.json と matched-segments-comparison.json。

## 条件B（探索的350走行）

```json
{
  "valid_runs": 350,
  "goal_reached_runs": 193,
  "tip_over_runs": 4,
  "mean_route_progress_m": 2.625676253426788,
  "mean_filter_stop_seconds": 0.00028571428571428574,
  "mean_follower_zero_command_seconds": 3.077142857142857,
  "linear_stall_windows": 36,
  "turn_stall_windows": 2,
  "mean_linear_stall_positive_seconds": 0.2057142857142857,
  "mean_turn_stall_positive_seconds": 0.011428571428571429,
  "mean_duration_seconds": 9.216000000000001
}
```

Aは未実施。B単独から傾斜除外の有効性を主張しません。実機互換性・実機走行は未検証です。

## 別PCでの取得と推論確認

```bash
git fetch origin codex/eatnav-model2000-pose2s-20261110
git switch --track origin/codex/eatnav-model2000-pose2s-20261110
# 初回checkout後、リポジトリの依存を満たすPython環境を使用
export PYTHONPATH="$PWD/src:$PWD/scripts"
sha256sum docs/eat-nav/model2000-pose2s-20261110/models/seed123/checkpoint.pt
python scripts/eat_nav_single_pose2s_predict.py \
  --checkpoint docs/eat-nav/model2000-pose2s-20261110/models/seed123/checkpoint.pt \
  --input docs/eat-nav/model2000-pose2s-20261110/example/input-fourteen.npz \
  --output /tmp/model2000-pose2s-predictions.json --all-fourteen
```

checkpoint SHA256: `78474a26145137ab12c7486c9b8dde72f2078e6fd61d861d6db58a07ccbb6125`。予測時間2秒、XYZ[m]は起点body座標、相対回転は6DからSO(3)/wxyz quaternion/RPYへ変換します。取得例は14候補の推論確認用であり、14候補すべての正解を含みません。

実行条件と入力schemaは承認仕様・model-index.json・normalization.jsonを参照。旧NNとは制御policy/profileが異なり、無条件互換ではありません。

## データ・検証制限

- RealSenseなし。4Hz・40×40 DEM一枚、実FAST-LIO（理想simセンサ）を維持。
- 旧dataset・旧予測重みは学習へ混入していません。地形は既知地形の独立走行評価です。
- raw/datasetはローカルのみ: `/home/ubuntu/eatnav-model2000-pose2s-data-20261110/production`。巨大データはGitへ保存していません。
- 関連テスト45件pass。全体pytestはROS/diffusers等の依存不足による28収集エラーがあり、test-limitations.txtに記録。
- 提供Actorの再取得: `python scripts/eat_nav_model2000_fetch_policy.py`。対応env/agentと組で使用。
- モデル単体/前処理込み14候補推論時間はinference-cpu.json・inference-cuda.json。
- 再収集/学習/評価: `PYTHONPATH=src:scripts python scripts/eat_nav_model2000_continue.py --root <new-root>`（manifest・承認・benchmarkを事前固定。既存partialを無断削除しない）。
