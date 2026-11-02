# 14候補CPU推論の最小例

リポジトリrootから実行する。

    PYTHONPATH=src:scripts python scripts/eat_nav_moving_pose2s_predict.py --checkpoint docs/eat-nav/moving-state-pose2s-20261103/models/M_move/checkpoint.pt --input examples/eat_nav_moving_pose2s/example-input.npz --output /tmp/eatnav-pose2s-predictions.json --device cpu

example-input.npzはtrain由来の走行中観測1件に固定14候補を与えた入力。dem、actor/body、候補、方向、有効mask、現在quaternion、現在とDEMの整数nanosecond時刻、input schemaだけを含み、未来GTは含まない。元データの対応はexample-source.json。実機topic購読や速度指令の送信はしない。

出力JSONのdelta_position_body_m、delta_quaternion_wxyz、relative_rpy_rad、future_tilt_rad、event_probabilityを確認できる。horizon_sec=2.0。身体状態の単位/scale/順序と相対位置の全車体座標は[成果物README](../../docs/eat-nav/moving-state-pose2s-20261103/README.md)と承認仕様を参照する。

標準ライブラリだけで14候補を1行ずつ表示する例（候補番号は入力と同じ順序）：

```bash
python - <<'PYTHON'
import json, math
with open('/tmp/eatnav-pose2s-predictions.json') as f:
    result = json.load(f)
assert result['horizon_sec'] == 2.0
print('events:', result['event_names'])
for i, (xyz, rpy, tilt, events) in enumerate(zip(
        result['delta_position_body_m'][0], result['relative_rpy_rad'][0],
        result['future_tilt_rad'][0], result['event_probability'][0])):
    print(i, 'xyz[m]=', [round(v, 3) for v in xyz],
          'relative roll/pitch/yaw[deg]=', [round(math.degrees(v), 1) for v in rpy],
          'future tilt[deg]=', round(math.degrees(tilt), 1),
          '2s events=', [round(v, 4) for v in events],
          'rotation_valid=', result['validity']['rotation_valid'][0][i])
PYTHON
```

現在quaternionを与えた場合の絶対未来姿勢は`future_quaternion_wxyz`／`future_rpy_rad`にある。上の一覧のRPYは現在からの**相対**回転であり、重力基準の未来RPYとは異なる。推論例は動作確認用のtrain入力で、未学習データの精度評価ではない。
