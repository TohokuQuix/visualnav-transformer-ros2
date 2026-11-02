# 14候補CPU推論の最小例

リポジトリrootから実行する。

    PYTHONPATH=src:scripts python scripts/eat_nav_moving_pose2s_predict.py --checkpoint docs/eat-nav/moving-state-pose2s-20261103/models/M_move/checkpoint.pt --input examples/eat_nav_moving_pose2s/example-input.npz --output /tmp/eatnav-pose2s-predictions.json --device cpu

example-input.npzはtrain由来の走行中観測1件に固定14候補を与えた入力。dem、actor/body、候補、方向、有効mask、現在quaternion、現在とDEMの整数nanosecond時刻、input schemaだけを含み、未来GTは含まない。元データの対応はexample-source.json。実機topic購読や速度指令の送信はしない。

出力JSONのdelta_position_body_m、delta_quaternion_wxyz、relative_rpy_rad、future_tilt_rad、event_probabilityを確認できる。horizon_sec=2.0。身体状態の単位/scale/順序と相対位置の全車体座標は[成果物README](../../docs/eat-nav/moving-state-pose2s-20261103/README.md)と承認仕様を参照する。
