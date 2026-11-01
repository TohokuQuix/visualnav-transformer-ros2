# 完了済み実batchの追加監査コマンド

このファイルは2026-11-02の手動監査用コードの再現資料です。入力は完了済みvalidationの実シミュレーションのみ。走行・推論・モデル・閾値の変更やROS指令送信は行わず、既存記録の読取りと新規監査JSONの保存だけを行います。policy同一性の共有結果は実走行として再監査・二重計上しません。

```bash
export PYTHONPATH=src:scripts:tests/eat_nav:.
/home/ubuntu/venvs/isaaclab_py310/bin/python docs/eat-nav/r6-2s-pose-20261030/execution-records/completed-batch-verifier.py event_2s-k_rail_field-l1-batch00 old_reference-k_rail_field-l1-batch00
```

この例の監査JSONは既に保存済みなので、再実行時には既存出力を拒否します。既存監査を削除しないでください。今後の新しい完了batchには第1引数でjob ID、第2引数で同じfield/level/物理batchの既存実参照job IDを指定します。第2引数がなければpaired一致は未検証と記録します。初期無効の環境を成功に置換しません。初期配列の有限値と平均指標はinitial_validの試行について確認し、最終条件比較の分母には別途全条件共通の初期有効組を使用します。

検査: fullprotocol完了、copied-output SHA証拠、gzipと展開前traceのSHA、50ms時刻連続性、観測行数、DEM時刻の因果性、初期有効配列の有限値、指定参照との初期配列/物理ファイル一致。最終研究全体の完了・安全性・test独立性の証明をこの1batch監査だけで代替しません。

2026-11-02修正: `copied_artifact_audit.py`により、消費前に元copy証跡の全original（JSON/NPZ等）をSHA照合します。除去済みtraceはgzip解凍後のSHAを照合します。旧監査JSONのverifier SHAは当時の版を指し、書き換えていません。`retained-copy-integrity-20261102.json`が修正版による16完了batch・28,660 original・480解凍traceの再検証証拠です。元manifestとの整合性検証であり、manifest自体の独立した真正性の保証ではありません。
