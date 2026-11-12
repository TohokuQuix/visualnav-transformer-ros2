# 30秒収集・履歴なしNN：学習と比較結果

2026-11-13（日本時間）。収集・学習・評価・独立再計算が完了。精度要件の全条件達成を意味しない。

## 結論

停滞の参考指標では、学習用の該当区間は最初10秒の15件から全30秒の62件へ増え、割合も0.86%から1.20%へ上がった。件数約4.13倍、割合約1.40倍。成功データだけが増えた結果ではないが、依然として学習用全体の約1.2%であり少数である。

合意した40cm条件（同じM30の予測水平変位量−実測水平変位量>=0.40m）は、独立testで0.61%から0.78%へ増えた。ただし対応付き親試行bootstrapの差の95%区間は−0.138〜+0.466ポイントで0を含む。割合増加を明確に確認できたとは断言しない。20〜30秒部分は1.20%だった。

新NNは同じ新testで旧NNより誤差が減った。先頭10秒の3D位置平均誤差は13.12→11.73cm。ただしScene1の到達率、実機性能、進行阻害不足が主因だったかは未検証。

## 収集・学習

- 全7field×level0〜4、親試行1330（train350/validation490/test490）、予定19950区間。
- 有効19546（train5148/validation7217/test7181）。欠測404：2秒前の終了67、終端後337。ゼロ埋めして分母に入れていない。
- 旧10秒データの有効6631区間に対して約2.95倍。新学習へ旧データ・旧NN重みは混ぜていない。
- 30台＋残り8台で各条件38親試行を収集。smoke/capacityは除外。GUI off、物理dt.005、制御.05、DEM4Hz1frame、実際のFAST-LIO処理、model_2000 Actor一式を維持。
- 元87次元NN、履歴なし。seed123、最大200/min30/patience25の同じ学習規則。44epochで終了、validation最良19epochを採用。旧学習は75epoch/最良50であり、実際の更新数は揃えていない。
- 31地形hashがroleを跨ぐ既知地形の独立走行評価。親試行・seed・raw・初期状態のrole重複なし。未知地形性能とは主張しない。

## datasetから見た停滞の参考指標

定義：2秒間の平均|要求vx|>=0.1m/s、かつ実測worldXY純変位<5cm。これは40cm予測誤差条件とは別の記述的指標であり、学習重みや除外処理には使用していない。連続停滞も2秒区間ごとに数える。

| 対象 | 先頭10秒の該当数/有効区間 | 割合 | 全30秒の該当数/有効区間 | 割合 |
|---|---:|---:|---:|---:|
|学習用|15/1746|0.86%|62/5148|1.20%|
|validation|17/2443|0.70%|88/7217|1.22%|
|最終test|18/2442|0.74%|103/7181|1.43%|
|全分割|50/6631|0.75%|253/19546|1.29%|

## 同じM30による40cm条件の比較

worldXYの純変位の大きさ同士の差。3D終点誤差や累積経路長ではない。実行候補1本の完全2秒区間だけを分母とし、未実行13候補は採点しない。

| 時間帯 | 該当数 | 有効予測区間数 | 割合 | 独立親試行数 |
|---|---:|---:|---:|---:|
|first10|15|2442|0.61%|490|
|full30|56|7181|0.78%|490|
|seconds10to20|13|2405|0.54%|485|
|seconds20to30|28|2334|1.20%|474|

先頭10秒は全30秒の一部であり、独立群とみなさない。後半まで生存・記録できた試行に偏る点もある。割合は件数合計の比で算出。

## 同一の新testに対する旧M10/新M30の平均絶対誤差

| 指標 | M10 先頭10秒 | M30 先頭10秒 | M10 全30秒 | M30 全30秒 |
|---|---:|---:|---:|---:|
|前後cm|10.681|9.831|21.723|10.453|
|左右cm|5.083|4.150|13.420|4.458|
|高さcm|2.276|1.897|8.440|2.162|
|3D位置cm|13.122|11.726|29.497|12.574|
|roll度|3.911|3.843|6.040|4.520|
|pitch度|3.310|3.221|3.959|3.575|
|yaw度|8.130|7.073|10.253|7.441|

各モデル自身のtrain正規化を維持。同じ2442先頭区間・7181全区間を比較した。M30全30秒の3D位置中央値10.01cm、P95 30.77cm。490組（候補14×field7×level5）は全組評価できたが、元の組別最大値基準で合格23/490、未達467/490。全体平均だけで精度達成としない。高さに追加合否閾値は設けていない。

## 重要な制約：FAST-LIO入力の推定発散

後半の一部でFAST-LIO pose差分由来速度に大きな外れ値がある。最終testで入力速度>10m/sは174/7181区間、最大219.94m/s。同じtestのシミュレータ真値速度の最大は1.78m/s。入力に未来は入っていないが、時刻・有限性監査を通ってもオドメトリ推定の正しさは保証されない。

M10の全30秒誤差はこの影響を強く受け、3D誤差>5mが45区間、最大79.31mだった。M30は全30秒の入力分布も学習しているため、29.50→12.57cmという平均改善すべてを「進行阻害例が増えた効果」と解釈できない。先頭10秒の13.12→11.73cmという比較も併記した。異常入力を隠すための除外・クリップ・正規化変更は行っていない。

FAST-LIO品質の点検は今後の課題であり、今回その修正や追加学習を自動実行していない。進行阻害例不足と入力推定誤差を切り分ける必要がある。

## 独立監査と推論確認

全1330親試行のraw/odom SHAと19,546有効教師を再確認。位置・6D回転教師、過去のLIO pose差分、実候補列、role分離、欠測収支を照合。予測保存値からSciPy回転・別の数値経路投影で全7181区間の位置/RPY/SO3誤差と40cm件数を独立再計算し一致。train位置正規化と14候補公開推論も確認。一括と単候補の位置差は最大約1.3e-7m。

関連37tests PASS。全pytestは既知ROS/diffusers等の依存不足による28 collection errorsがあり、全テスト合格とは報告しない（full-pytest-limitations.txt参照）。

## 成果物・再現

作業repo：`/home/ubuntu/.codex/worktrees/eatnav-30s-nohistory/worktree`  
branch：`codex/eatnav-model2000-30s-nohistory-20261112`  
data root：`/home/ubuntu/eatnav-model2000-30s-nohistory-20261112`

checkpoint：`production/model-seed123/checkpoint.pt`（data root基準）  
SHA256：`520a0af98184e26a74d01f6f75b4fec6c410eb22b8c2fe91f0dafce50e8b1f1e`

- `production/dataset/{manifest,split,audit,source-index}.json`：分割・元記録・欠測。
- `results/M30/`、`results/M10/`：490組CSV、区間CSV、候補/field/level別表。`prefix10s/`は先頭10秒。
- `results/M30-duration-rates.csv`：時間帯・field・level・候補別40cm条件。
- `results/M30-predictions.npz`、`M10-predictions.npz`：同一正解での保存予測。
- `independent-audit/`：独立再計算、raw停滞参考指標、FAST-LIO品質、14候補推論確認。

再現例（既存出力を上書きしない新出力先を指定）：

```bash
cd /home/ubuntu/.codex/worktrees/eatnav-30s-nohistory/worktree
export PYTHONPATH=src:scripts
/home/ubuntu/venvs/isaaclab_py310/bin/python scripts/eat_nav_single_pose2s_evaluate.py \
  --dataset /home/ubuntu/eatnav-model2000-30s-nohistory-20261112/production/dataset \
  --checkpoint /home/ubuntu/eatnav-model2000-30s-nohistory-20261112/production/model-seed123/checkpoint.pt \
  --output /tmp/eatnav30-evaluation-recheck --device cpu
/home/ubuntu/venvs/isaaclab_py310/bin/python scripts/eat_nav_long_pose2s_audit.py
```

監査scriptはこのローカル保存先を対象とする。raw/datasetはローカルのみ。今回GitHub push・実機指令・Scene1再走行・新経路選択評価は実施していない。単一学習seedの結果であり、学習seed間の安定性も未検証。
