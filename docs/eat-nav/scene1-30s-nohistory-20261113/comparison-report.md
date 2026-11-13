# Scene1: 30秒データ学習NNと従来NNの比較

復帰ルールなし。探索的評価（offline精度の全条件は未達）。NNと各自の学習済み正規化だけを交換。元10試行の初期root状態は全要素一致、実測物性配列も全要素一致。Actor・開始・goal・B60・300秒/85度終了条件を維持。入力は前回と同じideal synchronous simulator観測であり、actual FAST-LIOは使用しない。

|条件|goal|転倒|timeout|最接近距離|
|---|---:|---:|---:|---:|
|従来10秒NN|0|8|2|1.227m|
|新30秒NN|0|10|0|1.589m|

今回の10試行では成功率の改善は確認されず、最接近距離も悪化。単一seed・同じ開始goalの10試行で一般性能や原因を断定しない。平地迂回は0件のため壁追加版は実施せず。

|試行|終了秒|終了|転倒地点x,y (m)|領域|
|---|---:|---|---|---|
|original-t01|29.00|転倒|(1.492, 0.829)|central_lower_ramps|
|original-t02|22.60|転倒|(1.357, 0.908)|central_lower_ramps|
|original-t03|10.05|転倒|(1.274, 0.393)|left_V_section|
|original-t04|16.65|転倒|(1.449, 0.914)|central_lower_ramps|
|original-t05|67.00|転倒|(2.981, 1.391)|central_upper_ramps|
|original-t06|32.20|転倒|(1.465, 0.810)|central_lower_ramps|
|original-t07|10.90|転倒|(1.303, 0.483)|central_lower_ramps|
|original-t08|23.80|転倒|(1.354, 0.927)|central_lower_ramps|
|original-t09|23.00|転倒|(1.383, 0.954)|central_lower_ramps|
|original-t10|23.60|転倒|(1.409, 0.890)|central_lower_ramps|

領域名は座標による区分。接触相手や転倒原因を確定した分類ではない。

## 再現と保存

実行: `/usr/bin/python3 /home/ubuntu/eatnav-scene1-30s-nohistory-navigation-20261113/code/workflow.py`（同じrootで完了済みなので再起動しない。再実験時は別rootへ複製しパスを更新する）。環境・依存資産はprotocol.jsonとcode内の固定パスを参照。

ローカル動画: /home/ubuntu/eatnav-scene1-30s-nohistory-navigation-20261113/video/original/original-t05-8bfacc16e0.mp4

動画は実走行のroot/joints記録からの描画（新しい物理走行ではない）、336frames/67.2秒。巨大trace・動画・資産はGitに含めない。元データは /home/ubuntu/eatnav-scene1-30s-nohistory-navigation-20261113

新checkpoint SHA256: 520a0af98184e26a74d01f6f75b4fec6c410eb22b8c2fe91f0dafce50e8b1f1e
