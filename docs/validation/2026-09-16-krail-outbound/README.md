# 2026-08-03 10:21:21 往路bagのオフライン検証

実施日: 2026-09-16

対象: `/media/ubuntu/Extreme SSD/20260803_experiment/2026_08_03-10_21_21`

## 結果

- 前方圧縮画像458枚、画像stampで45.696秒。TF変換9848件を読み込み。
- 既存ImageHistoryで4枚・0.25秒以上の間隔を形成。履歴リセット0回。実際の採用間隔はカメラ周期のため通常0.3秒であり、厳密な0.25秒サンプリングではない。
- 0.9〜41.7秒の11観測を抽出し、baseline、最新Normal固定、最新switchback固定を比較。
- 全33推論で有限の8点を生成し、観測時刻のTFで通常9点のmap Pathを生成できた。
- CPU推論時間: baseline 0.113〜0.156秒、最新Normal 0.114〜0.115秒、最新switchback 0.113〜0.115秒。画像復号・ROS配信を含むend-to-end時間ではない。
- 最新Normalとswitchbackの同一観測・同一seed42での予測点差: 最大0.00250m、平均0.00095m。今回の観測に限る結果で、状態条件が常に無効だという意味ではない。
- 全抽出観測で到達判定False。最新モデルの距離出力は11.71〜15.91、閾値3.0。最終抽出は41.7秒であり、45.7秒のgoal直前・終端の停止は未検証。
- 画像10枚ごとのTF照合では46件中1件でTF取得不可。推論対象11観測はいずれも取得できた。
- bag記録時刻と画像stampの差は約60.7〜106.2ms。

## 条件と制限

goal.jpgはbag最後の前方画像から保存した検証用JPEG。画像列のtopomapは使っていない。終端が利用者の意図する目的地かは未確認。

baselineはROS2ブランチの単一goal推論を使用。最新モデルは学習元NoMaDクラスを読み込み、647キーをstrictロードした診断用アダプタを使用した。diffusionのglobal_condへ学習時と同じ状態条件を加算し、距離ヘッドは元の特徴を使用。これは最新モデルのROS2移植実装ではない。

TFをbag全体から先に読み込んだオフライン計算であり、再生時のTF到着遅延やROS QoS・watchdogは検証していない。ROSトピック配信、速度指令配信、追従ノードによる閉ループ走行は行っていない。検証に用いたsandboxからGPUデバイスへアクセスできず、検証用PyTorchも2.14.0+cpuだったためCPUで実施。Torchは1スレッド。2026-09-16の追加確認では、sandbox外のnvidia-smiでRTX 4090 Laptop GPU（16376 MiB）、ドライバ580.173.02を認識した。ホスト側GPUの故障やドライバ不在ではない。GPU検証にはCUDA対応PyTorchとGPUへアクセスできる実行環境が必要。

図の灰色線はbagに記録された車体位置、色付き線は各観測時点の予測Path。軌跡は観測時点からの短期予測であり、goalまでの全経路ではない。教師座標とTFの前処理由来の同期対応や経路精度は、この重ね合わせだけでは保証しない。

## 発見・修正した不具合

画像履歴snapshotのtupleをtransform_imagesへ渡すとAttributeErrorが発生した。接続側でlistへ変換して修正し、単一goalの回帰テストをtuple入力へ変更。修正後に同bag検証と31テストが成功した。

## 成果物

- [経路比較図](paths.png)
- [検証用goal画像](goal.jpg)
- [全予測と計測値](results.json)
- [診断スクリプト](offline_check.py)

診断スクリプトはこのPC用の絶対パスと `/tmp/krail-bag-check/source` のbaselineソースコピーを使用する一回の検証用。再実行時は修正後baselineのソースをその場所へ配置し、ROS2 Jazzyをsource、PYTHONPATHへそのsrcを追加する。モデルやROSノードの正式な起動スクリプトとして使わない。

## 今後の対象

ユーザー指示により、以後の動作確認はbaselineのみを対象とする。上記の最新モデル比較は過去の診断結果として残す。GPUでのbaseline推論・ROS連続再生は未実施。

## baselineの8候補表示

同じbag・goal・11観測・seed42・CPUでbaselineのみ再推論し、各観測8候補（各8点）を保存した。先頭候補は前回保存した結果と全11観測で完全一致。実行コードのget_action出力を診断用に捕捉し、本体の経路選択・Path配信処理は変更していない。

- [全体重ね合わせ図](all_candidates/baseline_all_candidates.png): 観測ごとに色分け。太線は採用候補0、細線は候補1〜7、灰色は記録走行軌跡。
- [観測ごとの拡大図](all_candidates/baseline_candidates_detail.png): 赤は候補0、青は残り7本、黒点は観測時の車体位置。
- [8候補の数値](all_candidates/results.json)
- [診断スクリプト](all_candidates/offline_all_candidates.py): 上記と同じ一時ソース・検証環境が必要。
