# 狭通路5配置＋元7field：最終結果

状態：独立実ファイル監査・全8画像の目視QAまで完了。

収集50親、100起点枠中95取得、14候補×2秒で1400予定枠。1328完全2秒教師、70状態未到達、2物理途中終了。固定spawnの共通開始起点168行はactual/ideal共通でinput_validをfalseにし、元教師・入力・rawを保持。有効新規1160件＝train783 / validation195 / final_test182。旧7fieldの配列・記録・役割・maskは完全保持し、混合有効件数は5931 / 7412 / 7363。

## 同一行・各モデル自身の学習データ由来正規化での比較

以下はMAE。位置と経路誤差はcm、姿勢は度。旧モデル520とscratch seed123新モデルを、完全に同じ有効行で比較。

### 狭通路・actual FAST-LIO：n=182

| 指標 | 旧520 | 新scratch123 |
|---|---:|---:|
| 水平位置 | 16.864 | 8.182 |
| 3D位置 | 17.758 | 8.495 |
| roll | 3.131 | 2.251 |
| pitch | 5.330 | 3.033 |
| yaw | 9.261 | 6.145 |
| 経路接線方向 | 14.820 | 6.619 |
| 経路横方向 | 6.571 | 3.674 |

### 元7field・actual FAST-LIO：n=7181

| 指標 | 旧520 | 新scratch123 |
|---|---:|---:|
| 水平位置 | 12.100 | 11.941 |
| 3D位置 | 12.574 | 12.407 |
| roll | 4.520 | 4.448 |
| pitch | 3.575 | 3.485 |
| yaw | 7.441 | 7.306 |
| 経路接線方向 | 10.453 | 10.130 |
| 経路横方向 | 4.458 | 4.630 |

### 狭通路・ideal入力の別評価：n=182

| 指標 | 旧520 | 新scratch123 |
|---|---:|---:|
| 水平位置 | 17.096 | 8.751 |
| 3D位置 | 17.542 | 9.056 |
| roll | 2.066 | 2.225 |
| pitch | 4.490 | 2.844 |
| yaw | 7.266 | 6.243 |
| 経路接線方向 | 15.074 | 7.204 |
| 経路横方向 | 6.688 | 3.797 |

狭通路actualの3D位置MAEは約52.2%減少。旧7fieldの全体3D位置MAEもわずかに改善。ただし旧7fieldの横方向誤差は4.458→4.630cmへ悪化し、ideal別評価のrollも2.066→2.225度へ悪化。全指標一律改善という結論ではない。

## 過大・過小評価と14候補選択

狭通路actual182件で経路接線方向の10cm超過大評価は旧22/182(12.09%)→新11/182(6.04%)。可動行で2cm超過小評価は126/182(69.23%)→92/182(50.55%)。ただし経路投影進捗2cm以下の新規test行は0件であり、停止・閉塞時の過大予測改善はこのtestから検証できない。経路投影進捗と実XY変位は別の指標。

完全14候補を同一起点で比較できた新規testは13起点。actual・各モデル自身の予測によるfilterで選んだ候補の実score平均は、c=0で0.34585→0.35423m、c=.45で0.23675→0.31139m。選択候補の実XY変位2cm以下・実tilt60度超・実終点床外はいずれも旧新0/13。小標本の保存分岐比較であり、閉ループ走行成功率ではない。旧7fieldは連続窓収集のため完全14同一起点の主比較は成立せず、0groupを明記。

## 収集状態と解釈

直線は通路10＋移動後10起点。左90は通路4/旋回前6/旋回中4/旋回後4、右90は4/6/6/4、左180は4/6/6/1、右180は4/6/6/4。これは全role合算の物理取得起点であり、共有開始起点の分割除外前。左180旋回後のcoverageは薄い。

実測起点のroll約−.159〜.636度、pitch約−.080〜17.456度、衝突支持点と壁の最小gap約.020〜338.812mm。直前2秒の非ゼロ指令割合.35〜1、床を含む接触合力最大約9079N。壁専用の接触力ではない。sampled gapは連続軌跡の非接触証明でもない。AFTERには逆向きでの後退退出を含み、180度前進旋回達成と混同しない。

ユーザーの「壁の反力は極力利用しない」という説明は確認質問の意図として記録。この話題を理由に今回のdataset・学習・評価を変更していない。壁反力の利用を目的・報酬・評価指標に追加していない。

## 学習・監査

既存コード構造・入力・損失・AdamW設定でwarmstartなしscratch seed123。train5931/validation7412、46epochsでearly stop、validation選択best epoch21。checkpoint SHA a9fddc4e05a846762277de2ea06ad3efba0016ae64e5c237ad64f54ca2ca22c6。学習スクリプトはfinal_test未参照。学習ループ所要約6.63秒（パイプライン全体所要ではない）。

独立監査は全50親/1400枠、実sensor/全prefix420jobpairs×2kind/5物性、全SciPy教師、因果FAST-LIO入力、共通除外168行、旧7完全保全、両train-only normalization、validation選択、保存予測15090行→32034統計/208候補選択を別実装で再計算PASS。最大CSV数値差6.66e−16。補足層別378行も独立再計算PASS、片側Euler singularは0。dataset5画像＋offline3画像を実view_imageで確認しSHA固定。独立CPU監査テスト8件PASS。

評価条件は既知5geometry・固定spawnでの別閉ループ実行。共通初期rootを持つ10群と3つのrole別global seedを明記。50独立初期条件、未知地形一般化、Scene1/2や実機・追加閉ループ成功の根拠にはしない。独立監査は保存予測の再計算であり、checkpointから別forwardで予測値を再生成したものではない。

## 実ファイル

- 独立最終監査：[independent-final-audit.json](audit/independent-final-audit.json)
- 独立補足監査：[independent-supplement-audit.json](audit/independent-supplement-audit.json)
- 全指標・field×level×candidate・direction・phase：[metrics.csv](offline-comparison/metrics.csv)
- 全保存予測誤差：[per-row-errors.csv](offline-comparison/per-row-errors.csv)
- 同一起点14候補c0/c.45：[complete14-selection.csv](offline-comparison/complete14-selection.csv)
- 壁方向/geometry/隙間などの補足：[metrics.csv](offline-comparison/supplementary-strata/metrics.csv)
- 実測起点：[measured-anchors.csv](dataset-report/measured-anchors.csv)
- 画像QA：[visual-QA.json](dataset-report/visual-QA.json)、[offline manifest](offline-comparison/figures/manifest.json)
