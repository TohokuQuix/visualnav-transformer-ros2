# R4: 直線スタックのみ・実測姿勢STOPなし 評価 v0.1

2026-10-26 JST。利用者の実行指示、および「表の直線スタックのみ使用、その他はR2/R3と変えない」という補足を実装条件とする。

| 項目 | R4 |
|---|---|
| 直線スタック予測 | 有効。採点重み1.0、既存の除外条件p_linear>=0.95かつprogress10<=0.05m |
| 旋回スタック、転倒、near-fall予測 | 採点・除外・STOP・再開に使用しない |
| 実測45度STOP / 35度以下での再開 | 無効。実測傾斜は記録のみ |
| 候補切替保持・マージン・前後進確認 | 現行の4秒・0.15・2回を維持 |
| 予測による全候補除外時のSTOP | 直線条件だけで判定。最短2秒停止、2秒周期で2回確認の再開を維持 |
| 学習モデル | 同じ12 checkpointをSHA固定して再利用。再学習なし |

既存RiskGuardに有効headだけを入力し、姿勢guardを無効にした新adapterを使う。旧source・sealed出力を変更しない。

- 全7field・level4・16m・front_folded・固定Actor・20Hz・候補4本・2秒replan・path継足し・waypoint列・最大60秒・初期化/settlingを既存と同じにする。
- 目標到達の定義（45度未満等）、進捗定義、boundary censor、finite検査、自然85度/tilt_over終了も変更しない。目標判定は比較指標/目標更新であり45度STOPではなく、goal進入で物理を止めない。
- 元R3割当の各モデル×走行seedをR4へ対応させる。K-Rail6モデル×3走行seed=18、他6field各3=18、計36予定。
- 既存settling failureのseedも入れ替えず処理結果を残す。成功するまでの再実行・seed追加なし。
- runtime image/Onix/source/Actor/config/モデル、開始anchor・全model入力を旧R1/R2/R3と照合。不一致は比較N/A、黙って採用しない。
- R4が転倒・boundaryで早期終了しても有効な物理outcomeとして保存する。起動失敗と区別し、失敗全件を報告する。
- GUIで実走行を表示、hostとDocker/Isaac実起動コマンドを表示・保存。保存入力から推論とguard全stepを再現し、STOP3軸0・45度以上でも姿勢によるSTOPなしを検証する。
- reserved test未開封。新しい学習データ生成・重み更新・閾値調整は行わない。
- R4とR2は旋回riskおよび姿勢STOPの2項目が異なるため、45度保護の単独効果とは結論しない。独立走行seed数と学習seed反復数を区別する。

実装順: guard回帰test → 新GUI adapterと36割当freeze → 同条件GUI36予定を処理 → 全件再現監査・対応比較・報告。
