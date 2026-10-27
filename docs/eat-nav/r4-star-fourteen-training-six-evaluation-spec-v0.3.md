# R4*: 既存14円弧候補で学習・前後3本ずつで評価 v0.3

2026-10-26 JST。教授の指示を受けた利用者の新しい実行指示。学習を前後7本ずつへ変更し、評価は前後3本ずつのままにする。v0.1/v0.2の待機処理は開始前に停止し履歴を保存。進行中のR4評価は継続する。

## 既存データから確認した経路

`/home/ubuntu/eatnav-student-share-20261018/training-evidence/contract.json`のcandidate_count=14とcandidate_order、および`collection-evidence.tar.gz`内の各候補path-activation.jsonのdirection/arc_angle_deg/path_body_sha256を確認した。splitのラベル、grouped/test配列、test予測は開かない。`bidirectional_bank.bank()`の14候補と全path SHAを照合して記録する。

候補順はforward_direct, forward_-30, forward_+30, forward_-60, forward_+60, forward_-90, forward_+90, reverse_direct, reverse_-30, reverse_+30, reverse_-60, reverse_+60, reverse_-90, reverse_+90。

弧長L=0.75m、8点、s=linspace(0,L,8)、非直進はtheta=指定角rad、R=L/theta、(x,y)=(R sin(theta*s/L), R(1-cos(theta*s/L)))。reverseは両座標へ−1を掛ける既存定義。指定角は円弧全体の接線角変化であり、直線放射の方向角ではない。後退曲線の符号・左右は既存travel frame/body frame規則を変えない。継足しは既存の解析的末端body headingを使う。

## 収集とdataset

全7field、既存8収集seed(train6/validation2)、anchor0/10秒、prefix、10秒branch、Actor・terrain/level4/16m・front_folded・20Hz・2秒再構成・path継足し・goal列・85度転倒判定・境界censor・教師v2をv0.2と同じに保つ。変更は不足候補の範囲だけ。

既存4候補は再収集せずsealed datasetを保持。canonical14のslot0/1/2/7が既存4候補に対応する。不足slot3/4/5/6/8/9/10/11/12/13の10候補を追加する。7field×8seed×2anchor×10=1120追加branch。学習用prefixは元の4候補selectorを維持し、selectorの旧reverse index3をcanonical index7へ対応させる。prefixを14候補で選び直してanchorを変えない。

全1120予定と全失敗を保存し、成功するまでretry/seed置換をしない。収集ledgerが成功、horizon/natural/boundaryの利用可能outcome、host/simulator failureなしを要求し、anchor全数値状態と共通入力を旧データと照合したものだけ教師化する。失敗/不一致は該当slotのunknown mask。旧4候補の教師は保持する。元datasetに含まれなかったanchorは追加しない。各fieldで追加全10候補の有効train教師があることを確認し、欠ける場合は「全14候補を学習した」と称さずN/Aを報告する。

旧の既存データを混合する際も14軸へ写像し、未観測10slotはcandidate_mask=False・ラベルunknown・progress mask=Falseとする。異なる候補へのラベル流用は禁止。

## 学習と評価

field別300epoch、Adam lr0.001、field:旧データsource loss2:1、同じevent/progress/pair損失、trainのみ正規化、validation BCEでepoch選択。K-Railは同じ6学習seed、他6fieldは123の1モデル、計12モデルを新規初期化から学習。全headを同じ方法で学習し、評価の制御へ使うのはlinear headのみ。既存validation fallbackを維持し、結果と候補別coverageを明記する。別の過去motion予測モデルの教師/weightsへ置き換えず、既存14候補の幾何を現在の転倒・スタック教師v2で学習する。

R4*評価入力は学習14候補のsubsetであるcanonical slot0/1/2/7/8/9。評価の順序は従来R4*と同じ前進直進/−30/+30・後退直進/−30/+30。実際の6候補GUIと選択ロジックは変更しない。新しい14候補学習済みcheckpointのpath/SHAを結び、同じ36モデルseed×走行seed・最大60秒・Actor・goal・停止/切替条件で評価する。

R4と終了後監査の両sealが完了してから収集を開始。現在のR4 source/outputは変更しない。GUI実走行とhost/Docker起動コマンドを表示・保存。保存入力推論・全step guard・初期状態照合・STOP3軸0・model SHAを監査し、進捗/転倒/STOP/観測時間/waypoint/linear stall/候補利用を対応比較する。R4との差は学習候補/追加教師と評価候補を合わせた効果。独立走行seed数を学習seed反復と区別し、reserved test非開封を維持する。
