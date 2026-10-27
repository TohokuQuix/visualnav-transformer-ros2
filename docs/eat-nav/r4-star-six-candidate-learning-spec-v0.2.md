# R4*: 後退曲線を収集・学習した6候補評価 v0.2

2026-10-26 JST。利用者が再学習なし案を変更し、後退±30度の実走行データを追加した学習後のR4*評価を指示した。旧待機runnerは開始前に停止し、queue statusをcancelled_before_startとして保存。現在進行中のR4評価とそのsource/成果物は変更しない。

実行順はR4の36予定と終了後監査の完了 → 後退2候補の追加収集 → 教師v2/dataset → 新モデル学習 → 6候補R4* GUI評価。単なる入力候補拡張の旧v0.1は実行しない。旧資料とqueueは経緯として保持する。

## 固定条件

全7fieldで既存all-fields-learning-20261023の収集seed(train6/validation2)、anchor0/10秒、prefix、Actor、地形level4/16m/front_folded、20Hz、2秒再構成、継足し/waypoint、85度転倒終了、境界censor、教師v2を再利用する。新たにcandidate index4=reverse_-30、5=reverse_+30を各seed/anchorで10秒継続。7×8×2×2=224追加branch。収集のrisk/45度STOPは従来同様使用しない。prefix selectorは旧4候補のままで、初期状態を合わせる。GUI実走行と起動コマンドを表示・保存する。

新候補が旧4候補の同一StateGroupに属するかをanchor全数値状態・DEM/Actor/body共通入力で照合する。旧4候補は再収集せずsealed datasetの同じ入力・教師・maskを保持する。追加候補の失敗/anchor不一致はunknown maskとし、旧4候補の教師を消さない。追加成功だけを別seedへ置換せず、224全予定と除外理由を残す。既存4候補datasetに含まれなかったanchorは今回も学習へ追加しない。

datasetは6候補軸。旧既存train/validationデータは先頭4候補を保ち、新規2slotを無効mask/unknownで拡張する。ラベルの流用・複製は禁止。全headを既存と同様に学習し、評価時にlinear headだけを使う。学習と評価のcandidate順は前進直進/−30/+30、後退直進/−30/+30。

学習はfield別に新規初期化から同条件300epoch、Adam lr0.001、field対旧データsource loss2:1、同じevent/progress/pair係数、trainのみ正規化、validation BCEでepoch選択。K-Rail学習seed123/456/789/101112/131415/161718、他6fieldは123。計12モデル。既存と同じ新field validation不成立時の旧validation fallbackを明記する。追加両候補の有効train教師がなければそのfieldは学習済み追加候補と称さずN/Aを保存する。陽性不足による追加batchは自動で行わない。

評価はR4と同じ36モデルseed×走行seed割当・最大60秒。使用checkpointのみ新学習済みへ変更し、候補6本と追加データ学習以外の条件はR4と同じ。直線riskのみ、45/35度姿勢STOPなし、保持4秒・margin0.15・前後進確認2回・予測STOP最短2秒/再開2回を維持。全12モデルを結果により選別しない。

R4/R4*の初期anchor全数値状態と共有入力を対応照合し、candidate配列は先頭4本の一致と新2本bank形状を確認する。全step guard/6候補の保存入力推論/STOP3軸0を再現する。モデルweightsが異なることによるTorch RNG初期化状態の差は、同じarchitectureと学習seedでのload手順が同じため許容せず、anchor照合を省略しない。失敗はN/A、成功するまでretryしない。

比較指標は進捗、転倒観測、STOP時間、観測時間、正常waypoint、動作中linear stall、候補別利用。結果は候補追加と追加教師学習の組合せの差であり、候補数のみの単独因果効果ではない。独立走行seed数と学習seed反復を分ける。test未開封、実機未実施を維持。
