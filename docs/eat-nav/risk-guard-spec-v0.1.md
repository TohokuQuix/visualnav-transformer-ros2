# 高リスク停止と候補切替安定化 v0.1

2026-10-22 JST。利用者の依頼により、モデル・datasetを固定した経路選択ルールのGUI比較を行う。
対象は既存28次元command-free Evaluator、checkpoint SHA
`7e6b6af0f56d63c76dce7d9a52cc00a02b95b32f587449a2e32ab079ff9812d1`。
Actor、network、正規化、教師、datasetは更新しない。

旧ログ監査:14秒時点では全候補のtip scoreが0.919以上でも走行継続した。
一方、18秒時点で選択候補tip scoreは約0.010だったが、その後2秒で傾斜53度に到達し、
20〜22秒で最大54.58度になった。予測だけで実傾斜増加を検出できたとは言えない。

以下は開発用に固定した探索的heuristicであり、校正済み事故確率・安全性保証ではない。

1. 2秒ごとの候補判定:
   - tipまたはnear scoreが0.7以上なら候補を除外する。
   - linearまたはturn scoreが0.95以上、かつ10秒progress予測が0.05m以下なら候補を除外する。
     10秒内のstall eventと、その後の移動は両立するため、stall score単独では停止させない。
   - 除外されない候補だけで既存のgoal progress proxy−risk penaltyを比較する。
     全候補が除外されたらSTOPとし、選択indexはnullにする。危険候補へのfallbackはしない。
2. 実姿勢の判定:
   - 20Hzで実測tiltが45度以上になったら、次のreplanを待たずSTOPをlatchedにする。
   - 再開にはtilt35度以下、tip/nearがともに0.5未満の候補あり、stall除外条件を満たさないことを要求する。
   - 最低40 control step停止し、2秒ごとの再開条件が2回連続で成立したら再開する。
     これは完全な静止や停止時の安全性を保証するものではない。
3. 候補切替:
   - 通常は最低80 step（4秒）保持し、新候補のscoreが現在より0.15以上高い場合だけ切り替える。
   - 前進/後退の変更は、その方向が上の条件を満たして2回連続の判定で優位な場合に限る。
   - 現候補が除外条件に入った場合は保持時間・score差・連続確認を待たず、残る候補へ切り替える。
   - STOPからの再開は安全側の候補から選び直す。同点は従来の固定候補順。
4. 実行:
   - STOP中も物理・観測・固定Actorは進める。PureFollowerのstepは呼ばず、速度指令(vx,vy,w)を全て0にする。
     既存の観測/指令遅延は保持するため、Actorが見る指令や実速度が即座に0になるとは扱わない。
   - 再開時に新しいPureFollowerを作る。経路表示はSTOP中は全候補を灰色にする。
   - 各decision前後のguard状態、除外mask、理由、入力、予測を保存。
     各20Hz stepでplanned候補と実際のSTOP/走行、tilt、指令、guard状態を保存する。
   - 予測が不正なら従来どおり実行失敗として保存し、別候補へのfallbackで隠さない。
5. 比較:
   - sealed旧28次元30秒走行（seed2191202837、K-Rail L4、front-folded、1env、20Hz、600 step、前方4m goal）と比較。
   - source/Actor/config/checkpoint、開始時の生観測を照合。score係数は[2,0.5,1,0.5]のまま。
   - goal残距離・最大傾斜・停止時間・実速度・候補/方向切替・再開回数を別々に報告する。
     止まるだけで傾斜が下がった結果を、自律走行性能改善とは呼ばない。
   - GUIを前面表示し、実行するhost起動コマンドと実際のDocker/Isaacコマンドを表示・保存する。
     終了後60秒pause表示。reserved testは開封しない。

上記閾値は本実走行前に固定し、結果に合わせて同じ試行内で変更しない。
元のscript・過去sealed成果物・収集binding対象sourceは変更せず、新規scriptと新規保存先を使う。
