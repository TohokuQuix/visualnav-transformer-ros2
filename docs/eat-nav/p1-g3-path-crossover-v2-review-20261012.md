# P1 G3 Path crossover 準備レビュー（offline、2026-10-12）

## 判定

既存48 direct trialのsealed rawから凍結済み集計JSONを再現し、新しい解析versionで **24/24 fresh-process seed×env対応**を再評価した。各組のroot、joint、Actor入力・action、command、contact、Critic、reward、torqueなどの時系列は、同じ長さで比較した範囲では完全一致した。Terrain canonical meshもStateGroup内で一致した。旧material SHAは12 processすべて異なるが、旧算出式は設定オブジェクト全体の`str()`であり、値と無関係な`func`のPython関数アドレスを含み得る。旧集計の`fresh_process_matched_pairs=0`と`independent_condition_count=12`は改変していない。

新しいcanonical fingerprintは、Isaac LabがUSDへ書く7つの材料設定値のみを、固定schema・sorted JSON・有限数値でSHA-256化する。現在のOnix rough設定を静的に読むと、terrain材料はstatic/dynamic friction `1.0/1.0`、restitution `0`、両combine mode `multiply`、compliant stiffness/damping `0/0`で、canonical宣言値SHAは`ff1c75072f159375d7f3fd74cd3f7e2f445df9430dede14d7ca8184b18023836`。関数参照やプロセス固有文字列は入らない。Robot側のstartup material randomizationは静的設定上 `0.8/0.6/0`だが、旧runの適用済み値は保存されていない。

**重要な証拠範囲:** このfingerprintは現行ソースの*宣言値*であり、凍結48 trialのlive PhysX materialを直接測ったものではない。旧出力にcomposed USD stageまたはmaterial属性の原本はなく、read-onlyの実測照合はできなかった。新しい`read_usd_material_values()`は将来のlive/composed material primから7属性を読取専用で抽出し、未author属性ならfail closedとするが、今回Isaacは起動していない。したがって新解析の24 matched pairは、同じ凍結source/terrain mesh/seed/env、現行宣言値、完全一致した数値traceに基づく**条件付き再評価**であり、旧runの実効摩擦値の直接証明ではない。候補rolloutの前に、terrain collision meshへのmaterial bindingとcomposed primの属性、robot contact materialをliveで保存し、canonical値と比較する必要がある。

## 原本と新成果物

- 凍結48 trial root：`/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0/`。12 host seal、全traceを読取専用で再検証した。
- 凍結manifest SHA：`39b19cec76e03d0a15daa0e297229ef9c9a9e1325462d786c9d7e2248dd00874`。旧集計SHA：`b75aa65549ca25d186ea2c93542e99f490908b22ce303c170b2dd562b33fb3e3`。
- 新解析：[JSON](p1-g3-path-crossover-v2-analysis.json)。旧aggregateと完全一致する再集計、24組の各Outcome差と時系列照合、旧material SHA、新fingerprint、Path geometry、**設計用**18-process割当を含む。旧manifest、trace、seal、集計は変更していない。

## Path geometryと評価対象

凍結`source-contract.json`と元G3割当のPath SHAを突き合わせた。Pathはbase-frame 8 waypoint、共通ゴールは`(0.75, 0)` m。左は`+20°`、右は`−20°`。いずれもPath公開は1回、途中更新なしとする。

| Path | endpoint (m) | 折線長 (m) | 最大離散曲率 (1/m) | 共通ゴールからのendpoint距離 |
| --- | ---: | ---: | ---: | ---: |
| direct | `(0.7500, 0)` | 0.7500 | 0 | 0 |
| left +20° | `(0.7048, +0.2565)` | 0.7652 | 0.9123 | 約0.2605 m |
| right −20° | `(0.7048, −0.2565)` | 0.7652 | 0.9123 | 約0.2605 m |

`local arrival`は各Path自身のendpointとproduction-equivalent followerのGoal判定を使う。**2秒local endpoint progress**と**2秒common-goal progress**は別フィールドとし、開始時・40 control step終了時の同じbase/map変換済みroot poseから算出する。曲線Pathのlocal arrivalだけを共通ゴールへのnavigation成功と読まない。Path長、endpointずれ、初期距離、曲率、理想Pathに沿う幾何学的進捗を保存し、body/terrain responseを伴わない幾何だけの差をembodiment-aware品質差と判定しない。

主要Outcomeはarrival有無・時間（未到達はright-censored、30秒へ代入しない）、2秒progress、tracking mean/p95、linear/angular/Path-progress recovery、contact force mean/p95/peak、joint torque mean/p95/peak。Trackingとbody loadは既存通りarrival前を主表示し、曝露step数と固定2秒窓も併記して到達時間差による曝露の違いを示す。40-step stop-hold、未到達600-step cap、600-stepで到達した場合のみ最大640-step、40秒episode capを維持する。Recoveryの評価には十分な論理時間と各stepのexact-stamp記録を要求する。

## Balanced crossover案（実行用manifestではない）

2 StateGroup × 3 seed bundle × 3 fresh-process period × 4 env = **18 process / 72新trial**。同一seed×env IDがdirect、left、rightを各1回受ける。3 periodのenv 0–3割当は次の通り。各seedでperiod実行順を循環させ、Pathと実行順の交絡を弱める。

| period | env0 | env1 | env2 | env3 |
| --- | --- | --- | --- | --- |
| 0 | direct | direct | left | right |
| 1 | left | right | direct | direct |
| 2 | right | left | right | left |

全体では各Pathが24 trial。各env IDも各seedで3 Pathを1回ずつ受ける。Period 0/1では既存pair `(0,1)`または`(2,3)`がdirect duplicateになり、計12 direct pairへ**凍結済み数値gateを変更せず**適用できる。Period 2にdirect pairはなく、そのgateは適用しない。旧48 directは同一seed×envの基準分布として保持し、新しいdirect 24 trialと突き合わせて時間経過による環境driftを検出する。新旧protocolの差を黙って統合しない。

各candidateの主要比較は同一StateGroup・seed・env IDの`candidate − 新direct`。旧directとの比較はsecondaryな再現性照合とする。各seedの4 envは同一process/terrain/paired-noise構造を共有するため、**独立単位はseed bundle、計6（各StateGroup 3）**。Envごとのpaired差、seed内4 env平均、StateGroup別の分布、全体の記述的傾向を出す。Process period、env ID、Pathの割当を表として監査し、env×Path完全balanceを検証する。Arrivalはcensorを維持したpaired event/timeとして扱い、未到達を架空の30秒arrivalにしない。

候補差の解釈では、既存direct duplicateの分散を記述的noise benchmarkとし、候補−directの**signed paired差のseed間分散**を測って必要反復数を再推定する。既存p90は検出限界でも95%保証でもない。各群3 seedでは統計的汎化や小効果の識別は主張できない。探索用の既報値ではsecond-gradedのdirect env-pair差SDから、progress 0.10 mの検出に約7 seed、0.05 mに約28 seed、arrival 2 sに約22 seedという正規近似例がある。ただしenv差SDは新しいcandidate−direct対応差のSDではなく、この数字を新pilotの検出力保証や確定sample sizeに流用しない。新pilotでσΔを測り、効果δごとに`n≈((1.96+0.84)σΔ/δ)^2`を再計算する。観測されたcandidate差がPath endpointや長さから予想される幾何差だけで説明できる場合は、execution-quality identifiabilityを支持しない。Common-goal progress、追従、recovery、contact/torqueに対応する身体・地形応答の差を独立に確認する。

## 資源見積もりと実行前gate

旧direct 12 processは49.36–90.24秒/件、計818.37秒、最終保存208.47 MB、GPU device-total peak 5,195.25 MiB、container anon+shmem peak 3.57 GB、1件active storage peak 24.58 MB。G2の600-step 4-envは306.02秒、最終63.80 MB。18件をdirect型で単純外挿すると約20分・313 MB、全件が600-step型なら約92分・1.15 GB。20%余裕を入れると**約24–110分・0.38–1.38 GB**が計画レンジ。これは測定前の概算であり、到達率とtrace長が変わるため最初の独立processで更新する。既存hard capの600秒/process、GPU 7,680 MiB、attribution gap 256 MiB、container anon+shmem 6 GiB、active storage 1 GB/process、heartbeat 90秒を緩めない。実際の永続空き容量・合計保存量を別途確認する。

Rollout承認前に、実行用manifest/source/config SHA、元Path SHA、terrain meshおよび**live material binding/属性**の記録方式、同一seed/envの状態・shared DEM/Actor gate、4 follower readiness/exact-stamp、one-publication、同一episode、direct duplicate gate、batchごとのresource/cleanup/SHA/sealをfreezeする。Protocol不一致やmaterial実測不能ならcandidate Outcomeへ進まない。

今回は静的監査・offline再解析・synthetic testまで。Path rollout、Evaluator学習、full collectionは実施していない。
