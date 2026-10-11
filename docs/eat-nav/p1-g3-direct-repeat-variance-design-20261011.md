# P1 G3: direct Path反復のばらつき評価計画（2026-10-11）

## 目的と判定範囲

現在の4-env Isaac Lab・shared DEM v2・exact-stamp follower系で、**同じdirect reference Pathを実行したときのOutcome分布**を先に測る。既存のdirect duplicate Failはそのまま保存し、PhysX内部状態の完全複製や初期contact差の厳密な因果証明を前提にしない。反復実験だけから曲線Pathの優劣は結論しない。曲線候補の将来のpaired contrastが上回るべき「同一Pathの差」の基準を作る。

本書は未実行の設計である。新規Path rollout、曲線候補、Evaluator学習、full collectionは行っていない。[旧R0–R3計画](p1-g3-contact-physics-repro-design-20261011.md)は補助診断として維持し、反復ばらつきがPath比較を妨げて原因解析が必要になった場合にだけ別途検討する。最初から全physics substepのraw contactは保存しない。既存attempt-0のseal・traceは変更しない。

## 固定する比較条件と実験単位

提案は **2 StateGroup × 3 seed bundle × 2 fresh-process repeat × 4 env = 12 process / 48 direct trials**。StateGroupは既存の`first-differential`と`second-graded`で、terrain・goal・fixtureを各group内で固定する。6つのseed bundleはG2の統計pilot `assignment.json` にある`2026101101–03`と`2026101111–13`を使う。旧G3 assignmentは各groupの最初の2 blockのみなので、旧G3 slot 0 のdirect Path SHAをG2のA条件と照合して継承する。現行sourceではこの`batch_seed`が4-env環境生成とpaired Actor observation noiseの両方へ渡り、terrain fixture側ではanchor固有の`terrain_seed`も設定される。実行前の新manifestには、これらのseedと有効なIsaac scene/reset/RNG seedを**別フィールド**で記録し、同じbundleのrepeat 0/1では全入力を同一にする。複数bundle間ではseedが変わり、実現したinitial root/joint、terrain fingerprint、material、env originも保存する。initial conditionが変わった場合はseed効果と初期条件効果を分離できないと明示する。

各processは固定4-env sceneで20-step settling→env 0を基準とする既存visible state clone→追加physicsなし→env 0由来のimmutableな4-frame shared robot-relative DEM→pre-Path physical/shared-input/Actor gateという現行v2手順を使う。**4 env全てへ同じ0.75 m・0° direct Pathを1回だけ公開**し、同じ`act_inference` checkpoint、同じfollower設定、同じnoise streamを使う。native DEM、contact、Criticは別診断として残し、人工的には一致させない。4 branchのfollower/bridge化は新runnerの実装変更であり、production followerと既存runnerを変更しない。全branchのreadiness、stamp、one-publication、40 s episode cap、20 Hz、arrival後40-step hold、未到達600-step censor、step 600で到達した場合の最大640-stepという条件を維持する。

比較を三つに分ける。(i) 同一process内のenv 0–1、2–3のdirect duplicateペア（1 processをclusterとし、2ペアを独立標本と数えない）、(ii) 同一StateGroup・seed・env IDのrepeat 0–1（fresh process差）、(iii) StateGroup内の異なるseed bundleとenv IDの差。独立した条件単位は`StateGroup × seed bundle`の**6個**、fresh processはその条件の2回測定である。2 StateGroupしかないためterrain母集団への一般化は行わない。異なるenvのworld tile/solver配置差は「この4-env実験系の自然なばらつき」に含める。

全slotがdirectであるため、この段階ではPath割当とenv IDは交絡しない。将来の曲線Path比較では、事前固定したbalanced/crossover割当で各Pathを各env IDに同回数置く。**今回の結果を見てから**有利なenvにPathを割り当てない。

## 取得とgate

各trialにprocess ID/PID・start tick、env ID、StateGroup/terrain・mesh/material SHA、env origin、realized root/joint/initial condition、seed bundleと個別RNG seed、noise key/array SHA、checkpoint SHA、Path SHA/geometry、follower/bridge source・params SHA、Isaac/PhysX/GPU provenanceを記録する。controlごとにlogical stamp、clock/TF/command/bridge acceptance/physics順序、Actor入力とaction、root pose/velocity、contact force summary、Critic、follower mode/recovery、Path追従とprogress、torqueを保存する。pre-Pathのnative DEM/共有DEMとsensor actual stampは別系列にする。raw全substep contactは採らない。

Pre-Pathのphysical equality、shared DEM byte/SHA/frame/time equality、Actor input `1e-4`、native DEM診断、same-episode、exact-stamp、nonfinite、resource、native exit、cleanup、host/container SHA・host sealは従来どおり適用する。失敗したprocessは**measurement-invalid**として停止し、Path公開後のretryはしない。post-Pathの凍結済みdirect duplicate gateはペアごとに、arrival category、arrival spread **≤0.10 s**、tracking mean spread **≤0.02 m**、2 s progress spread **<0.025 m**、recovery category一致で判定する。gate Failは**ばらつきの観測結果**として保存し、他のprotocol条件が正常なら後続の予定processを止めない。旧attempt-0のFail判定を変更しない。pre-Path gate不合格・stamp不整合・unexpected reset/termination等は後続を止める。

主要Outcomeはlocal arrival statusとtime（未到達は600 stepでright-censor、terminalは別理由）、Path cross-track error（到達またはcensorまで、hold除外）、recovery発生・mode・回数/持続、body load（接触forceとapplied torqueの時間平均・p95・peak、曝露時間を併記）を別々に出す。2 sの**local Path終点へのprogress**を固定40-step窓で出す。common-goal progressはnavigation diagnosticとして別フィールドで保存する。directでは両終点が同じでも意味を混同しない。到達後hold中のzero commandと再発進有無を保存し、holdを追従誤差平均へ混ぜない。

## 解析と判断

各OutcomeについてStateGroup別・seed別・env別のtrial値、平均/標準偏差/中央値/p10/p90、同一process duplicateの符号付き・絶対paired差、同一seed/env fresh-process差を報告する。未到達のarrival timeを600-step値で代入せず、到達割合とright-censorを別表示する。gate Pass/Failを**ペア数とprocess数の両方**で表示し、信頼区間は6条件単位をclusterとして扱う。少数条件のbootstrap区間は探索的とし、精密な母集団推定とは呼ばない。

seed・env・processの分散は、同一seed/envのrepeat差、各process内env差、seed bundle間のprocess平均差から記述的に分解する。n=3 seed/group・2 repeatなので不安定なvariance componentは`not estimable`とする。Actor noise seedとscene/reset seedが束になって変われば、そのどちらが原因かは識別不能と記録する。env ID差はtile位置・並列solverも含む。

同一Pathのpaired絶対差について中央値、p90、最大値を**記述的repeatability benchmark**として報告する。24 duplicate pairも24 fresh-process matched pairもcluster相関があるので、p90を統計的な検出限界や95%検出保証とは呼ばない。将来のPath比較で検出し得る効果は、独立条件ごとのpaired effectの分散、独立条件数、反復数を使って別途検討する。local quality/body responseの差がPath終点の幾何差だけで説明されないことも必要である。今回のdirect-only結果は「候補間品質差が識別できた」にはならない。ばらつきが大きい場合は追加反復数の検討資料を提示し、原因分析が必要ならR0–R3を補助診断として選択する。

## 資源・時間見積もりと実行境界

既存のsealed direct 2-branch実績は**52.56 s、8.4 MB、GPU 5194.75 MiB、container anon+shmem 3.50 GiB**。4-env・600-stepのG2実績は**306.02 s、62 MB**。今回の4 follower/4 direct traceはG2よりやや重くなり得る。設計見積もりは1 process **約1–6分**、12 process **約12–72分**、監査・cleanupと20%余裕込み**約1.5–2時間**を計画値とする。既存の1 process 600 s capを全部使う保守的上界は12×600 s=**2時間**、最終監査に15–30分を別途確保する。まず最初の1 processの実測で残り時間を更新するが、試行数・seed・順序・閾値を事後変更しない。

保存はG2の62 MB×12×1.2≒**0.9 GB**を初期見積もりとし、4 follower化と最大640 stepによる上振れを考え、実行前に永続rootの**2 GB以上の空き**を確認する。既存host monitorのactive storage 1,000,000,000 B/process、GPU device-total 7680 MiB、GPU attribution gap 256 MiB、host available RAM 10 GiB floor、container anon+shmem 6 GiB cap、heartbeat 90 s、600 s/process capを変更しない。GPUの観測値5195 MiBに20%余裕を乗せると約6234 MiB、container RAM 3.50 GiBに20%で4.20 GiBであり既存cap内。ただし4 follower化の追加負荷は未測定なので、最初のprocessをresource calibrationとして測り、cap超過なら停止する。全processを逐次起動し、永続subroot、fsync済みtrace、resource series、SHA ledger、atomic marker、writer終了後sealを残す。既存root/証拠はread-only。

## 実装前の静的・synthetic確認と承認境界

新しいrunner/manifest/集計器は既存sourceを変えずに別versionで作り、(1) 4 env全てが同一Path SHAで各1回publish、(2) seed/repeat/env割当、(3) shared/native DEMの分離、(4) frozen duplicate gateとFailの記録、(5) censor/holdとbody-load exposure、(6) process単位集計、(7) SHA/seal/cleanupをsynthetic testで確認する。source SHAと見積もりは実装後にfreezeする。**この段階でrolloutは開始しない**。実装・synthetic確認後、実行manifestと実測可能な資源上界をレビューに提示し、明示承認を受けてから新規Path rolloutを開始する。
