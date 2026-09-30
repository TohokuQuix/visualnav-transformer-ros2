# EAT-Nav NoMaD Goal-conditioning / Goal-OOD offline診断

## 実行範囲と監査

凍結best EMA checkpoint、complete-only train 49,598 window・validation 6,171 window、既存split hash、train-only action normalizationを照合した。train 32 state、validation 32 stateへ、各13 goal × 同一8 seedを固定し、合計6,656 Pathをoffline生成した。held-out、simulator、再学習、候補再生成、閾値変更は使っていない。推論前にstate・goal・seed・順序・入力SHAを固定した。

relative goalは現在のrobot-local frame（x前方、y左）の2秒後までの最後のvalid future endpointを半径0.8 mへclipした値である。complete-onlyでは8点目そのものを使う。clip前endpointは累積actionから復元でき、保存goalはclip後である。partial terminalは最後のvalid point、zero-validはゼロgoalを使うが、この診断には含めていない。

## Train goal support

train goal半径の中央値は0.355 m、P90は0.800 m。clip率18.9%。方向sectorは前方65.6%、後方29.1%、左3.1%、右2.2%。半径<0.2 mは34.7%、≥0.6 mは30.7%。x<0の割合31.7%、|y|≥0.1 mは11.4%（sector定義と異なる）。

trainだけで極座標binの独立episode支持とkNN k=16をfitした。kNN q95=0.0159 m、q99=0.0372 m。validationは境界決定に使っていない。1,024個の固定train DEM probeのうち、近傍DEMで別episodeが見つかったのは870件、そのうち複数goal binは868件。この近傍検索は候補shortlistを用いる標本診断であり、全件厳密近傍保証ではない。

## Goal adherence と OOD

| 区分 | train endpoint誤差（state中央値平均） | validation | train 0.2 m内 | validation 0.2 m内 |
|---|---:|---:|---:|---:|
| ID | 0.0065 m | 0.0059 m | 100.0% | 100.0% |
| near-OOD | 0.0451 m | 0.0437 m | 100.0% | 100.0% |
| far-OOD | 0.1690 m | 0.1694 m | 74.3% | 74.5% |

同一DEMとseedを保ちgoalだけを変えると、ID Path RMS差中央値はtrain 0.139 m、validation 0.140 m。同一goalのseed変更差はそれぞれ0.028 m、0.028 mで、比は約5.0倍。分母へepsilonを足していない。goal効果がseed効果を上回るstateは64/64。

同距離左右goalのPath endpoint移動方向はnear・farともtrain/validationで全32 state × 8 seedで整合した。IDの同方位0.2→0.4 mへの延長も全件でendpoint距離が増えた。ID goal sweepは64/64 stateで複数Path mode（中央値4）を生成した。一方、同一ID goalの8 seedは平均約1.25 modeで、前回K=8 collapseと整合する。

far-OODは方向反応を保つ一方、endpoint誤差と0.2 m未達率が増えた。DEM mask・範囲・footprint・高さ不連続のscreenは別途保存した。screen通過は`geometry_screen_valid`であり、物理的到達可能性を意味しない。

## Gates と設計判断

- G-ID: **pass**。ID goalの位置・方向・距離・Path modeはtrain/validationで一貫して反応。
- G-OOD: **near_ood_only**。near-OODは維持し、far-OODで誤差が増える。validationのboxes supportはない。実走行は未確認。
- G-Diversity: **pass**。goal sweepは64/64 stateで複数mode。
- G4内部activation: goal-conditioning collapseが0件のため**未発火**。hookは実行していない。

validation選択には`boxes`が存在しないため、全7 terrainへのvalidation一般化は主張しない。`unsupported` goal stratumは固定sweepに含めておらず、性能評価はNAである。

主な推奨は**D1 local subgoal proposalをID/near-OODへ制限して検討し、D3 support guardを併用**すること。IDでもgoal無視というD2の根拠はなく、D4 maneuver tokenは今回の結果から優先しない。offlineだけで「任意goalへ到達するplanner」とは認定しない。

## 10の問いへの回答

1. train goalは半径0～0.8 mの保存領域を持ち、方位は全周に例があるが、横方向は薄い。分布図とterrain別表を参照。
2. 後方sector 29.1%、左右sector合計5.3%、短距離<0.2 m 34.7%、長距離≥0.6 m 30.7%。
3. IDでendpointはgoalへ平均約6 mmのstate中央値誤差で移動した。
4. ID goal効果はsampling効果の約5倍で、64/64 stateで上回った。
5. near-OODは約4.4～4.5 cmの誤差、far-OODは約17 cmへ悪化した。
6. goalを変えると64/64 stateで複数のplanned Path modeを得た。
7. 前回同一goal K=8のcollapseは主にsampling多様性不足を示し、ID/nearでgoal入力は無視されていない。
8. local subgoal sweepはoffline Path多様化に有望。ただし実走行のmode・outcomeへ保存されるか未確認。
9. 今回のID/near結果だけでは再学習・maneuver conditioningを優先しない。far-OOD運用にはtrain support guardが必要。
10. 承認後の提案は4 train state×4 goal×K2＋nominal duplicate 12＝44 run、約2.32時間、GPU約5.55 GiB、保存量上限約0.64 GB。goal-conditioning collapse対象は0件のためNA。

## 将来のsimulator pilot案

[simulator-pilot-proposal.json](simulator-pilot-proposal.json)に短prefix・4 terrainの4 state、logged/ID/near/farのscreen-valid goal、seed、重複control、実測に基づく時間・容量、測定項目を固定した。今回このpilotは実行していない。offline結果のレビューと別承認が必要。

## 図とraw

11枚の要約図を同ディレクトリに、64 stateのresponse mapを`/tmp/eatnav-candidate-coverage/goal-ood/response-maps/`に保存した。per-path結果と推論NPZは`/tmp/eatnav-candidate-coverage/goal-ood/`にある。stateを統計単位とし、goal/seed pairを独立標本とは扱わない。欠損支援stratumはNAとする。
