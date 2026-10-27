# 現在のdatasetによる転倒・スタック学習 v0.1

2026-10-21、利用者の「現在のデータで学習する」依頼に基づく実行仕様。
追加収集batch03を待たず、batch01＋batch02の固定train/validationでEvaluatorを学習する。
陽性seedの最低coverageは正式比較の条件であり、重み更新の開始条件にはしない。

- 入力: `additional-development-batch02-cumulative-20261021/cumulative`。
  ledger SHA256: `33ed73e061c3a00c5b2bf3176935e4963daa3bcda07455d27035ed46dc42cd64`。
- 固定split: train 24 seed・96 StateGroup・384候補、validation 8 seed・32 StateGroup・128候補。
  pilot、旧Motion14 dataset、reserved testを追加しない。未知ラベルはmaskで除外する。
- 対象: 初期DEM・身体状態・候補経路から10秒の4 eventと1/2/5/10秒progressを予測するEvaluator。
  Actor、経路追従器、経路生成器は更新しない。
- 比較: state＋candidate、DEM＋state＋candidate。各optimization seed 123/456/789。
  定数baselineはalways-event-freeとtrain-prior。seedはデータの独立試行数ではない。
- 既存の承認済みbaselineと同じnetwork/lossを使用。trainだけから正規化・pos_weightを計算。
  Adam、lr=0.001、weight_decay=0、full batch、CPU 1 thread、deterministic、300 epoch。
  主lossはseed平均のmasked BCE、progress/pair lossはそれぞれ0.1倍。
  validationのtrain/validation両方に正負があるheadのunweighted BCE最小epochを採用。
  同値なら最初のepoch。閾値0.5固定。ハイパーパラメータ探索は行わない。
- train/validation予測、raw logits、指標、history、checkpoint、正規化、source SHAを保存。
  checkpoint再読込予測一致と保存予測からの指標再計算を確認し、成果物をsealする。
- 全4 headはtrain/validationに正負があり学習可能。ただし転倒・near-fallのvalidation陽性は
  各2候補・1 seed、turn-stallは1候補・1 seed。これらの指標は記述的・暫定で正式比較N/A。
  linear-stallの開発側coverageは達成しているが、全headで正式testは未実施。
  重み付きBCEのsigmoidを校正済み事故確率とは呼ばない。
- 保存済みdatasetの重み更新中はepoch/lossと実測進捗JSONを表示する。新しい走行は発生しない。
  利用者が希望した画面は新しいシミュレータ走行の実画面であり、既存データのリプレイは作らない。
  後続の新しい走行・収集ではGUIを表示する。今回のfitだけで閉ループ接続済みとはしない。

実装・検証・実行順: 新規scriptのepoch callbackと固定split loaderを検証 → source/configを固定 →
6学習と2定数baseline → checkpoint・予測・指標の再検証 → 結果報告。
既存のsealed成果物と収集runtimeにbindingされたsourceは変更しない。
