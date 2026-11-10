# model2000再収集・学習・評価 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for implementation in this session.

**Goal:** 承認仕様に従い新Actorで再収集・scratch学習・精度/B評価・新規branchへのpushを完了する。
**Architecture:** 前回pipelineを継承し、policy identityと対応するActor観測/actionだけを版管理して置換。新rootへ全出力を保存する。
**Tech Stack:** Python, PyTorch, IsaacLab/Docker, FAST-LIO, Git.
**Spec:** `../model2000-pose2s-retraining-spec-v0.1.md`（2026-11-10ユーザー承認）

## Global Constraints

旧raw・重みを学習へ混ぜない。DEM4Hz/1frame40×40、14候補、physics.005/control.05秒、1330親、最大30台。RealSenseなし。新branch通常push。既存成果上書き禁止。

## Review Focus

- 間違うActor/env/agentを拒否し、runtime identityを一貫させる。
- 旧legacy actuator指定を勝手に物理4.0へ移行しない。
- manifestの親/候補/seed/splitが前回と一致し、raw identityだけが異なる。
- 終端/reset/FAST-LIO欠測を正常な2秒教師へ混入させない。
- 厳密精度未達のB探索を達成済みと表示しない。

### Task 1: policy交換と小規模検証
- [ ] 対応3ファイル固定、純Python設定loader/identity検査の失敗テスト→実装→pass。
- [ ] pipeline staging/collector/selectorのActorとagent読込を更新。profileを新identityへ変更。
- [ ] 前回manifest、geometry、physicsの差分監査。2親headless→FAST-LIO→dataset検証。
- [ ] 検証pass後に新設定を固定・commit。

### Task 2: 本収集・dataset・学習・精度
- [ ] 前回manifestと同じ1330親で収集（最大30台）、raw hash/sealを監査。
- [ ] 実FAST-LIO処理、dataset生成・監査、完成通知。
- [ ] seed123/batch128/max200のscratch学習、validation選択freeze、final精度/baseline/推論時間を保存。

### Task 3: B評価と公開
- [ ] 前回B manifest350走行を新Actor/予測NNで評価し、探索的結果として比較。
- [ ] checkpoint/schema/normalization/14候補サンプル、SHA、再現手順を整理。
- [ ] 検証と独立コードレビュー、巨大/秘密ファイル検査、通常push、remote commit照合。

## Execution ledger

Pre-flight: Task1→Task2→Task3はprofile/manifest/checkpoint identityを共有。旧Actor hashを残したgateをすべて更新し、旧raw再利用は禁止。承認済みのため通常実装の再承認は求めない。

Task1: policy identity/paired settings tests RED→GREEN; related suite43pass. First prototype failed on IsaacLab None→float config API; failure preserved at prototype/. Explicit literal actuator assignment fixes API without migrating legacy limits. prototype-r2 produced10/10 valid windows with actual FAST-LIO and positive finite masses/inertias. Full pytest has28 dependency-related collection errors listed in test-limitations.txt. New 30-env capacity batch in progress.
