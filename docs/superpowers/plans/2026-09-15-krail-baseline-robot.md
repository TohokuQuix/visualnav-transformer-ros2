# K-Rail baseline Robot Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 承認済み9月4日baselineを既存ROS接続ノードで起動できる設定を用意する。
**Architecture:** 標準NoMaDと共通接続コードは維持する。専用YAML、重みの出典、起動・検証手順を追加する。
**Tech Stack:** Python、PyYAML、PyTorch、ROS 2 Jazzy。
**Spec:** [承認済み仕様](../specs/2026-09-15-krail-baseline-robot-design.md)

## Global Constraints

- branch: feat/krail-baseline-robot。main・条件付きモデルブランチを変更しない。
- context_interval_sec=0.25、action_scale_m=0.05、入力front、出力map Path。
- ネットワーク変更なし。goal画像は利用者が明示指定する。

## Task 1: 専用設定・重み・手順

- [x] 学習YAMLからモデル設定を抽出し config/krail_baseline_model.yaml に保存。
- [x] 共通設定を継承した config/krail_baseline_robot.yaml を作成。topomapはnull。
- [x] 重みを model_weights/krail_baseline_20260904_latest.pth にコピーし、承認済みSHA256を照合する。Gitには含めない。
- [x] docs/krail-baseline-robot.md に配置、goal指定、起動、bag検証、未検証項目を書く。

設定のみの変更なので実装をなぞる単体テストは追加しない。既存コードによる読み込みと実重み推論で検証する。

## Task 2: 検証・記録

- [x] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests` をROS2環境で実行。
- [x] 実重み・4枚の実画像・乱数seed42で、既存navigate.pyのNoMaD分岐とNoMaDPredictorの8点を比較する。np.testing.assert_array_equalで検証。
- [x] load_settingsで相対パスを解決し、実画像列を指定したRobotNavigationNodeの構築・終了を確認。
- [x] 検証環境と結果を手順書へ記録。実機とTF教師同期の未確認事項を明記。
- [x] `git diff --check` 後、専用設定とドキュメントをコミットする。
