# 実機PCとの仕様共有・確認票

対象：[今回の仕様書](../single-dem-pose2s-spec-v0.1.md)。状態：仕様承認待ち。新モデル・新dataset・新推論CLIはまだ存在しない。本票は新しいsensorや実機走行を要求するものではない。

## 確定済み

- DEM：4m×4m・0.1m・40×40、**4Hz更新・NNへ1frameだけ**。約3Hzへ変更しない。
- Actor：ユーザー回答により旧R4*/R5 sim用`model_10000.pt`、SHA256 `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`。
- mode=0・毎周期Actor実行。初期姿勢以外に前π/後0固定は行わない。
- goalはNNへ入れない。未来観測・実指令は入力禁止。
- 予測精度未達のまま経路選択へ進まない。

## 承認を求める提案

1. train350＋validation490＋final490＝1330試行（予定6650区間）。finalの各490組に5独立試行を割り当てる。
2. 各組5有効区間・5独立親以上、**最大絶対誤差**で左右10cm・前後30cm・RPY各10degを判定。全490組passが精度達成。高さは参考値のみ。
3. 旧sim followerのnominal追従：lookahead.5m、最大.30m/s、20Hz。自動退避・boostなし。実機のcycle follower .8mと同一と扱わない。
4. 精度達成後の最初の選択関数：XYゴール進捗、予測重力傾斜60度超を除外。他の旧planner制約なし。
5. DEM age上限.50秒、現在state age上限.20秒を入力品質判定として扱う。plannerの追加停止制約ではない。

これらは未承認の設計案。修正希望がある場合は収集・学習前に仕様へ反映する。

## 実機側で確認すること

| 項目 | 確認事項 | 現状 |
|---|---|---|
| 地図 | 4m四方を実際に観測/出力できるか、40×40/0.1m、elevation_inpaintedと元elevationの区別 | 旧bagは20×20。40×40へ単純拡大して有効観測にしない |
| 地図時刻 | 4Hzの実更新stamp、取得/受信時計、t0でのage、地図pose/circular buffer | 旧bagの約3Hzや時計差を今回の確定profileに流用しない |
| IMU | sensor→body TF、gravity/gyroの符号・軸・有効性、covariance−1の意味 | 数値が存在するだけで有効としない |
| joints | RR/RL/FR/FL角度の0点と符号、FR/RR/FL/RL sprocket軸速度の対応、単位rad/s | 水平外向0・上90の物理定義は確定。encoder変換は別確認 |
| odom | twistのchild_frame、基準点、観測可能成分。planarの未観測列はmask | root真値や0を観測済み値として補完しない |
| tf | 地図・IMU・odom・chassis/bodyの変換、取得frame→t0の相対pose | 絶対地図原点をNNへ入れない |
| mode | 実機のfold mode sourceと意味 | sim0はユーザーが選んだ実験条件。実機未記録を0に補わない |
| 制御差 | 旧sim Actorと実機bridge処理、.5mと.8m followerの差 | 今回は旧sim profile。実機同一とは主張しない |
| モデル | 既存R4*/R5・moving-stateモデルとschemaが異なること | 新モデル完成後、専用loaderとshadow例を共有 |

motor current、電圧、追加加速度sensor、DEM履歴は初版必須入力に追加しない。effortはraw保存・原因解析用で、単位未確認のまま新NNへ入れない。

## 別PCで今回の仕様だけを取得

現在の走行branchを上書きせず、repositoryが正しいことを確認して別worktreeへ取得する。

```bash
git remote get-url origin
# TohokuQuix/visualnav-transformer-ros2 を指すことを確認
git fetch origin exp/eatnav-single-dem-pose2s-spec-20261103
git worktree add --detach ../eatnav-single-dem-spec-review origin/exp/eatnav-single-dem-pose2s-spec-20261103
cd ../eatnav-single-dem-spec-review
git rev-parse HEAD
sha256sum checkpoints/eat_nav/rl_actor/model_10000.pt
cd docs/eat-nav/single-dem-pose2s-20261103
sha256sum -c SHA256SUMS
```

既存review directoryと名前が重なる場合は別名にする。上記はROS起動・駆動指令publish・parameter変更を行わない。参考コードの凍結コピーはdocument attachmentで、実機へinstallするものではない。

返却項目：`reviewed_spec_commit`、`spec_approved`、`max_error_rule_accepted`、`collection_plan_accepted`、`nominal_sim_follower_accepted`、`predicted_tilt_60deg_accepted`、地図/state/TF確認結果、未確定事項。未回答を承認と扱わない。
