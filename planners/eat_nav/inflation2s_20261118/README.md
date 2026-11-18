# 2秒NN + 全車体endpoint inflation planner

閉じたL字・幅1.12mで実行した2026-11-18版です。最新NN移行ブランチを基にしているため、同じブランチ内にcheckpointとown normalizationも含まれます。

## 設定

全14候補に `score[m] = NNの2秒後goal接近量[m] - 0.5[m] * C` を適用します。
CはNNが予測した2秒後の位置・相対姿勢に現在の関節形状を配置し、全車体投影が占めるセル内の最大inflation costです。既知壁から20cmで1→0に線形減衰します。ロボットの半径を二重加算しません。

- c=0。旧+8cmや0.2O加点はありません。
- 全14候補とSTOPを40 control steps=2秒保持。±60/90度も同じです。
- 従来の傾斜・floor bounds等のallowed maskを保持し、最終選択時に適用します。
- 有限の減点なので壁際候補を必ず除外するhard maskではありません。
- 未観測・範囲外そのものでは減点しません。ただし既知壁の近傍にある未観測セルには既知壁由来のcostがあります。
- 途中のswept pathや将来の関節運動を予測するものではありません。

## 別PCで取得・確認

```bash
git fetch github-pose2s codex/planner-inflation2s-lambda05-20261118
git switch --track github-pose2s/codex/planner-inflation2s-lambda05-20261118
# 通常のcloneでremote名がoriginの場合は github-pose2s を origin に置換
python -m pip install -r planners/eat_nav/inflation2s_20261118/requirements.txt
python planners/eat_nav/inflation2s_20261118/verify.py
python -m pytest -q planners/eat_nav/inflation2s_20261118/test_bundle.py
```

NNの確認は `python checkpoints/eat_nav/narrow_old7_mixed_20261118_seed123/verify.py`。

## 既存plannerへの接続

`runtime`に実験で使用したソースをそのまま同梱しています。ファイル名`rp_change_rule.py`の冒頭docstringは旧O実験の説明が残っていますが、実装は`RULE=goal_progress_minus_endpoint_inflation_v1`です。ソースSHAを維持するためこのsnapshotは変更していません。

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path('planners/eat_nav/inflation2s_20261118/runtime').resolve()))
from feasibility import DEMGrid
from inflation_rule import predicted_costs
from rp_change_rule import score_components

# raw_dem: (3,40,40), 0.1m grid。ch0は取得時root高さに相対、
# NN正規化前のメートル値DEMを渡す。ch0の穴埋めは許容するが、
# ch2は元raw観測support maskを維持し、補間valid maskに置換しない。
# acquisition_root/root: xyz+wxyz。DEM取得時座標を必ず保持する。
# world_collision_points: 現在の全body衝突形状をworldへ変換した点群のlist。
# P: NN予測(14,3), R: NN相対回転(14,3,3)、現在root full-body座標。
grid = DEMGrid(raw_dem, acquisition_root)
geometry = predicted_costs(grid, root, world_collision_points, P, R)
C = np.array([x['C'] for x in geometry])
components = score_components(goal_progress_m, R, current_world_rotation, C)
ids = np.flatnonzero(existing_allowed_mask)
selected = int(ids[np.argmax(np.asarray(components['score_m'])[ids])]) if len(ids) else None
# NoneはSTOP。更新・実行時間はhold_schedule.pyの40stepsに従う。
```

`runtime/navigation_sim.py`が全body poseとcollision形状、causal DEM、NN出力、allowed mask、スコア、ログ保存を接続した実際のIsaac adapterです。`collision_geometry.py`はUSD衝突形状抽出、`kinematics.py`は姿勢変換、`nav_contract.py`は推論/選択の契約です。

Isaac/Actor・ロボットasset・元実験の絶対パス依存の`workflow.py`も出典snapshotとして含めています。これは別PCでそのまま実行できるlauncherではありません。Actor/Isaac/ROS環境は別途必要です。ROSノードへの配線や実機動作は未検証です。

## 検証結果

- 元実験64 CPU tests PASS。
- 独立監査: 223判断×14=3122候補のC/未観測率差0、score/mask/argmax一致。
- 全10動画・31ページ目視: 壁越えを観測しないgoal8、角部内転倒2。
- 同一global seed123のbatched10slotであり、10独立seedではありません。
- 姿勢変動が大きい走行もあり、無接触・連続車体clearanceを保証しません。

`protocol.json`、監査snapshot、geometry calibration、manifestを同梱。動画・raw datasetは同梱していません。監査JSONの絶対パスは元実験の出典を表します。
