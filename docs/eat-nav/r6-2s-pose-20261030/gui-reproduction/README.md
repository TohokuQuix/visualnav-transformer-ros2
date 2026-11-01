# 代表GUI再現レシピ（準備済み・未実行）

保存された実評価assignment/manifestから作成したレシピです。全30環境・seed・地形・低レベル制御・モデル・閾値・runtime sourceの同一性を維持します。環境数を1や10へ削減すると動的ランダム化のbatch構成が変わるため、代表1台へカメラを合わせても元batchは保持します。

- pose_q3_2: 移動とSTOPを含む比較例。
- pose_q1_1: group000/004の自然転倒、group020のSTOP後転倒を含む例。転倒を避けられたというデモではありません。
- これは保存済み状態の動画リプレイではなく、固定条件での再走行です。元の定量結果は書き換えません。

## 実行（全定量評価の終了後）

GPU競合を避けるため、現queue中には起動しません。ホスト/コンテナのDISPLAY・X11認証・GPU環境を確認します。未表示を表示成功と扱いません。リポジトリrootで:

```bash
export PYTHONPATH=src:scripts:tests/eat_nav:.
/home/ubuntu/venvs/isaaclab_py310/bin/python scripts/eat_nav_r6_gui_reproduce.py run --recipe docs/eat-nav/r6-2s-pose-20261030/gui-reproduction/pose_q3_2-k_rail_field-l0.json --checkpoint checkpoints/eat_nav/r6_2s_pose_20261030/model_seed123.pt --output /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/representative-gui-q3_2 --focus-env 0
/home/ubuntu/venvs/isaaclab_py310/bin/python scripts/eat_nav_r6_gui_reproduce.py run --recipe docs/eat-nav/r6-2s-pose-20261030/gui-reproduction/pose_q1_1-k_rail_field-l0.json --checkpoint checkpoints/eat_nav/r6_2s_pose_20261030/model_seed123.pt --output /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/representative-gui-q1_1 --focus-env 0
```

出力は新規ディレクトリを指定します。起動helperは正確なdocker起動コマンドを表示しcommand.jsonにも保存します。実走行を始めずコマンドのみ準備する場合は`--dry-run`を付けますが、これはGUI表示の証明ではありません。

実行時のcommand/log/sim-output・可能なら画面画像を永続成果物へ保管し、GUI表示と完了gateを別に確認します。現時点ではレシピのモデルSHA・元manifest SHA・30groupを検証しただけで、GUI表示・再走行完了は未検証です。
