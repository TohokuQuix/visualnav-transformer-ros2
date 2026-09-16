# baseline ROSノード・bag・RVizのオンライン確認

## 起動

このPCの検証用CPU環境 `/tmp/nomad-interface-venv` と保存済みgoal.jpgを使う。学習モデルや速度指令ノードは変更しない。

```bash
env -i HOME=/home/ubuntu USER=ubuntu PATH=/usr/bin:/bin \
  DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority XDG_RUNTIME_DIR=/run/user/1000 \
  bash /home/ubuntu/visualnav-transformer-ros2/scripts/view_krail_baseline_bag.sh
```

- localhost限定、ROS_DOMAIN_ID=188。他のロボット運用ノードと分けて確認する。
- 実際のnavigate_robotノード、RViz、指定bagの等速ループ再生を起動する。
- カメラ・/tf・/tf_staticのみ再生し、--clock 50とuse_sim_timeを使用。
- goalはbag終端の前方画像。採用候補1本を緑色のPathで表示する。8候補のオンライン表示ではない。
- RViz Fixed Frame=map、俯瞰表示。PathのDurability=Transient Local。
- CPU版PyTorchで推論し、OMP/MKLスレッド数は1。GPUを使う起動手順ではない。
- ログは `/tmp/krail-online/{node,rviz,bag}.log`。
- 起動端末でCtrl-C、またはRVizを閉じると、このスクリプトが起動した全プロセスを終了する。
- goal到達はノードの終端状態。到達した後にbagを先頭から確認したい場合は全体を再起動する。
- /tmpの検証環境が削除された場合はPython環境を再構築する必要がある。このスクリプトはこのPC用。

追従ノードや速度指令は起動していない。画面上のPathは各観測からの短期予測であり、車体は記録された遠隔操作の動きを再現する。

## 検証結果（2026-09-16）

最終設定は `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`、等速再生、CPU推論。起動スクリプトに反映済み。

- 実際のROS Pathを55秒間監視: 144件受信、有効経路142件、空Path2件。
- 有効経路は全てmap frame・9点・有限座標。最大観測年齢0.269秒。
- 別の20秒間の画像監視で200枚受信、画像stamp差は連続して約0.1秒。画像ageは0.061〜0.089秒。
- RVizの起動・OpenGL初期化を確認。保存設定では生成Pathを緑色、車体座標をTF表示する。
- 監視記録: [online_monitor.json](online_monitor.json)。[監視スクリプト](online_monitor.py)はこのPC用の出力先 `/tmp/krail-online/monitor.json` を使用する。

初期のFast DDS環境では画像受信欠落が多く、0.5倍速でも20秒に18枚しか届かなかった。Cyclone DDSへの切り替え後、等速で上記の受信数となった。これはこのPC・このbagでの結果であり、実機のDDS環境でも同様かは別途確認する。

未検証: GPU版PyTorchによる推論、実機追従、goal到達停止の妥当性、長時間の安定動作。今回、ROSノードのモデルや停止期限は変更していない。
