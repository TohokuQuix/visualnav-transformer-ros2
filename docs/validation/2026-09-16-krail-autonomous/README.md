# 自律走行bagでのbaseline GPUオンライン検証

対象: `/media/ubuntu/Extreme SSD/20260803_experiment/2026_08_03-10_31_05`

## 起動

他の検証用RVizを閉じてから次を実行する。

```bash
env -i HOME=/home/ubuntu USER=ubuntu PATH=/usr/bin:/bin \
  DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority XDG_RUNTIME_DIR=/run/user/1000 \
  bash /home/ubuntu/visualnav-transformer-ros2/scripts/view_krail_autonomous_baseline_gpu.sh
```

- `--start-offset 25`: bag開始から25秒位置から等速ループ再生する。画像履歴が4枚揃うまで推論は開始しない。
- 再生トピックは `/front_camera/image_raw/compressed`、`/tf`、`/tf_static` のみ。
- 新しい予測は `/nomad/baseline_predicted_path` へ配信する。bag中に記録された `/teaching_playback/path` は再生しない。
- RVizは専用予測トピックを緑のPathとして表示。表示は採用された先頭候補1本。
- GPU baseline、PyTorch2.8.0+cu128、device=cuda、Cyclone DDS、localhost、ROS_DOMAIN_ID=188。
- Python環境は前回と同じ `/tmp/nomad-gpu-venv`。依存環境の詳細は [GPU環境手順](../2026-09-16-krail-outbound/GPU.md)。
- goal画像は今回のbagの最後の前方画像（記録開始から約75.03秒）を抽出した `goal.jpg`。前回のbagのgoalは使用しない。
- 追従ノード・速度指令は起動しない。RVizを閉じるかCtrl-Cで一式終了する。
- ログは `/tmp/krail-autonomous-gpu/`。goal到達による終端停止後に繰り返す場合は一式を再起動する。

## 検証対象と限界

サブクローラの映り込みが少ない自律走行bagで、同じbaselineの生成経路を観察するための検証。画像のcropやモデル構造・重みは変更していない。

経路が変わっても「サブクローラを壁と認識した」ことを直接示すものではない。観測位置、走行姿勢、goal画像、サンプル乱数等も異なる。原因を調べるには同一観測・goal・乱数で画像条件だけを変えた比較等が別途必要。

bagは約75.10秒、前方画像748枚、記録済みPath4件を含む。起動後のROS graphで、専用PathのPublisherはnomad_robotのみ、元Pathと/cmd_vel_autoのPublisherは0件を確認。

## 実測結果（2026-09-16）

65秒間のROS購読で139件受信、有効Path138件・空Path1件。有効Pathは全て9点、最大観測年齢0.491秒（設定期限0.8秒以内）。GPU推論プロセスのメモリ使用478 MiBも確認した。

[全Path座標・時刻の受信記録](monitor.json) / [Publisher確認結果](publishers.json)

ループ先頭へ戻ったときのTF bufferクリア警告は再生時刻の巻き戻りによるもの。図形の妥当性、障害物回避性能、goal到達判定の正しさや実機追従を保証する検証ではない。
