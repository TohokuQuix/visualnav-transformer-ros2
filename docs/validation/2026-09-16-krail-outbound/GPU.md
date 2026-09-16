# GPUでbaseline ROSノード・bag・RVizを起動する

## このPCでの起動

CPU版の検証が起動中ならRVizを閉じて終了してから、次を実行する。

```bash
env -i HOME=/home/ubuntu USER=ubuntu PATH=/usr/bin:/bin \
  DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority XDG_RUNTIME_DIR=/run/user/1000 \
  bash /home/ubuntu/visualnav-transformer-ros2/scripts/view_krail_baseline_bag_gpu.sh
```

- baseline ROSノードを明示的に `device:=cuda` で起動する。CUDAがなければエラーとなりCPUへフォールバックしない。
- Cyclone DDS、localhost限定、ROS_DOMAIN_ID=188。
- 指定bagを等速ループ再生。再生対象は前方圧縮画像・TFのみ。全ノードでシミュレーション時刻を使用。
- RVizの緑色Pathは先頭候補1本。8候補を同時にROS配信する変更は行っていない。
- goalは保存済みのbag終端画像。モデル・重み・Path仕様はCPU版と同じ。
- 追従ノード・速度指令は起動しない。
- RVizを閉じるか、起動端末でCtrl-Cすると一式を終了する。
- ログ: `/tmp/krail-online-gpu/node.log`、`rviz.log`、`bag.log`。

## GPU確認

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

実行時に `/tmp/nomad-gpu-venv/bin/python` のGPUメモリ使用を確認した（478 MiB）。

## 検証環境

- NVIDIA GeForce RTX 4090 Laptop GPU、driver 580.173.02。
- Python3.12、ROS2 Jazzy、PyTorch2.8.0+cu128、torchvision0.23.0+cu128。
- GPU用venv: `/tmp/nomad-gpu-venv`。ROS用system-site-packagesを有効にしている。
- モデル周辺の依存ライブラリは `.pth` により既存 `/tmp/nomad-interface-venv/lib/python3.12/site-packages` を参照する。この2つの一時環境が必要であり、どちらかが削除されると再構築が必要。GPU環境内のtorch/torchvisionが優先される。
- CPU環境と比較してPyTorch版も異なるため、CPUとGPUだけの厳密な速度比較ではない。

CUDA版の組み合わせは[PyTorch公式の配布手順](https://pytorch.org/get-started/previous-versions/#v280)に従った。

```bash
UV_CACHE_DIR=/tmp/nomad-interface-uv-cache /tmp/nomad-interface-venv/bin/uv venv \
  --python /usr/bin/python3 --system-site-packages /tmp/nomad-gpu-venv
UV_CACHE_DIR=/tmp/nomad-interface-uv-cache /tmp/nomad-interface-venv/bin/uv pip install \
  --python /tmp/nomad-gpu-venv/bin/python torch==2.8.0 torchvision==0.23.0 'numpy<2' \
  --index-url https://download.pytorch.org/whl/cu128
printf '%s\n' '/tmp/nomad-interface-venv/lib/python3.12/site-packages' \
  > /tmp/nomad-gpu-venv/lib/python3.12/site-packages/nomad_shared_dependencies.pth
```

## 確認済み事項

- 実重みをstrictロードし、モデルパラメータがcuda:0にあることを確認。
- 実画像4枚とgoal画像によるGPU推論で8点生成。CUDA同期を含む計測は初回0.296秒、続く3回は0.067秒前後。ROSの画像復号・転送・配信遅延は含まない。
- 新しいPython環境で既存テスト31件成功（テスト内モデルはCPU使用）。これとは別に上記の実GPU推論を確認。
- 実機走行・到達停止の妥当性・長時間運用は未検証。

### ROSオンライン測定

55秒間の実Path購読で144件受信、有効経路142件・空Path2件。有効経路は全てmap座標の9点で有限値。観測から受信までの遅れは中央値0.221秒、95パーセンタイル0.225秒、最大0.421秒。ノードの0.8秒期限内だった。

[GPUモデル確認値](gpu_model_check.json) / [実Path受信記録](gpu_online_monitor.json)

この測定でGPUノード・bag・RVizを同時に起動した。CPU版の過去計測より最大遅延が小さいとは限らず、初回実行・履歴サンプリング・通信などを含むオンライン遅延は単体推論時間とは区別する。
