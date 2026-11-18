# 狭通路5field + 元7field・最新NN移行パッケージ

2026-11-18の最新NN。scratch seed123、validationで選択したepoch21。
checkpoint SHA256: `a9fddc4e05a846762277de2ea06ad3efba0016ae64e5c237ad64f54ca2ca22c6`

## 別PCでの確認

Python 3.10、PyTorch 2.7.0、NumPyが動く環境で、リポジトリルートから実行してください。
CUDAは確認処理には不要です。学習時のPyTorchは2.7.0+cu128です。

```bash
python checkpoints/eat_nav/narrow_old7_mixed_20261118_seed123/verify.py
```

全同梱ファイルのSHA、checkpoint内のschema/profile/正規化と14候補のCPU推論を確認します。
この合成入力による確認は精度評価ではありません。

## plannerへの接続

`checkpoint.pt`をNN checkpointのパスに指定してください。正規化はcheckpoint内の
`normalization`を使います。旧NNの正規化と組み合わせないでください。
モデル定義と前処理の凍結ソースは`inference_src`に含めています。
既存インストールに同名モジュールがある場合は、このソースを優先して使ってください。

```python
import sys
from pathlib import Path
import torch

bundle = Path('checkpoints/eat_nav/narrow_old7_mixed_20261118_seed123').resolve()
sys.path.insert(0, str(bundle / 'inference_src'))
from visualnav_transformer.eat_nav.single_pose2s_model import predict

payload = torch.load(bundle / 'checkpoint.pt', map_location='cpu', weights_only=True)
# result = predict(payload, causal_input_batch, device='cpu')
```

入力はDEM `(B,3,40,40)`、state/mask/age各`(B,27)`、DEM context`(B,6)`、
経路`(B,K,8,2)`、前後方向`(B,K,1)`と取得時刻情報です。Kは1または14です。
詳細は`single_pose2s_contract.py`の`validate_input`と`single_pose2s_model.py`の`predict`を参照。
出力は起点のfull-body座標における2秒後の位置[m]と相対姿勢です。
モデルの予測時間は常に2秒です。plannerのcや更新間隔はcheckpointの外側の設定です。
直近の外側開放L字走行はc=0、全14候補2秒更新、その他は以前と同条件でした。

## 同梱内容と制約

- checkpoint、train-only正規化、学習設定・履歴・完了記録
- 推論に必要なモデル・入力契約・幾何・14候補のソース
- `evaluation-report.md`: 学習後オフライン評価と限定条件
- `manifest.json`: 元のバイト列を照合するSHA256一覧

データセット、Actor、Isaac/ROS環境、FAST-LIO本体は同梱していません。
Actorを使う場合は入力契約の既存Actor/profileとの一致を保ってください。
このパッケージは既存planner全体の移行や実機での動作検証を意味しません。
