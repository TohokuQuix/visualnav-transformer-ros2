# 速度司令を除外したEvaluatorの再学習・同条件GUI再走行

2026-10-21 JST。利用者が承認した修正範囲を記録する。

原因は、停止後の初期入力で学習した速度司令3成分がtrain全例で0だったこと。
std floor0.001に対して走行中の値0.25〜0.31が正規化後250〜313となり、旧checkpointの出力を飽和させた。

実装条件:

- Actor pre-noise22成分のうちindex6/7/8（現在の速度司令）を、Evaluatorの学習・推論の両方から除外する。
  状態入力は残り19＋body9の28次元。raw観測22値とActorへの実指令は保持する。
- 新しいnormalization/checkpoint/model schemaを作り、旧31次元checkpointとの混用を拒否する。
  正規化はtrainのみでfit。学習と推論で同じkeep-index契約を使用する。
- networkの変更は最初のstate Linearを31→28にする部分のみ。
  共通成分の初期重み・biasは、同seedの旧architectureから保持列を切り出して揃える。
  DEM/path encoder・fusion・head・loss・重み・optimizer等は変更しない。
- batch01＋batch02の固定train96/validation32 StateGroup・24/8 base seedで再学習する。
  同じ2方式×optimization seed123/456/789、Adam0.001、full batch、CPU1 thread、300 epoch。
  validationで最小のunweighted seed-balanced BCEを選び、同点は早いepoch。
  同じDEM＋State / seed123をGUI走行へ使い、別seedや指標でモデルを後付け選択しない。
- 新旧の定数baseline・classification/ranking/progress・checkpoint再読込を記録する。
  train/validation指標は開発用で、正式test・希少headの正式比較は未実施のまま。
- 旧30秒走行のseed2191202837、K-Rail level4、front-folded、1env、20Hz、2秒再選択、
  初期pose前方4m goal、score係数[2,0.5,1,0.5]、最大600 control stepを保持する。
  GUIの実画面と予測パネルを表示し、終了後60秒pause表示を残す。
- 保存済み旧走行の入力でも新旧モデルを比較し、速度司令変更に対する新モデルの予測不変性を確認する。
  新走行は同一seedだがpolicyが変わるため、初期状態以外の入力まで一致するとは扱わない。
- 走行中の身体状態など、速度司令以外の分布外入力や固定candidate10秒教師とオンライン切替の差は残りうる。
  予測飽和や走行が改善しない場合も、結果をそのまま報告する。正規化値や出力の隠れたclipを追加しない。
- 元dataset、旧checkpoint、過去のsealed報告、収集binding対象sourceと旧実行scriptは変更しない。
  新規scriptと新規出力directoryへ保存する。reserved testは開封しない。

実行順: 不変性・新schema・再読込テスト → 実装レビュー → config/source固定 → 再学習 →
同条件GUI再走行 → 保存入力からの再現・旧走行との比較 → 成果物のsealと報告。
