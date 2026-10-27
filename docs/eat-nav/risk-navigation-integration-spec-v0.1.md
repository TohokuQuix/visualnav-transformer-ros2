# 学習済みEvaluatorのGUI走行接続 v0.1

2026-10-21、利用者が依頼した「学習済みEvaluatorを候補経路選択につなぎ、実画面で走行確認する」実行仕様。

- 新規ROS-freeスクリプトで既存Isaac環境・PureFollower・固定RL Actorを利用する。
  収集binding対象sourceと過去sealed成果物は変更しない。
- checkpoint: current datasetのDEM＋State / optimization seed123、best epoch159。
  SHA256 `924f6438ca7addb94f48dd358b16fd6cc182d75004c3505340dbe456f5f4d0e7`。
  正規化・4 event head・progressの単位は保存済みcheckpointから復元する。
- 入力: 250ms間隔のDEM4枚、Actor pre-noise22値、body gravity/velocity/angular velocity9値、
  学習と同一の4候補・方向・mask。Actorの速度司令成分を勝手にゼロ化しない。
- 4候補の順序: forward direct、forward -30、forward +30、reverse direct。
  20Hz制御、2秒ごとに再選択して新しいPureFollowerへ経路を渡す。
- 選択式: 候補endpoint方向へ予測10秒progressだけ進んだと仮定したgoal距離減少proxyから、
  `2*tip + 0.5*near + 1*linear + 0.5*turn` を減算し、最大値を選ぶ。
  progressは0〜10mにclipし、同点は固定順序の先頭。係数は探索的な固定値で、
  校正済み事故確率・安全制約・予測到達位置を意味しない。全候補が高リスクでも相対比較する。
  入力・予測が不正、head未対応なら停止する。
- 学習は同一candidateを10秒継続したoutcomeを対象としている。
  今回のオンライン切替は条件を拡張する接続試験で、性能改善の正式評価ではない。
- K-Rail level4、front-foldedの既存初期化、1 env、最大30秒。
  初期poseの前方4mをgoalとし、goal25cm以内・自然termination・画面close・期限で終了する。
  転倒・スタックを誘発する操作や物理変更は加えない。
- seed2191202837は新規開発用。pilot・Motion14・batch01/02/03の120 base seedと、
  地形生成tile0..19、command/initial env0..3の3360派生seedから独立。
  照合は固定registry metadataのみ。reserved testのtrace・ラベル・coverageは開封しない。
- GUIを有効にして実走行を表示する。選択候補を緑の経路で表示し、別パネルへ4 headとprogressを表示。
  走行終了後は物理をpauseし60秒間結果表示を残す。リプレイではない。
- 各意思決定の入力tensor・raw DEM・全候補予測・選択・実経路と、各20Hzの姿勢・指令・terminationを保存。
  現在のログを固定candidate教師datasetへ自動追加しない。再学習に使うときは切替policyとの対応を別途扱う。
- checkpoint/Actor/sourceを新規temporary stageへコピーしてSHA照合し、起動条件を記録。
  container内timeoutで実プロセス群に期限を適用。失敗attemptを残し、修正後の再実行は別attemptへ保存する。

実装: `scripts/eat_nav_risk_navigation.py`、`eat_nav_risk_navigation_sim.py`、
`eat_nav_risk_navigation_run.py`。CPU推論用のthread数以外のTorch全体設定を変更せず、
GPU物理・遅延bufferへCPU学習時のdeterministicフラグを波及させない。
