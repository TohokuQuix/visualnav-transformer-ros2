# R5: 14候補・直線＋旋回スタック・実測姿勢STOPなし v0.2

2026-10-27 JST。利用者の最新指示により、評価はR4*と同じ前後7本ずつの14候補へ変更する。候補形状・canonical順は[14候補学習・評価仕様v0.4](r4-star-fourteen-training-fourteen-evaluation-spec-v0.4.md)に従う。

v0.1のその他条件を維持する。直線risk重み1、旋回risk重み0.5、除外は(p_linear>=0.95 OR p_turn>=0.95)かつprogress10<=0.05m。tip/nearは制御に使わず、実測45度STOP/35度復帰なし。最短2秒STOP・再開2回確認、保持4秒・margin0.15・方向2回確認を維持。自然85度終了・waypoint到達基準等も変更しない。

R4*の14候補学習済み12モデルをpath/SHA/正規化ごとそのまま再利用し、再学習・教師追加・閾値変更はしない。両stall headのsupportを要求し、不成立はN/A。同じ36割当・全14候補入力と初期stateを照合する。R4*の学習・14候補評価・最終監査seal完了後にGUIで実行。全head生予測と使用head、全step状態とSTOP3軸0、候補別利用を保存・監査する。転倒・進捗・STOP・観測時間・両stallを対応比較する。

変更前の6候補待機処理は停止し、旧source/planを改訂履歴として保持する。進行中の14候補収集と学習設定は変えない。reserved test非開封。R4*対R5はturn追加効果の比較であり、旧R2と45度STOPだけの因果比較にはしない。
