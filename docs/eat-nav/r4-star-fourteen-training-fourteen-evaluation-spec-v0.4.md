# R4*: 学習・評価とも前後7本ずつ v0.4

2026-10-27 JST。利用者の最新指示でR4*とR5の評価を6候補から14候補へ変更する。v0.3の14候補学習・収集1120予定・seed分割・教師・12モデルの学習設定はそのまま継承する。収集済みデータを再収集せず、再学習回数も増やさない。

評価候補は既存データと同じcanonical順のforward直進/−30/+30/−60/+60/−90/+90、reverse直進/−30/+30/−60/+60/−90/+90。既存bankの円弧0.75m・8点をそのまま使い、学習と同じ14本を同時採点する。同点はcanonical順、前後進判定はdirectionで判別し、reverse_directの旧slot番号を固定判定に使わない。

R4*はlinear headのみ有効、転倒/near/turn予測と実測45度STOP/35度復帰は無効。R5は同じ14候補学習済みcheckpointを再利用し、linearにturn重み0.5を追加。再学習なし。他のSTOP/復帰/保持/margin/前後進確認、Actor、地形、goal、追従、継足し、20Hz/2秒更新、最大60秒、自然85度termination、boundaryを変えない。両条件とも全14候補除外で予測STOP。

R4*とR5は各36予定のモデルseed/走行seed/順序を維持する。R4*後にR5を実行し、同じcheckpoint path/SHAと14候補全入力・全初期stateを照合する。R4*と旧R4の比較ではcanonical slot0/1/2/7の4候補と共有DEM/body/Actor入力を照合し、追加10候補はbankと照合する。異なるcandidate数を理由に共有状態の差を黙認しない。保存入力推論・全step guard・STOP3軸0を再現監査する。

実行中の収集GUI/collector sourceは変更しない。hostの次のsource検証境界で切り替え、完了attemptのseal・コマンド・ログを保持して次の未処理indexから再開する。変更前plan/source/bindingを別名で保存し、評価候補とhost import/runtime staging/報告だけの改訂を記録する。学習結果がまだない段階でcheckpointを捏造しない。失敗N/A・test非開封の規則を継承。

歴史的な保存先directory名のsix-evaluation部分は保持し、実行manifestにtraining/evaluation candidate_count=14、candidate_orderを明記する。directory名を実際の候補数の根拠にしない。
