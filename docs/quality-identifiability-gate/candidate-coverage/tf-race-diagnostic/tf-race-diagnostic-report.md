# EAT-Nav same-stamp TF/command race diagnostic

## 判定

**R1: TF delivery race supported.** 凍結済み`coverage-001` nominal、8 branch、20-step上限で1 fresh-process repeatを実行し、logical time **1.70 s**にbranch **4**で同一stampのcommand差を再現した。ここで停止し、repeat 2/3、parity、duplicate、Stage B、candidate rolloutは実行していない。

| | callback A | callback B |
|---|---:|---:|
| sequence | 302 | 303 |
| ROS clock / command stamp (s) | 1.70 | 1.70 |
| returned TF stamp (s) | 1.65 | 1.70 |
| TF x/y (m) | -0.090870 / 0.138552 | -0.093134 / 0.150440 |
| heading error (rad) | -0.193358500 | -0.195948962 |
| command vx / wz | 0.281535623 / -0.096679250 | 0.281288252 / -0.097974481 |

motion historyのsample数とelapsedはA/B間で変わるが、両方でwindow ready=falseであり、recovery分岐は発火していない。Pathは同一SHA `6790f278…`、Path seq=2、mode=`default`、stuck/recovery=false。TFのXY差は0.012102 m。既存パラメータで`wz=0.5×heading_error`、`vx=0.3×(1−|heading_error|/π)`を再計算した残差は両者とも0。controller内部のmotion elapsedはA 0.60 s、B 0.65 sだが、20 s windowは両方ready=falseで制御modeに差はない。

時系列は `/clock` publish開始→AのTF lookup（旧1.65 s）→新1.70 s TF publish→BのTF lookup（新1.70 s）→bridge receive。Aのlookup終了は新TF publish開始より前、Bのlookup開始はpublish終了より後。publish wall時刻だけでなくfollowerが返したTF stampで確認した。図: [timeline](tf-race-timeline.svg)。

## 実行・保存

R0でbranch/HEAD、stateとepisode、Path・NoMaD・RL checkpoint、production/診断follower、bridge/runner、T4 failureログと閾値を固定した。旧branch 4 raw logにはcallback単位のA/B値がなく推定で補っていない。logging-only follower copyのC++差分とbinary SHAは監査JSONに保存した。production source/binary/config、clock protocol、TF lookup、bridge command選択は変更していない。

最初の起動はoutput directory未作成でsimulation開始前に停止した。directory作成だけ修正し同じ凍結入力で実行した。conflict artifactをatomic保存し、全8 follower logのflush/close後にdetached cleanupがsimulatorへSIGTERMを送った。bridgeはSIGTERM後もsocket acceptで残ったため、run完了と全follower終了を確認した後に専用cleanupで終了した。結果retryは行っていない。Docker側status.jsonはcopy前後SHA一致。

実測wall 295.6 s、GPU peak 5551 MiB、raw log/traceは`/tmp/eatnav-candidate-coverage/tf-race-diagnostic`。全branchのcallback JSONLとclock/TF/Path/bridge JSONLをgzip保存し、SHAを`tf-race-source-log-sha.json`に記録した。CPU RAM peakは計測していない。bridgeの最終cleanupではSIGTERMで終了しなかったためSIGKILLを使用したが、全run marker・follower logを閉じた後である。unit testは4件通過し、診断C++ build・Python syntax・原本SHA・8 branch JSON parseを最終検証した。

## 制限

診断ログはcallback時刻に影響しうるため、再現確率を評価するものではない。今回の実行で旧TF→新TFという因果順序は直接観測できた。TF lookup失敗時の例外文字列はcallback JSONLには含めていないが、A/Bはいずれもlookup成功。修正案は[比較表](tf-race-fix-options.md)に示し、実装していない。
