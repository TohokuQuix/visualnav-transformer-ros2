# 実FAST-LIOと最大30台収集への変更

ユーザーは「1回あたり30台」「FAST-LIO本体を実行して収集」を選択した。車体真値をFAST-LIOと呼ぶ方式は採用しない。dataset完成時に一度報告し、dataset本体はGitHubへ入れない。

## 入力と取得元

新input/model schemaは `eatnav_single_dem_pose2s_fastlio_input_v2` / `eatnav_single_dem_pose2s_fastlio_model_v2`。state27の18:21はFAST-LIO poseから因果的に差分した車体基準xyz速度[m/s]、21:24は同様の車体角速度[rad/s]。車輪換算速度ではない。関節回転速度14:18は引き続き別入力。27成分を必須とし、FAST-LIO欠測は正常なゼロ速度で補わない。

実topicは `/Odometry`、型 `nav_msgs/msg/Odometry`。実機で使用される個別forkの確認とは区別し、このPCの `/home/ubuntu/onix/dep/FAST_LIO/src/laserMapping.cpp:633` でposeのみ代入、twist未代入を実ファイル確認した。このため常にゼロのtwistを使用せず、受信済み過去2時点のpose差分を用いる。並進は最新推定車体軸へ回転、角速度はSO(3)差分。時刻・受信順がt0以前の観測だけで生成、age上限.2s、差分間隔上限.25s。地図絶対位置はNNへ入れない。DEM取得frameからt0への相対変換にも、利用可能なFAST-LIO poseを使用する。教師は別保存のシミュレータroot状態から作る。

## 実装の固定

FAST-LIO source repository HEAD `842cae773d1478ab551ece358be18d8d7247920e`。実行binary SHA256 `d91e9e2806280a56070e55fec2490f754bdbd8e7c85f362911c887bb2c7e026e`。設定 `config/eat_nav/fastlio-sim.yaml` SHA256 `46e0f4a9aaa8d6542665c988d586b7090fea352751a9f3d285d8adf632071ea8`。設定・binary・sensor・出力ファイルhashを親試行ごとに記録する。ROS_DOMAIN_ID=77、localhost、CycloneDDSで隔離し、各親試行の `/Odometry` を固有topicへremapして混線を防ぐ。実機指令はpublishしない。

## シミュレーションのセンサ条件と制約

学習NNへ点群は入れないが、実FAST-LIOを実行するための内部観測として点群を生成・一時保存する。これはユーザーが新たに指定したFAST-LIO実行に必要な変更で、従来の点群を収集しない方式とは異なる。

LiDARは16垂直方向(-40〜20deg)×512方位、360deg、上限12m、10Hz。静的な当該地形meshへraycast。車体rootにIMUとLiDARを共置しextrinsicはidentity。自己遮蔽・他ロボット・ノイズをモデル化しない。scanは終端での理想瞬時観測で、header=t-.1s、全point offset=.1sとして実取得時刻tを維持する。Velodyne互換PointCloud2のtime/ring付き形式で実FAST-LIOへ渡す。MID360の実走査や実extrinsicが再現できたとは主張しない。

IMUは20Hz。姿勢から重力を車体軸へ変換し、直前との速度差分による区間平均加速度と重力からspecific forceを生成、角速度は物理状態を車体軸へ変換する。未来状態を使わない。実IMUの高周波・ノイズ・バイアスを再現したとはしない。

初期化だけ2秒settling+2秒センサ／LIO初期化を追加。その後の10秒・5区間は停止やresetなし。FAST-LIO入力にはその初期化観測を含める。過去収集のwheel-odom profileと新profileを混ぜない。終端後のreset状態をLIOへ入力しない。

センサ記録後、実FAST-LIOプロセスへ時系列順にreplayして出力を保存する。利用可能時刻はcallback時点までに解放したセンサ時刻として記録し、受信後の未来poseで過去入力を埋めない。これは同じ因果的センサ列から推定する処理だが、オンライン実機の計算遅延分布の再現ではない。

## 30台のバッチ設計

同じfield/levelの残り親試行を最大30台へまとめ、各親のrole/seed/候補列を保持する。train10+validation14+final14=38親/cellのため、未収集cellは30+8台となる。端数を30へ水増しせず、490組や独立試行の条件は変更しない。物理dt.005s・制御20Hz・DEM4Hzを維持し、倍速指定で物理を変えない。

## 保全と共有

旧wheel-odom収集は区切りで停止・保存済み。新実FAST-LIOデータは `/home/ubuntu/eatnav-single-dem-pose2s-data-20261103/actual-fastlio-v1/` 配下へ保存し、旧データを上書きしない。manifest、条件別件数、欠測理由、実行コマンド、hashと保存場所を共有する。dataset本体と大量点群は通常Gitへpushしない。

実機互換性未確認、シミュレーション先行の承認は維持。精度未達なら経路選択へ進まない。dataset生成・監査終了時に通知し、学習未完を学習済みと報告しない。
