# Simulation clock TF race: 修正案（未実装）

| 案 | 根本原因への対応 | production followerとの差／制御則 | replayとfixture | 実装量・主要test |
|---|---|---|---|---|
| 1. simulation-only exact-stamp TF lookup **推奨** | 対象stepのTF以外をlookupで受け付けない。前step poseを使う経路を直接閉じる | simulation用copyのlookup時刻だけ変更。速度則・stuck則・20 Hz wall timerは維持 | logical stampと一対一。`use_sim_time=false`のproduction fixtureは元binaryで維持 | 小。TF遅延時のtimeout、全8 branch stamp一致、first-command、40-step parity、120-step duplicateを再検証 |
| 2. follower TF-ready acknowledgment後の採用 | follower内のTF buffer到着を確認してからcommandを選ぶ | follower→bridgeの診断ackと採用条件が追加される。制御則は維持 | ack順序を凍結できるがmessageとcommandの紐付けが必要 | 中。ack欠落・遅延・古いack拒否と同一stamp複数callbackをtest |
| 3. simulation step同期adapter | follower callback起動をstep境界へ限定する | wall timer経路をsimulationで置換するためproductionとの差が大きい | 一定のcallback数にできるがfixtureの実行経路が変わる | 大。ROS lifecycle・timer・TF・command全体の等価性test |
| 4. `/clock`/TF barrier | clockを進める前にTF配送完了を保証する協調protocol | follower側ackなしではROSの2 topicを原子的に配送できない | 完全なbarrierを作れれば決定的。現構成への追加が多い | 大。全branch ack、timeout、clock単調性、deadlock test |

**推奨は案1。** 今回のAはROS時刻1.70 sで1.65 sのTFを返し、Bは1.70 sのTFを返した。exact-stamp lookupならAに対応する古いposeは使用されない。TF未到着時の待ち／timeoutに伴うcommand欠損を含めて再評価する。今回の診断ではどの案も実装しない。

Stage B再開前には、simulation-onlyの変更に対するsource/binary SHA監査、production fixture、8-branch 20-step smoke、40-step parity、120-step duplicate、motion-window 6 s非ready、全stampのTF/Path/command整合と旧20-run入力不変をgateとして再確認する。
