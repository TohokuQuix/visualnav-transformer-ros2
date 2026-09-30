# Simulation-clock protocol: T4-A停止報告

判定: **clock_protocol_fail**。`coverage-001`のnominal Pathで40-step smokeを開始したが、logical time 1.60 s（step 12）、branch 4で同一stampのcommandが異なった。指定された停止条件に従い、parity、coverage-003 duplicate、20-run Stage B、Gate Nは実行していない。T5 driverは停止gateを越える経路として未実装のまま残し、今回の実装範囲はT4-Aまで。

## 確認できたこと

- T0でHEAD、source/binary/config SHA、20 runの順序・Path SHA、閾値、予算を固定。旧8 runは使用していない。
- 純粋なclock protocol単体テストは2件通過。実ROSの1-branch smokeは5 stamp（1.00–1.20 s）でbridge／TF／Path／commandが整合し、command拒否は0。
- production followerを`use_sim_time=false`とした既存fixtureは7/7完全一致。1-branchのsim-clock first commandは既存production previewと1e-5以内で一致したが、8-branchのT4-A全体は不合格。production source／binary／parameter SHAはT0から不変。
- 8-branchの実simulator smokeはstep 12で停止。branch 0の診断ログは14 distinct stamp、2523 callbackを記録し、最後のmotion-window elapsedは最大0.60 s、recoveryなし。停止条件のbranch 4についてcommandの2値そのものは保存できていない。
- runnerログでは`different commands at same stamp 1600000000 branch 4`が最初のprotocol error。停止処理中の手動interrupt後にdocker cpが失敗したためstatus.jsonの二次エラーとは区別した。

## 原因候補

production followerはTFを`tf2::TimePointZero`で引く。`/clock`だけ先に新stampになり、TF配送前のcallbackが前step poseを使い、その後同一stampのcallbackが新poseを使った可能性がある。複数processへの`/clock`／TF配送順も未検証。branch 4のTF取得stampとcommand値は現ログにないため、因果は確定していない。control則・timer・thresholdは変更していない。

## 証拠と制限

- raw log: `/tmp/eatnav-candidate-coverage/sim-clock-long-horizon/t4/runs/t4-00/runner.log` (SHA256 `ab0ef251cec02c7e459a5a06c09508956466202e6257312867ef5e2ccf504124`)
- bridge log: `/tmp/eatnav-candidate-coverage/sim-clock-long-horizon/bridge.log` (SHA256 `14f1905342c36908e9f84807dc7b76c265ed226fff94ecf26e067c4e1e534e40`)
- branch 0/4 follower log: `/tmp/eatnav-candidate-coverage/sim-clock-long-horizon/follower-logs` (SHA256 `73188759d3e260ce151864fe5cc0c4e54bcd6f6759664dda0851f9a7eb590b7c` / `ffb8c2029dec93b6438e3ebdb5ed636f18eb60803b811868c6b8e7f0bdfc95a6`)
- 120-step clock motion history、全node clock一致、40-step parity、duplicate、T5/T6は未検証。

次の変更は、branch 4を含む全branchでTF lookup stampとcommand値を同一callback単位で記録し、TF配送順のraceを確認したうえで別途判断する必要がある。今回のgateではprotocolを変更・再実行しない。
