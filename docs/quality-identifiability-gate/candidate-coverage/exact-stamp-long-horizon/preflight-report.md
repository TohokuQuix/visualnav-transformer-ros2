# Exact-stamp TF protocol: preflight

Branch `exp/eatnav-candidate-coverage`, base HEAD `4adc4fc101300b30c214dafce31ee0307f220baa`; worktree was clean before implementation. The production follower source, header, binary, and `onix.yaml` retain their recorded SHA values in `source-provenance.json`. The simulation-only copy is built from the logging diagnostic follower and its patch is retained at `/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/exact-follower.patch`.

The follower requests TF at the logical ROS clock stamp with zero lookup wait, checks that returned TF stamp is exact, skips callbacks while TF is absent, and latches each published stamp. Path activation/update and clock rewind clear the latch. The bridge accepts one exact-stamp command per branch and counts stale, future, duplicate, pre-Path, and other rejections. The control law, speed, motion-window threshold, production follower, checkpoints, candidates, Path files, seeds, and frozen order are unchanged.

T1 red: three race regression tests failed against the old path (missing exact gate/copy, duplicate command accepted). T3: 16 focused tests passed (raw log `/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/t3-tests.log`), C++ follower copy built, Python syntax passed, and archived seven production fixtures remained exact with unchanged production SHA. T4-A delayed TF by 180 ms: three callbacks skipped old TF, then one exact-stamp command was published/accepted; subsequent callbacks were latched. T4-B: eight branches × 20 steps, one command per stamp, no mismatch or recovery. T4-C: coverage-001 40-step parity passed on 19 stored signals with exact equality. T4-D: coverage-003 three 120-step fresh-process repeats had XY RMS 0 m, identical DEM and recorded Actor/Critic/command/reward, with motion-window not ready and no recovery. T4-E: the old 20-run identities, order, candidate IDs, seeds, and Path SHAs match exactly. Compact JSON files here give the measurements and raw SHA links.

Stage B runs from `/tmp/eatnav-candidate-coverage/exact-stamp-stage-b`, in detached tmux session `eatnav-exact-stamp`, with `status.json`, `runner.log`, atomic per-run markers, input/output SHA, and a 90-minute/250 MB budget. Closed follower logs are gzip-compressed with raw and compressed SHA. No old eight-run result is mixed into this protocol.

Check and reconnect:

```bash
tmux ls
tmux capture-pane -pt eatnav-exact-stamp:stage_b
tmux attach -t eatnav-exact-stamp
cat /tmp/eatnav-candidate-coverage/exact-stamp-stage-b/status.json
tail -n 100 /tmp/eatnav-candidate-coverage/exact-stamp-stage-b/runner.log
```
