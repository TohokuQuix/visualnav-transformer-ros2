# EAT-Nav exact-stamp TF protocol and frozen Stage B rerun

## Decision

- **Clock/TF protocol: pass.** Simulation-only follower copy requests TF at the logical ROS stamp, checks the returned stamp, skips while the exact TF is unavailable, and publishes once per stamp. The bridge accepts one command per branch/stamp. The production follower source, binary, and parameters retain their previous SHA values.
- **Duplicate reproducibility: pass.** In preflight `coverage-003` and in each of the four Stage B states, three independent 120-step nominal repeats have XY RMS **0.0000 m**, below the frozen 0.0105 m limit. Recorded DEM, Actor, Critic, command, and reward signals agree within each nominal group.
- **Long-horizon candidate coverage: mixed four-state oracle evidence.** The fixed NoMaD winner is constraint-valid in all four states and improves six-second ΔV over nominal in three. The bank winner is constraint-invalid in two states; it is higher in ΔV in those two, so those comparisons cannot establish feasible bank superiority. The two states with a valid bank winner show higher NoMaD ΔV and progress. No evaluated candidate stacked or terminated during six seconds.
- **Gate N: inconclusive.** Four selected train states do not establish coverage across terrain, validation, or hardware. The 2-second and 6-second ΔV ordering changes in three states. No Evaluator was used to choose an online candidate, and these results are oracle outcomes rather than selection performance. A separately approved formal validation is the next decision point.

## Frozen six-second oracle results

| State / terrain | Nominal ΔV | NoMaD ΔV / progress m | Bank ΔV / progress m | Bank valid | NoMaD–bank realized XY RMS m | ΔV rank changed from 2 s |
|---|---:|---:|---:|:---:|---:|:---:|
| coverage-001 / hf_pyramid_slope | 1.024 | 0.950 / 0.318 | 1.042 / 0.308 | no | 0.0127 | yes |
| coverage-003 / k_rail_step_field | 1.064 | 1.111 / 0.293 | 1.274 / 0.335 | no | 0.0173 | no |
| coverage-004 / pyramid_stairs | 0.179 | 0.282 / 0.671 | 0.179 / 0.496 | yes | 0.0732 | yes |
| coverage-006 / random_rough | 0.872 | 0.918 / 0.721 | 0.872 / 0.670 | yes | 0.0296 | yes |

The stored per-run and per-state JSON includes H-step advantage, drawdown, recovery, path length, maximum roll/pitch tilt, contact, stack and termination, command difference, realized trajectory difference, and two-second/six-second rankings. Outcome rankings depend on the metric: ΔV and H-step advantage do not always agree. `coverage-001` NoMaD/bank outcomes remain close in command/trajectory space under the pre-existing mode-separation thresholds. The statistical unit is a state, not a candidate pair; no all-terrain or real-robot generalization claim is made.

## Execution and audit

T1 recorded three expected RED failures before implementation. T3 focused tests passed **16/16**, Python syntax passed, and the simulation-only C++ copy built. The prior seven production fixtures remain exactly equal and production SHA was checked again after all runs. The T4-A delayed-TF ROS smoke had three callbacks with no TF at 1.05 s and zero command adoptions; after exact TF delivery it published and adopted one command, with exact production-formula command value. T4-B eight-branch 20-step smoke had one accepted command per stamp, no branch command difference, no stale/future/duplicate adoption, and no recovery. T4-C coverage-001 40-step comparison matched all 19 recorded signal groups exactly. T4-D three fresh-process 120-step repeats had XY RMS 0, no stamp mismatch or recovery, and motion-window `ready=false`. T4-E matched the old 20-run state, candidate, seed, Path SHA, and order. The completed 20-run Stage B used 120 control steps per run and took **2760.68 s (46.01 min)** including the bridge launch. Recorded size at completion was **91,166,209 bytes** against the 250 MB cap; after verified gzip archival of 160 follower logs it is about 69 MB. No startup retry was used.

The normal 120-step endpoint metric uses the existing `endpoint_metrics` function; ΔV is final Critic value minus initial Critic value. Success labels are not inferred from timeout. The four six-second traces had no terminal event. Dataset or model weights, control law, candidate paths, and frozen thresholds were unchanged. Old protocol run traces were not mixed with this rerun.

Full repository pytest collection was attempted with `/usr/bin/python3` and stopped with 20 environment import errors: 18 missing `torch`, one missing `rclpy`, and one missing `geometry_msgs`. Its raw log is `/tmp/eatnav-candidate-coverage/exact-stamp-stage-b/full-pytest.log`. The targeted 16 tests and real ROS/simulator gates passed; this does not substitute for a full-suite run in an environment with both ML and ROS dependencies.

## Artifacts and reproduction

- Small results in this directory: `source-provenance.json`, `t4a-smoke.json`, `t4b-summary.json`, `t4c-parity.json`, `t4d-duplicate.json`, `t4e-input-audit.json`, `follower-callback-audit.json`, `stage-b-runs-complete.json`, `stage-b-results.json`, `final-decision.json`, and `compressed-log-sha.json`.
- Raw preflight: `/tmp/eatnav-candidate-coverage/exact-stamp-long-horizon/`.
- Raw Stage B: `/tmp/eatnav-candidate-coverage/exact-stamp-stage-b/`; its 20 `run-complete.json` markers include input identity and output SHA. `compressed-log-sha.json` records each raw and gzip digest. The original race evidence remains under `/tmp/eatnav-candidate-coverage/tf-race-diagnostic/`.
- Recompute outcomes with `PYTHONPATH=src:scripts /usr/bin/python3 scripts/eat_nav_exact_stamp_analyze.py` from `/tmp/eatnav-vnt-candidate-coverage`. The runner entry points are `scripts/eat_nav_exact_stamp_build.py`, `scripts/eat_nav_exact_stamp_bridge.py`, `scripts/eat_nav_exact_stamp_run.py`, and `scripts/eat_nav_exact_stamp_stage_b.py`. The frozen 20-run identity file is `/tmp/eatnav-candidate-coverage/sim-clock-long-horizon/clock-corrected-run-identities.json`.
- To inspect a detached run while active: `tmux ls`; `tmux capture-pane -pt eatnav-exact-stamp:stage_b`; `tmux attach -t eatnav-exact-stamp`; `cat /tmp/eatnav-candidate-coverage/exact-stamp-stage-b/status.json`; `tail -n 100 /tmp/eatnav-candidate-coverage/exact-stamp-stage-b/runner.log`. The session was stopped after completion and log archival.
