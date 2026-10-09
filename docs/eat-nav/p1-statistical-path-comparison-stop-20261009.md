# P1 statistical Path comparison: protocol stop (2026-10-09)

## Decision

`protocol_invalid`. The prospectively frozen pilot configuration requested 80 candidate control steps, whereas the approved protocol requires 120: primary steps 0–39 and diagnostic steps 40–119. This was detected by checking the approved instruction against the frozen configuration during the limited pilot. The first 40-step primary records exist, but the 40 missing diagnostic steps make these runs ineligible for the complete P1 pilot. No Path Outcome comparison, confidence interval, ranking, or replication claim was produced.

The detached continuation driver was paused while its current owned child finished. That child sealed successfully; the paused driver was then stopped before it could start the next process. There was no retry and no further Path publication. The external Zenoh router was not started, stopped, or changed.

## Valid pre-Path work

- The historical 28-run golden reference, 112-env batch v1/v2, and minimal pre-Path physics evidence were SHA audited read-only. Their old gate remains failed.
- A new seven-slot candidate-free scene generated one canonical mesh per StateGroup, cloned it to all seven slots, and isolated NumPy plus PyTorch CPU/CUDA terrain RNG from policy/observation RNG. Eight of eight preflight processes passed controlled root, joint, flipper, ActionManager, actuator-target, latency, and observation-history checks. Canonical terrain mesh and composed collision geometry hashes matched across slots and the two batch seeds for every StateGroup.
- The 56-trial assignment manifest was frozen before Path exposure: four StateGroups, two independent batch seeds, four alternative Paths and three direct references per batch. Alternatives moved to different slots between batches. The RL Actor used `act_inference()` as confirmed by the user. Therefore no Gaussian action draw was made and policy-noise seed pairing was not claimed.
- The production follower binary and exact-stamp ROS bridge were used in five subsequent physical runs. The bridge's eighth follower received a monitor copy of slot 0's Path; only seven commands were injected into the seven Isaac environments.

## Partial pilot evidence

Five of eight prepared processes completed and were sealed: `first-differential` batches 0/1, `second-differential` batches 0/1, and `first-graded` batch 0. Each contained seven continuous 80-step traces, for 35 trajectories and 2,800 step records; none terminated. Three prepared roots had no Path publication. Each completed process had native exit 0, host/container trace SHA agreement, no resource violation, no owned follower/bridge residual, and unchanged external router PID/start tick/ZID. Peak GPU device use was 5,170 MiB, below the frozen 7,680 MiB cap. Combined host storage for the five completed roots was 46,137,600 bytes. The read-only integrity audit is `pilot-protocol-failure-audit.json` in the archived output root.

The first two StateGroups had matching canonical mesh and composed terrain collision hashes across both batches. The data establish that initialization and the first 80 control steps ran, but do not establish the approved 120-step diagnostic window or a statistical Path effect.

## Corrective preparation; no rerun

The original `eat_nav_p1_statistical_v1.json` and all sealed outputs remain unchanged. `eat_nav_p1_statistical_v2.json` specifies 120 candidate control steps with all other seeds, Paths, Outcome margins, resource limits, and initialization tolerances unchanged. The future pilot simulator and host refuse a configuration other than 40 primary and 120 total steps; the analysis CLI also refuses to present the v1 data as a complete pilot. The short-terminal Outcome extraction now censors a physically terminated trace without reading post-reset values. These changes were tested but no v2 Path run was launched.

## Artifacts and provenance

- Partial output and status: `/tmp/eatnav-p1-statistical-20261009/`
- Integrity audit SHA-256: `cee9ea91a642927f7c26267c8d71899120bf7d491b4b42cb003d5a45d117f8fb`
- Frozen v1 Path simulator source SHA-256: `5b32ac5901a5d79d79d4f2e36f7d690bd9b2e7331e0c29296d7ff1ddf3ce006e`
- Durable archive: `/home/kojima/eatnav-backups/2026-10-09-p1-statistical-protocol-invalid/artifacts.tar.gz`
- Archive SHA-256: `14a26cc5f74e8bee33111db3f4817ecf1713d54ad4d23914c0935c943be87a2c`
- Tests: 16 relevant tests passed; all changed scripts compiled. No full collection or Evaluator training occurred.

## Next decision

Review the stop as a protocol implementation failure. If a corrected limited pilot is authorized, use a new v2 output root, freeze its 120-step config and source SHA before Path exposure, repeat candidate-free verification and the fixed randomized assignment, and keep the 80-step records excluded from the complete pilot analysis. The pilot remains descriptive with only two independent batch seeds; tile effects and confidence intervals cannot be treated as confirmatory.
