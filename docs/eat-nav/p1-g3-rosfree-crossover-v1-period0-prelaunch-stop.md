# ROS-free G3 crossover v1: period 0 prelaunch stop

## Decision

The one authorized period-0 attempt stopped before owner-wrapper startup, Isaac Lab startup, settling, core construction, Path activation, or physics. Protocol-valid processes: 0/1. Candidate trials and Outcomes: 0/4. No automatic retry was made. Periods 1 and 2 were not started.

## Frozen attempt

- Protocol: `eatnav_p1_g3_rosfree_crossover_v1`.
- StateGroup/seed/period: `first-differential / 2026101101 / 0`.
- Roles by env: `Direct, Direct, Left +20°, Right −20°`.
- Preparation manifest SHA-256: `685a5cac0f06d29125749a2feaa5039340e7557a79b25834ef29aac3a447c635`.
- Authorized attempt manifest SHA-256: `c659d12da30abec943e5e13429bfe0e6241b9a6ef8283300e3c1d2a77501ec3b`.
- Assignment SHA-256: `8c910f1204502cdf749c2d265887c3b29026bcdef4647674b8548d49a6277d29`.
- Simulator source SHA-256: `0132a4e4bcfc6fa651f27fba13420ec5286a43aae34e7f4be553fc08347c80f5`.
- Host source SHA-256: `e986962907db94600318dd8fb9d52f7e72a4e09e4ecfa8096a1481767393a5e9`.
- Failed attempt root: `/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-0/`.

The attempt manifest, assignment, input ledger and all 199 staged source/input files passed read-only SHA verification after the stop. `rosfree-prelaunch.json` exists. The batch has no `coord/`, `output/`, or `host-seal.json`; no owned simulator or wrapper process was observed afterwards. Thus no material/pre-Path, resource-series, cleanup, or Outcome gate was measured. This attempt must remain failed and must not be promoted.

## First failing signal

The frozen host runner was invoked from its staged copy inside the batch root. At host-runner line 345, `Path(__file__).with_name('eat_nav_p1_owner_wrapper.py')` selected a nonexistent sibling file and raised `FileNotFoundError` before the wrapper or Isaac launch. The identical repository host source has the wrapper as its sibling. The prelaunch failure is an invocation-path error, not a Path, follower, physics, Actor, or material measurement failure.

The next runtime would require separate authorization and a fresh persistent attempt root. It should invoke the repository host source with the frozen SHA, or explicitly stage and verify the owner wrapper adjacent to the staged host source. The failed attempt is retained unchanged. No retry is authorized by this report.
