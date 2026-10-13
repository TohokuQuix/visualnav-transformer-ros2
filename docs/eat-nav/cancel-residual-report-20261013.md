# Cancel/Goal zero-command residual motion audit (2026-10-13)

## Decision

The measured movement is **not a delayed Cancel command**. In the sealed
boxes-level-0 run, the follower and Actor-delivered command were already
zero for about 6.1 s before Cancel. In new paired runs A (no Cancel) and B
(Cancel), all 280 stepwise Actor actions, root/joint/contact states, rewards,
and DEM timestamps were byte-equal; only the Cancel event and follower/Path
diagnostic metadata differed. In C, a fresh process with **no Goal ever sent**
also moved under zero commanded velocity. Nonzero RL Actor outputs and
simulated joint motion/torque persist during these periods. This strongly
supports zero-command policy/actuator activity as a contributor. Contact,
gravity, slip, and solver dynamics may also contribute; their individual
causal shares cannot be isolated without a separate intervention protocol.

Cancel in this demo means clearing the Path and sending zero velocity
command. **It is not a physical emergency stop.** Neither the Actor nor the
physics engine is disabled by Cancel.

## Preserved evidence and timing

The original terrain-v2 root
`/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-20261013-173938/`
was read only. Its 7-file SHA ledger and host seal verify. `trace.jsonl`
contains 260 consecutive 50 ms steps and a Cancel event at step 190,
logical time 9.50 s. The [read-only derived analysis](cancel-residual-v2-analysis-20261013/analysis.json)
and [plots](cancel-residual-v2-analysis-20261013/) have a separate SHA/seal.

| Original v2 measurement after step 190 | Value / meaning |
| --- | ---: |
| Follower command nonzero steps | 0/70 |
| Actor-delivered command nonzero steps | 0/70 |
| Integrated root XY path length | 0.2624 m |
| Final net displacement from Cancel pose | 0.2146 m |
| Maximum displacement from Cancel pose | 0.2146 m |
| Root yaw change | −0.4405 rad |
| Path length, 0–0.5 / 0.5–1 / 1–2 / ≥2 s | 0.0350 / 0.0412 / 0.0794 / 0.1068 m |
| Final 0.5 s mean pose-difference XY speed | 0.0769 m/s |
| Final pose-difference XY speed | 0.0729 m/s |
| Episode termination/reset | 0 |

The original v2 trace has no measured root velocity tensor, joint velocity,
applied torque, or contact force. The last two speeds above are pose finite
differences, not Isaac velocity tensors. The cumulative distance is larger
than the net displacement, but the nonzero net displacement and persistent
late speed rule out a merely stationary oscillation.

### New A/B/C diagnostic protocol

Only trace capture was added under protocol
`eatnav_ros2_isaac_gui_demo_cancel_residual_v3`. The scene, checkpoint,
terrain tile, DEM, 20 Hz control, Actor inference, follower, Path geometry,
noise/latency, and physics are unchanged. Each fresh 1-env boxes-level-0
process used seed `2026101101`, 280 steps, a 0.75 m goal for A/B, and the
same tile geometry SHA
`e694887dde930372ddf9b2eee714a9167e2fd8debc6fb5282a228cbaa233b1ef`.
The simulator source SHA was
`8c13430c8e1d610319bf0ae2c96c5640b9f948651019b5a6ce11739179191387`;
the exact runtime diagnostic helper was
`684cebd4709739feec767979a23fb1fb8565428a37b1bc58ab68be6867adf910`
and is preserved with its [provenance and separate seal](cancel-residual-runtime-source-20261013/provenance.json).
The repository helper later gained offline-only analysis functions, so its
current SHA differs from the staged runtime copy.
Evaluator/RL checkpoint SHAs remained
`329341fbf56ccb6bbc2f05b3db6b143cc1a1e9553ba2796e3dec3c785f30e470`
and `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`.
Each raw root has inner simulator and outer trial-controller SHA ledgers/seals.

| Mode | Persistent raw root | Goal / Cancel | Result |
| --- | --- | --- | --- |
| A | `.eatnav-ros2-gui-demo-cancel-residual-v3-20261013-181140` | Goal step 25; no Cancel | 280/280, protocol-valid |
| B | `.eatnav-ros2-gui-demo-cancel-residual-v3-20261013-181325` | Goal step 25; Cancel step 191 (9.55 s) | 280/280, protocol-valid |
| C | `.eatnav-ros2-gui-demo-cancel-residual-v3-20261013-181456` | No Goal; no Cancel | 280/280, protocol-valid |

All roots are under `/home/kojima/lib/onix_isaac_lab/`. The
[machine-readable A/B/C comparison](cancel-residual-abc-20261013/comparison.json)
verifies source seals and 50 ms step alignment. A/B pre-Cancel physical
state, Actor action, command, and DEM were identical; all these signals
remained **exactly identical** for steps 191–279. The first follower Goal
was step 69 (3.45 s). The follower command was zero from that step onward,
yet the B root accumulated 0.6485 m from step 69 through step 279. A and C
were identical through step 29; the first differing command, Actor action,
and physical state were all at step 30, when A's first selected Path began.

| Measurement | A and B, steps 191–279 | C, steps 191–279 |
| --- | ---: | ---: |
| Follower/Actor command nonzero steps | 0 / 0 | 0 / 0 |
| Integrated XY distance | 0.3350 m | 0.2079 m |
| Net XY displacement | 0.2128 m | 0.1736 m |
| Mean Isaac post-step root XY speed | 0.0779 m/s | 0.0529 m/s |
| Mean absolute sprocket joint velocity | 1.702 rad/s | 1.585 rad/s |
| Mean absolute simulated sprocket `applied_torque` | 90.4 N·m | 76.6 N·m |
| Mean sum of per-body net contact force norms | 514.9 N | 515.2 N |
| Termination/reset | 0 | 0 |

C's full 14 s net displacement was 0.5226 m, with follower command zero
throughout. Its Actor-delivered command was nonzero only at step 0, from the
initial latency buffer; it was zero thereafter. C therefore demonstrates
long-lived motion without any prior Goal/Path command. In B, the post-Cancel
follower output, injected command, reconciled pre-noise and post-noise command
slice, and post-latency Actor-delivered command were all zero on 89/89 steps.
The latency coefficient was approximately 0.7, yet it did not retain a
nonzero command at Cancel. Mean absolute Actor actions over those 89 steps
were approximately `[0.341, 0.459, 3.907, 3.234, 0.111, 0.114]`. The resolved
Onix action config maps the last two dimensions to right/left track velocity
actions; joint velocity and torque were nonzero concurrently. The logged
`ArticulationData.applied_torque` is simulated applied joint torque, not a
real effort measurement or a direct force attributable solely to the Actor.
`ContactSensorData.net_forces_w` is net simulated contact force by body;
its summed norm is descriptive, not evidence that friction caused the motion.

B's 89 post-Cancel steps lasted 4.45 s. It traveled 0.0366 m in the first
0.5 s, 0.0386 m in 0.5–1 s, 0.0805 m in 1–2 s, and 0.1793 m afterward.
The final 0.5 s mean pose-difference speed was 0.0765 m/s, and the final
Isaac `root_lin_vel_w` XY magnitude was 0.0306 m/s. The last instantaneous
pose finite difference is 0.0546 m/s; these are different measurements at
different substep/aggregation boundaries and must not be conflated.

### Hypothesis review

| Hypothesis | Evidence-based conclusion |
| --- | --- |
| H1: command propagation delay | Rejected as the explanation for B. All command-chain stages were zero before and after Cancel; A/B physics was identical. A startup-only step-0 latency value in C is separately recorded. |
| H2: RL Actor response to zero command | Strongly supported. Action remains nonzero; track action dimensions, simulated joint velocity/torque, and root motion persist; C moves without a Goal. An action intervention would be needed to quantify its exact causal share. |
| H3: inertia, slip, gravity, contact | Possible contributing mechanisms, not separable from active action with these observational traces. Contact magnitude alone cannot identify force direction or attribution. |
| H4: oscillation/yaw rather than translation | Partly present (integrated distance exceeds net), but B also displaced 0.2128 m net and yaw changed −0.5447 rad. |
| H5: control/log mismatch | No evidence in the measured chain: step/stamp continuity, exact A/B paired physical series, command reconciliation, 280/280 trace, and both seals verify. |

The main interpretation is **continuing closed-loop RL actuation under a
zero velocity target**, interacting with the simulated contact dynamics.
This is high confidence as an observed pattern, but a precise percentage
attributable to each physical mechanism remains unresolved. A future
comparison that deliberately zeroes physical action after Cancel would be
a *different control protocol* and was not authorized or run here.

## Student GUI reproduction

The documented one-env launcher was run again in GUI mode for 220 steps in
fresh root
`/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-20261013-181954/`.
The existing ROS sender published a 0.75 m map goal, received at step 25;
six Evaluator selections occurred, the robot moved, and ROS Cancel was
received at step 201. All 19 remaining commands were zero, while the root
traveled 0.0793 m. Isaac Sim and RViz2 were both displayed; the RViz image
shows the candidate Paths, selected Path, score labels, odometry, and DEM.
[Isaac screenshot](/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-20261013-181954/repro-isaac.png)
and [RViz screenshot](/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-terrain-v2-20261013-181954/repro-rviz.png)
are auxiliary files outside the inner 7-file simulator seal. The simulator
completed natively, inner SHA/seal passed, and no episode reset occurred.
The RViz window was verified, but the **manual 2D Goal Pose click was not
exercised**; the already implemented goal sender published to the same topic.
The RViz script is invoked with `bash` because its current file mode lacks
the executable bit; the student setup document records this tested command.

## Resource and remaining work

A/B/C device-total GPU peaks were 5.28/5.26/5.27 GiB, minimum host available
RAM was 81.34/81.24/81.36 GiB, and persistent roots were 4.34/4.26/4.09 MB.
The GUI reproduction peaked at 7.42 GiB device-total GPU and used 1.25 MB
in its root. These are not per-PID GPU or process RSS figures. All four new
runs completed without an episode termination; no router, checkpoint,
follower, Actor, terrain, or physics setting was changed.

Future work may evaluate a separately specified stop/hold controller,
including physical-action intervention and contact/slip diagnostics, on
new versioned data. That change needs its own validation, especially before
any hardware use. It was not implemented in this task.
