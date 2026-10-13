# GUI demo Stage A: existing four-env boxes run

Read-only source: `/home/kojima/lib/onix_isaac_lab/.eatnav-ros2-gui-demo-v1-20261013-171527/`.
Its 170-step trace, 5-file SHA ledger, and host seal verify. The GUI/RViz
screenshots are `final-gui.png` and `final-rviz.png` in the same root.

| Measurement (env 0) | Result |
| --- | ---: |
| Start map pose (x,y,yaw) | (-3.9983, -4.0409, 0.4951) |
| End map pose (x,y,yaw) | (-3.4022, -3.9195, -0.4952) |
| Common map goal (x,y) | (-3.3199, -3.8136) |
| Traveled XY distance | 0.7915 m |
| Initial / minimum / final goal distance | 0.7155 / 0.1285 / 0.1342 m |
| Net goal progress | 0.5814 m |
| Candidate selections | 9: Direct 2, Right 7 |
| Evaluator latency, median / max | 0.839 / 1.466 ms |
| Nonzero follower command steps | 39; steps 40–78 |
| Maximum commanded forward / absolute yaw speed | 0.294 m/s / 0.097 rad/s |
| Mean absolute actual forward / yaw speed | 0.086 m/s / 0.118 rad/s |
| Mean absolute command-to-motion residual on active command steps | 0.035 m/s / 0.076 rad/s |
| First local Path Goal | step 79 |
| Final 50 command steps | all zero |
| Actor action / physics | finite action on 170/170 steps; changing root pose |
| Episode termination | 0 |

Actual speed is the finite difference of recorded pre/post root XY/yaw over
50 ms; forward speed is projected on pre-step heading. It is not the commanded
speed. Goal progress uses the recorded map goal and map root pose. The observed
displacement and decrease in goal distance establish physical movement after
ROS goal reception, candidate scoring, Path selection, follower command, and
Actor inference. The local Path reached Goal at step 79; global-goal proximity
crossed 0.15 m later. Motion continues under zero command, so the trace
verifies command zeroing, not a physical stop. This run did not exercise a
ROS cancel message, so cancel is a
separate runtime check in the new series.

Four envs were a demo runner constant: `make_collection_cfg(..., 4)`,
`CommandOverride(..., 4)`, `(4,3)` command buffers, `(4,22)/(4,267)`
observation checks, and `(4,6)` action check. The policy weights do not
encode a four-env batch requirement. Env 0 alone receives the ROS goal;
the other command rows are zero. The one-env runtime still requires an
independent reset and checkpoint compatibility check.
