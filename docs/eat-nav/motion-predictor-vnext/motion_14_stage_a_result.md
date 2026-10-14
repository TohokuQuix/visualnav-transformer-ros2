# 14-Path Stage A feasibility result

**Decision: Pass**, under the predeclared protocol gate. The fixed
`random_rough` level-2 seed `2026101602` produced 32/32 protocol-valid fresh
processes and 112/112 finite, 40-step candidate trial traces. Two independent
pre-Path references per Front/Rear posture and all candidate Gate P comparisons,
Path publications, labels and inner/outer seals passed. This is a one-seed
feasibility result, not terrain generalization.

The 14-Path geometry audit passed. Existing Direct and ±30° candidate byte
SHA matched the MI-A v3.1 records. The ±60°/±90° Paths have eight waypoints on
0.75 m reference arcs; their chord sums are approximately 0.7493/0.7484 m.
At ±90°, the first chord differs from the forward/reverse tangent by 6.429°,
within the frozen 7° bank limit. The angle is a Path geometry parameter, not
the robot's realized yaw.

| Initial posture | Candidate | Mean 2 s body X (m) | Mean follower linear command (m/s) | Mean angular command (rad/s) |
| --- | --- | ---: | ---: | ---: |
| Front-folded | Forward Direct | +0.419 | +0.290 | +0.051 |
| Front-folded | Forward +90° | +0.062 | +0.076 | +0.309 |
| Front-folded | Reverse Direct | −0.209 | −0.295 | +0.029 |
| Front-folded | Reverse +90° | −0.053 | −0.071 | +0.317 |
| Rear-folded | Forward Direct | +0.317 | +0.296 | +0.008 |
| Rear-folded | Forward +90° | +0.077 | +0.054 | +0.312 |
| Rear-folded | Reverse Direct | −0.457 | −0.288 | +0.064 |
| Rear-folded | Reverse +90° | −0.103 | −0.067 | +0.307 |

Large-angle forward/reverse command magnitudes decline while angular command
increases. All four env tiles per Path are retained, including low-progress
results; no follower, Actor, physics or numeric gate was changed. Actual yaw
and contact/body response vary by tile and posture, so command magnitude alone
does not describe realized motion. Full per-env XY/yaw, command, recovery,
contact and torque measurements remain in the read-only analysis JSON and raw
trace files.

Observed median/max process wall time was 20.94/32.45 s, peak GPU device-total
usage 5,372 MiB against the 6,656 MiB hard cap, minimum host available RAM
79.76 GiB, and Stage A evidence size 38.48 MB. Stage B's 1,152-process plan
implies roughly 6.7 h of process execution at the Stage A median, before
dataset auditing and training; its own smoke will refine this estimate.

Evidence root:
`/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/motion-14-stage-a-20261014-01/`.
Analysis JSON SHA-256:
`0da61a6d82b94b681205aef4213ffd6fc026060cf61ebba8b62779e8f25a7ed6`.
