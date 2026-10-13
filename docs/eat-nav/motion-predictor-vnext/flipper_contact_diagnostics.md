# MI-FI v1 contact and body-state diagnostics

The existing 60-body `contact_forces` sensor captured net forces and three-sample history. Its body-name map identifies front and rear flipper subrollers. Mean over 40 control steps, four envs, of summed per-body force magnitudes was:

| Condition | Candidate | Front subrollers N | Rear subrollers N |
|---|---|---:|---:|
| Front-folded | Forward Direct | 369.6 | 64.7 |
| Front-folded | Reverse Direct | 314.9 | 131.2 |
| Front-folded | Reverse +30° | 300.9 | 140.9 |
| Front-folded | Reverse −30° | 305.7 | 153.0 |
| Rear-folded | Forward Direct | 62.6 | 394.8 |
| Rear-folded | Reverse Direct | 65.8 | 369.7 |
| Rear-folded | Reverse +30° | 73.9 | 376.0 |
| Rear-folded | Reverse −30° | 62.6 | 365.7 |

The dominant contact region switches with the initialization condition and remains different during execution. The sensor records net force per body, not contact partner or penetration; these values do not prove that a particular flipper caught a terrain edge. Mean absolute flipper torque over the same traces was 29.6 N·m (Front-folded Reverse Direct) versus 33.8 N·m (Rear-folded Reverse Direct). The torque difference and contact redistribution are consistent with changed load paths but do not identify a unique causal mechanism. Root pitch and height also differ after settling, so motion and contact effects cannot be attributed to angle alone.

The raw per-step trace stores all-body contact vectors, all-joint velocity/torque, each flipper's position/velocity/torque and soft-limit margin, root state, command and Actor observation/action. `prepath.npz` stores contact history, four DEM frames, Actor pre/post-noise input and Critic input. No contact sensor was added, and the physics model was unchanged. All candidate traces had zero early termination; all flipper soft-limit margins in the derived diagnostics were finite.
