# MI-FI v1 motion and initialization interaction

**Decision: the 16-process, one-seed diagnostic is protocol-valid and supports a measurable initialization effect.** This is descriptive evidence from one `pyramid_stairs` level-2 terrain seed (`2026101305`) and four env tiles. It does not establish a population effect, mirrored terrain symmetry, or Motion Predictor learnability. All 16 inner and outer attempt seals passed read-only verification; the 12 candidate processes yielded 48/48 finite, same-episode 40-step trials. Gate P passed within both initialization conditions. Gate M remained diagnostic only.

The only experimental assignment difference was the existing reset pose selection. Both groups used fold mode 0, the same terrain/material fingerprints and RNG states, Actor checkpoint, follower core, Path geometry, noise/latency and physical settings. At reset, Front-folded measured `[0, 0, π, π]` rad and Rear-folded `[π, π, 0, 0]` rad in RR/RL/FR/FL order. After the 20-step settling, mean angles were `[-0.029, 0.024, 3.158, 3.121]` and `[3.235, 3.232, -0.167, -0.204]` rad. The posture distinction survived settling; no post-settling overwrite occurred. Rear-folded also had mean root height 0.812 m and pitch −0.089 rad versus 0.792 m and +0.007 rad for Front-folded. These settled body/contact changes belong to the initialization intervention, so the experiment does not isolate joint angle alone.

| Condition | Candidate | Mean ΔX m | Mean ΔY m | Mean Δyaw rad |
|---|---|---:|---:|---:|
| Front-folded | Forward Direct | +0.5685 | −0.0301 | −0.0757 |
| Front-folded | Forward +30° | +0.5002 | +0.0012 | +0.0333 |
| Front-folded | Forward −30° | +0.5111 | −0.0685 | −0.2145 |
| Front-folded | Reverse Direct | −0.3850 | +0.0141 | −0.0304 |
| Front-folded | Reverse +30° | −0.4151 | +0.0165 | −0.0256 |
| Front-folded | Reverse −30° | −0.2783 | +0.0209 | −0.0812 |
| Rear-folded | Forward Direct | +0.5018 | −0.0265 | −0.0957 |
| Rear-folded | Forward +30° | +0.4259 | +0.0018 | +0.0389 |
| Rear-folded | Forward −30° | +0.4302 | −0.0470 | −0.2185 |
| Rear-folded | Reverse Direct | −0.5852 | +0.0384 | −0.0957 |
| Rear-folded | Reverse +30° | −0.5654 | −0.0090 | +0.0643 |
| Rear-folded | Reverse −30° | −0.5009 | +0.0807 | −0.2607 |

Rear-minus-Front Direct ΔX was −0.0668 m for Forward and −0.2002 m for Reverse. All four env tiles moved less forward with Rear-folded. Reverse gained backward distance in all four tiles, by 0.026–0.262 m; the effect size was tile-dependent. The stronger reverse performance therefore does not imply an exact forward/reverse symmetry swap.

Arc-minus-same-direction-Direct yaw for Reverse +30° changed from +0.0048 to +0.1600 rad, and Reverse −30° from −0.0508 to −0.1651 rad. The corresponding Rear-minus-Front interactions were +0.1552 and −0.1143 rad. Reverse +30° interaction yaw was positive on all four tiles; Reverse −30° was negative on four, but nearly zero on two. XY interactions were also nonzero: Reverse +30° `[+0.0498, −0.0498]` m and Reverse −30° `[−0.0224, +0.0355]` m. Forward Arc interactions were smaller in mean yaw (+0.0256 and +0.0159 rad). The response changed with both initialization and Path, beyond a uniform speed offset.

The resolved Actor policy input is 22D and includes four individual relative flipper positions in slice `[10:14]`. At Path activation, the pre-noise slice matched the four measured joint angles exactly in both conditions (maximum absolute difference 0). The policy also receives projected gravity and base angular velocity. Thus the **input representation can distinguish** these body states. A trained Motion Predictor's use of this information remains untested.

The per-env labels, interactions and signed Arc-minus-Direct values are in `mi_fi_v1_motion_results.json` in the evidence root. The findings support testing both initialization conditions in a balanced multi-seed MI-A design. That design should assign initialization at the StateGroup level and keep all candidate Paths within a StateGroup on the same settled state. No formal schedule or learning was started here.
