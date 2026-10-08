# Local motion primitive bank offline review

Decision: `local_motion_primitive_terrain_support_fail`. No live process or physical rollout was started.

The common global goal is an offline relabelable target while the candidate Path and anchor remain fixed. The SHA-frozen E1 source uses `mission_logged_goal` only in trace diagnostics; the former Outcome progress target instead used the local Path endpoint, so it must not be reused as the new navigation teacher. The historical selected-DEM goal in the pre-Path anchor gate is held fixed and is separate from the relabelable global goal. Source audit and archived-trace fixture both pass.

Five analytic circular arcs at (-20, -10, 0, 10, 20) degrees and 0.75 m chord are valid at all eight anchors. Their analytic maximum curvature is 0.912/m, below the unchanged 1.236/m follower screen. All 40 anchor/Path combinations pass the static first-command screen. This is not evidence of realized tracking.

The primary known DEM terrain gate **fails**: pair counts by class/seed are {'transition': {'first': 0, 'second': 0}, 'differential': {'first': 7, 'second': 9}, 'graded': {'first': 0, 'second': 0}}. The 0–0.60 m interval is primary; 17 pair differences seen only by 0.75 m are diagnostic. DEM centerline and main-track coverage pass. The already measured R3 reserve inventory has no eligible physical-supply replacement for the missing transition/graded combinations. The geometry is retained as a hypothesis; no Outcome-driven adjustment or P1 execution follows from this result.

There are 40 physical definitions and 200 correlated goal-label identities. The archived 40-step fixture has one trace SHA across five relabels, unchanged locomotion reward and varying goal-distance targets. These 200 identities are not independent trajectories. The conditional P1 manifest has 0 processes because the terrain gate failed; the reviewed plan would require 28 processes if all gates passed. Forecasts are recorded without starting the pilot clock.

The old common-endpoint Bank A/G1–G3 and 0.75 m bank could not supply a shared five-Path set. The new separate-endpoint semantics remove that geometric contradiction, but do not by themselves create sufficient transition/graded terrain separation in the saved DEMs. The common goal remains the navigation target; locomotion and progress remain separate Evaluator heads. A short local detour could initially lose goal distance while improving locomotion, which can only be judged after a separately approved physical pilot. This handcrafted bank is a data-generation instrument, not the final NoMaD generator.

See JSON artifacts and ten figures for source SHA, formula, follower commands, DEM corridors, reserve inventory, label semantics, resource forecast and decision. P1 rollout requires a new review and explicit approval.
