# Lcorridor closed latest narrow-old7 scratch123 c0.0 NN-endpoint inflation: all14 fixed2s

Numeric goals 8/10; observed no-wall-crossing goals 8/10. Outcomes {'goal_reached': 8, 'tip_over': 2}. Previous latestNN/c0/fixed2s closedL no-bonus achieved0/10 goals,8tipovers2timeouts; only NN2s endpoint inflation penalty differs. One global seed/batched N10; exploratory comparison.

Latest narrow-old7 scratch123 NN+ownnorm; sameActor/seed123/N10/reference5physics/originalLreferencewidth1.12m, closed6walls H.5m T.15m/90s/tilt85/goalradius.2/c0.0 score and B60/actual-floorbounds[-8,8]^2 (same floorboundary as open baseline). All14 forward/reverse paths including±60/±90 fixed2s; no follower-completion earlyreplan; no stuck recovery. Score=rawXYgoalprogress -.5m*C for all14, margin.2m; c0RPpenalty0; eligibility masks retained. NN stillpredicts2s; offgrid latest4Hz DEM usesactualcausalcontext.

All10 initialphysicalstates andDEM exact preceding no-bonus closedL reference. Closedloop prefix not required identical. Actualdecisions 223; earlycompletedlongholds 0; offgridDEMcontexts 0. 2swindowcategories {'valid': 213, 'terminal_partial': 9, 'STOP': 1}; predictioncoveredtime 436.65000000000003s +unpredictedholdcontinuation 0.0s =recorded 436.65s. No zeroteachers forpartial/STOP/continuation.

|trial|result|seconds|finalXY m|visual judgment|
|---|---|---:|---|---|
|original-t01|goal_reached|79.25|(2.254,0.744)|goal_without_observed_wall_crossing|
|original-t02|goal_reached|37.55|(2.511,0.558)|goal_without_observed_wall_crossing|
|original-t03|goal_reached|19.50|(2.179,0.425)|goal_without_observed_wall_crossing|
|original-t04|goal_reached|58.70|(2.136,0.574)|goal_without_observed_wall_crossing|
|original-t05|goal_reached|38.90|(2.156,0.464)|goal_without_observed_wall_crossing|
|original-t06|goal_reached|43.55|(2.479,0.460)|goal_without_observed_wall_crossing|
|original-t07|goal_reached|27.30|(2.292,0.364)|goal_without_observed_wall_crossing|
|original-t08|tip_over|48.05|(1.282,0.436)|tip_over_inside_corridor|
|original-t09|tip_over|56.45|(1.583,0.426)|tip_over_inside_corridor|
|original-t10|goal_reached|27.40|(2.137,0.493)|goal_without_observed_wall_crossing|

One seed10batchedenvslots, exploratorycomparison. Same closedT015 mesh with6walls .15m thick. SameNN/c/hold as previousclosed; actual left/inner_vertical initialwalltop rayhits required; allpose detection and realrobot thinwall performance unverified. VisualQA doesnot certifycontactfree or continuousbodyclearance; targetwallforces notisolated. Only inflationpenalty -.5C for all14 differs versus previousclosed. No continuity bonus or extra holds.

NN SHA: a9fddc4e05a846762277de2ea06ad3efba0016ae64e5c237ad64f54ca2ca22c6
Mesh SHA: 2b379a10cb232729ca2731fc59c25301abb6a1ceae58b1e6c258e2c538797229

Reproduce: /home/ubuntu/venvs/isaaclab_py310/bin/python -u /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/run.py (exclusiveGPUlease, frozeninputs/SHA; sealedrunreuse, partialoutputs neveroverwritten). CPUaudit: audit_run.py; fulltrace/results/holds/videos retained here.

C is max knownwall linearEDT inflation cost over NN2s endpoint fullbody footprint; margin.2m, lambda.5m. Unknown/outside alone do not cause cost. O and fixed8cm bonuses removed. Baselinefixed8cm had3goals, opening20had0goals.
Videos:
- [original-t01](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t01/original-t01-6044dc652e.mp4)
- [original-t02](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t02/original-t02-64a80dfeb2.mp4)
- [original-t03](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t03/original-t03-a99fe84136.mp4)
- [original-t04](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t04/original-t04-9b63f49901.mp4)
- [original-t05](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t05/original-t05-7e00393ded.mp4)
- [original-t06](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t06/original-t06-4700120d1a.mp4)
- [original-t07](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t07/original-t07-a3e5b3d664.mp4)
- [original-t08](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t08/original-t08-dc26d3ed24.mp4)
- [original-t09](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t09/original-t09-fc77d2bd28.mp4)
- [original-t10](/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030/restart-backup/1793452520/worktree/analysis/2026-11-18-Lcorridor-closed-narrow123-c0-inflation-fixed2s/condition/video/anglehold-original-t10/original-t10-870aaef217.mp4)
