# Path-corridor architecture comparison: frozen design

The experiment uses only the 28 StateGroups and 252 saved protocol-valid live traces in `/tmp/eatnav-evaluator-independent-expansion-20261006-v7`. It leaves every source artifact unchanged. The independent unit is a StateGroup; its three direct repeats and six other candidates stay together in every one of the 40 existing folds. The comparison has four architectures, A1–A4, three seeds (123, 456, 789), and the six existing regression heads plus reward/goal pair ranking loss. Safety, Critic residual, terminal Critic, and sprocket inputs are excluded.

## Coordinates and shared input

The saved 20×20 DEM has 0.1 m cells with row increasing in robot-left y and column increasing in robot-forward x. Each of the four saved frames was captured in the robot pose of that frame. The candidate Path is in the later anchor's robot-local frame. For each frame, map an anchor-local query through the anchor pose into world XY, then into the frame's source grid. This aligns both the full DEM and the candidate corridor at the anchor pose. The same aligned full DEM is used by A1 and A4, and the same aligned corridor by A2–A4. A1 is therefore a flatten-DEM **architecture** reproduction; its shared pose alignment differs from the historical B4 preprocessing and must be reported.

The recorded source DEM height is world Z. Subtract the scalar median height of valid cells in the anchor source grid's 0.3–0.8 m annulus, for every frame and architecture. No plane fit or roll/pitch leveling is applied. The existing transformed DEM is checked against its source frame before any comparison. Bilinear sampling rejects a point if any contributing nonzero-weight cell is invalid or out of bounds. Invalid height is stored as zero with mask zero; models always receive the mask. No class or threshold uses Outcome.

Arc-length resample each origin-prepended eight-point Path at 16 stations. Sample seven lateral offsets from −0.4 to +0.4 m in the anchor frame. Corridor tensor shape is `[4,2,16,7]`. The full aligned DEM shape is `[4,2,20,20]`.

## Outcome-blind support rule

Apply `corridor-config.yaml` before reading labels. If any candidate has less than 50% valid corridor samples, classify its group `insufficient_or_unknown`. Otherwise a group is `transition_corridor` if its first nonflat station is at least 0.15 m and at most 0.55 m from the start, with the earlier stations flat. Otherwise classify it `differential_corridor` when any candidate pair has corridor RMS or maximum-step difference above 0.03 m. Otherwise classify it `flat_corridor` when every candidate's height standard deviation and maximum longitudinal step are at most 0.03 m. All other groups are `common_rough_corridor`. A stratum with fewer than three StateGroups is `insufficient_support` for performance claims. Classification never removes a training example.

## Models and training

Common inputs are candidate Path XY, path geometry, the historical 64-point path profile and coverage, relative goal, portable body Core, and availability/stale/arm masks. This preserves the historical B4 feature contract as the common branch and excludes sprockets. A1 flattens aligned full DEM height/mask through a two-layer projection. A2 uses fixed per-station corridor statistics through a two-layer 1D CNN. A3 uses the fixed 8→16→32→32 2D corridor CNN. A4 adds a separate full DEM CNN to A3. The same 384-wide fusion MLP and six-output head follow each architecture. GroupNorm and SiLU are used in CNN branches; no BatchNorm.

All models use train-StateGroup-only normalization: one shared scalar mean/std for valid DEM heights, train-only feature scalers for common vectors and A2 profile, and train-only group-balanced target scalers. Masks remain binary. Adam LR 1e−4, no weight decay, 200 epochs maximum, train-loss early stopping with patience 30/min_delta 0, the historical six-head Smooth L1 weights and Bradley–Terry pair weights are fixed. Checkpoints are selected on train loss only. Parameter counts are frozen before model Outcome evaluation and reported; the target is within approximately ±20% but any shortfall remains visible.

## Gates and reporting

C0 audits every source SHA, episode/Path/goal identity, trace/label mapping, source DEM geometry, pose correspondence, and split isolation. C1 freezes the geometry-only corridor tensors, support metrics and classes before training reads labels. A failed identity, frame, mask, leakage, or nonfinite gate stops the run. The 40-fold × 4-model × 3-seed evaluation then reports all six regression heads, ranking and multiobjective metrics, and StateGroup-paired bootstrap intervals. E-goal is shared-anchor diagnostic; E-terrain is diagnostic only. The final decision does not authorize rollout, online selection, new data collection, or robot control.
