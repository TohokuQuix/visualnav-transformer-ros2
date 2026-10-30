# R6: 2-second event and future-pose predictor

Authority: user explicitly adopted the complete handoff prompt and requested execution through a new-branch push, including ordinary implementation decisions without repeated approval. This design records builder decisions; it does not replace any requirement or introduce an approval stop.

## Scope and frozen references

Preserve the original dirty working copy, all sealed R1–R5 evidence, old model/checkpoint contracts and the handoff bytes. Work in /tmp/eatnav-r6-20261030, cloned from verified commit4dc17d8ec5c3ddeb5fa206460081aa1449d67c8a, new branch exp/eatnav-r6-2s-pose-thresholds-20261030. Large outputs are stored under /home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r6-2s-pose-20261030. Never publish robot commands; read the bag offline.

## Approach

Reuse actual old initial inputs/traces for true2-second targets, masked per candidate; add missing level0–3 and contact/approach/narrow/wall/one-sided scenes with a new vector collector. The alternative of renaming10-second outputs or applying posture rules to old progress is rejected: it cannot meet the predictive task. Model training stays supervised; the frozen low-level RL Actor is unchanged.

## Contract

Input: DEM(4,2,20,20), physical extent2m and resolution0.1m unchanged, 4 causal historicalframes at nominal250ms; preserve actual timestamps and diagnose real3Hz mismatch. State28 is old22actor+9body with actor command columns6–8 removed. Paths(14,8,2), directions(14,1), candidatevalidmask14. Inputs contain no future truth or map coordinates.
Output: event_logits(B,14,4) for events within2seconds only; trajectory(B,14,20,9) at t=.1,.2,...,2.0seconds. Each point is relative xyz3 plus sin/cos for relative yaw, gravity roll and gravity pitch. Relative positions are rotated by inverse initial yaw and origin-subtracted; world-up preserved. Absolute yaw is removed. Roll/pitch reconstruct in gravityframe; sin/cos normalization and circular errors handle wrapping. Checkpoint schema identifies2s horizon and exact time grid. No old10-secondprogress output is used by new policies: 2s expected route/goal progress derives from predicted positions.
Trajectory masks are per candidate/time; masks stop after physical termination/boundary. Event positives observed before censor may remain positive; incomplete negatives and incomplete40intervalstallwindows remain unknown. Training normalization fits train only. Weighted masked event BCE + masked position Huber + circular orientation loss; output sin/cos unit-norm regularizer. GPU minibatches, one trainingprocess.

## Collection, groups and evaluation

Seven named existingfields, levels0–4. Keep candidate bank shape, lowlevelactor and physical timestep20Hz. New headless collector benchmarks num_envs1,2,5,10 before production and records performance/independentreset checks. Failure/censor/unknown are distinct. Every new trace saves root quaternion, joint signals, contact forces, commands, velocities, terrain geometry, per-envtimestamps and terminalpre-resetstate. Repeat branch candidates from verified same anchor per state group; do not use othercandidate outcomes aslabels.
Source seed/initialgroup split, including all overlappingwindows and candidatebranches. Old reservedtest meanings remain unopened. New collectiontrain/validation and independentevaluationtest registries are fixed before collecting; exclude old publicregistries and derivedseeds. Nominal fixedgoal2m and60s all closedloopconditions, reachradius.25m; record rawrouteprogress, physicaldistance, netgoalapproach and reached. Naturaltermination stops physical run; remaininghorizon is unobserved rather than successful or commandedSTOP. Ranking explicitly includes failures and observationtime.

Validation closedloop: at least30 physicalrolloutseeds perfieldlevel percondition as the startingplan. Conditions: oldmodel+oldplanner, new2sevent, >=10 statisticallyderived pose/DEM thresholdprofiles, combinedevent+pose. Evaluate same seeds/lowlevelcontroller. Select top3 on validation lexicographically by fallcount, meancommandedSTOP, then equalweight validation-normalized progress/stall metrics. Freeze profilevalues/calibration/selectionscale before unusedtest. Newtest re-evaluates top3 plusreference,event,hybrid with at least30 percell. Report macro andcounts; insufficienttrials neverbecome safetyclaims.

Threshold profiles derive pitch/roll/height conditional failurequantiles/validationrisk with counts and confidenceintervals. Stairgeometry and solidwall geometry remain distinct; no newcriterion silently bans allstairs. Groundtruthpose is forlabels andstatistics, thresholds at inference apply predictedpose+causalDEM. Contactleadtime uses actual saved contactdefinition; missingcontacttruth yields unavailable rather than inventedpositiveleadtime.

## Bag boundary and unresolved calibration

ROS Python3.12 extracts MCAP into an intermediate format; torch training Python3.10/Isaac3.11 remains separate. Use receipt/logclock to constrain availability, retain sourceheaderclock, and diagnose clockoffsets without letting future receipt records into inputs. Same mapgoal initially2mforward for all3frozenprofiles. SourceIMU covariance orientationinvalid and flipperzerocalibration unresolved are reported explicitly; user calibration question pending. No mean-imputation disguised as calibration. Bag results are openloop candidate feasibility, not physical avoidance success. Geometry coverage and obstacle clearance include body/flippers+stopping margin; unknown map is not flatfreeground.

## Completion

Real artifacts, model support/finite/load/schema/speed checks, old regression checks, numerical pose/mask/causality tests, independent finalreview, and pushverifiedremoteSHA. Finalreport distinguishes planned/executed trialcounts, failedjobs and unsupportedheads. If a long job takes hours, persist progress andresume at an immutable job boundary; do not declare completion early.
