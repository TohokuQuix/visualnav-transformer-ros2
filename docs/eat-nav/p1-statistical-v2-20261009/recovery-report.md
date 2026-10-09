# P1 statistical v2: single-batch recovery

**Decision:** The frozen 56-trial descriptive pilot is complete and protocol-valid. This is a limited feasibility comparison, not evidence for a population Path ranking or authorization for further collection or Evaluator training.

## Recovery and provenance

- The only missing process was `second-graded / batch 1`. Read-only comparison found no blocking difference in startup arguments or display environment against the successful batches. The earlier failure was in Kit/XOpenDisplay before Python `main`, Path publication, or physics.
- The first launch command in this recovery session stopped at the host router audit because the sandbox could not see the user-owned router PID. It did not start Docker or the bridge, publish a Path, or consume a candidate measurement. The same new root was then used for one host-namespace launch. It passed Kit startup and completed in 61.39 s with native exit 0, no resource violation, peak device use 5,217 MiB, and zero owned residual processes. The external router was not changed.
- The recovered `source.py`, assignment, source contract, statistical config, anchor fixtures, YAML, and preflight input are byte-identical to the failed attempt. Its freeze SHA is the original frozen SHA `e50c09bce3ec1a050ed887bfbac4b5800d521cacdb744a67479a6c9c6d41f3fb`. The assignment SHA is `709fa5ccc98c152e75dc24cbd659dc8ea249c740efeb14bdabe31b5cc5affc69`. No Path geometry, initial condition, observation noise, follower, Outcome, or resource limit changed.
- The seven prior sealed batches and failed attempt remain in `/tmp/eatnav-p1-statistical-v2-20261009/`. All eight original freeze SHA values still match the frozen ledger. Analysis used read-only links to the seven successful roots and the separately sealed recovery root. The recovered host seal SHA is `b48703db6987e930e0958c94d56ef0f6bf134c212804b8b698bed371d1278726`.

## Descriptive pilot result

- All eight batches passed seal, input/source/config/Path provenance, initialization, resource/native exit, trace SHA, finite value, shape, timestamp, and step-continuity checks. There are 56/56 valid primary 2 s trials and 56/56 complete 120-step traces; none terminated. The 6 s segment is diagnostic only.
- Each StateGroup has **two independent batch seeds**. Three direct-reference slots within a batch do not constitute three independent replications. `act_inference()` was used; policy observation noise and simulator/contact variation are not paired between candidates.
- For the 16 candidate-minus-direct comparisons, 2 s discounted-reward point estimates are positive in 4 and negative in 12. Global-goal-progress point estimates are negative in all 16. All 16 reward point estimates and 9/16 progress point estimates are below their frozen practical margins; these are descriptive classifications, not significance results. The two-batch bootstrap intervals are underpowered and should not be treated as confirmatory confidence intervals.
- Direct-reference standard deviation across the six slots per StateGroup is 0.060–0.218 for 2 s discounted reward and 0.030–0.078 m for 2 s global-goal progress. Slot effects are not separately estimable from this sparse design. Goal-related reward is unavailable in these traces (`NA`), and no claim is made about it.
- Existing analysis estimates of additional batch counts vary widely because each comparison has only two independent contrasts. They are planning diagnostics, not a validated replication requirement. No definitive Path ranking or population superiority claim follows from this pilot.

## Artifacts and scope

- Recovery root: `/tmp/eatnav-p1-statistical-v2-recovery-20261009/pilot-second-graded-b1-v1/`
- Read-only integrated view: `/tmp/eatnav-p1-statistical-v2-recovered-view-20261009/`
- Analysis: `/tmp/eatnav-p1-statistical-v2-recovery-20261009/analysis/`; analysis seal SHA `91d2c6a1078160280f9dfb1836afbb3da51d440185ac46f2debc29f108203736`.
- Durable recovery archive: `/home/kojima/eatnav-backups/2026-10-09-p1-statistical-v2-recovery/artifacts.tar.gz`; SHA `1748e90b41c5d8c8755230ae1bdba2fcde6b4294531b4d37e778bd2276e2d984`.
- The original seven successful batches were not rerun. No pilot expansion, full collection, or Evaluator training was performed.
