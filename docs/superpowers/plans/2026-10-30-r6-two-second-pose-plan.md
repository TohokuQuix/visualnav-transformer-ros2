# R6 execution plan

Spec: ../specs/2026-10-30-r6-two-second-pose-design.md. Explicit user authorization permits continuing through push without another planning approval.

1. Freeze handoff/environment/source identity; preserve originals, isolated new branch. Audit old dataset support and bag clocks/calibration.
2. RED→GREEN contract and target tests: true2s horizon; yawalignedrelative xyz and orientation period; terminal/reset/censor masks; group split exclusion; no future input.
3. Implement separate R6 target/dataset and model modules. Reuse real R4star inputs/traces without contact fabrication; sparse unsupportedheads masked. GPU mini-batch training, strict schema, trajectorytimes, trainnormalization, metrics andtiming.
4. In parallel, add new headless vector collector and independentreset/contactrecording. Benchmark env counts1/2/5/10 and select from evidence; freeze new manifest and run missingfieldlevel/approachscenes, resume jobledger.
5. Statisticalvalidation module: signedroll/pitch,geometry+success distributions/CIs, >=10 derived YAMLprofiles and fixedranking/scales. Freeze nominal2m/60sprotocol and cohortseedregistries before actual closedlooprollouts.
6. New closedloop evaluator: oldreference,event,pose10profiles,hybrid sameconditions. Complete validationtrials, freeze top3 withSHA, then unusedsimulationtest. Record allrequired metrics including exposure,censor,reached,latency andleadtime.
7. Offline ROS extraction with receiveclockcausality and exact intermediatecontract. Only afterfreeze, apply top3profiles to samebagobservations andcommongoal. Publish no ROSmessages. Save obstacle/coverageoverlay andperdecisionreasons.
8. Review defects with reproductiontests, run regression/model/integritychecks. Package code/checkpoints/config/smallresults, reference bigdatawithSHA and retrievalinstructions. Commit andpush newbranch; verifyremoteSHA; finalreport allrequirednumbers and limitations.

Ledger: docs/eat-nav/r6-2s-pose-20261030/work-log.md; large execution artifacts live under original workspace docs/eat-nav/r6-2s-pose-20261030 and are not force-added wholesale.
