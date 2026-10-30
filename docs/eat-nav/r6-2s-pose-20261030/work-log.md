# R6 work ledger — 2026-10-30

- User adopted full handoff and explicitly authorized implementation, experiments and new-branch push. Existing robot commands prohibited.
- Isolated clone from4dc17d8, branch exp/eatnav-r6-2s-pose-thresholds-20261030. Original dirty checkout untouched.
- GPU access is sandbox-limited: host sandbox nvidia-smi fails, escalated read confirms RTX4090Laptop16GB, driver580.178.04; Isaac container torch2.7cu128 CUDA available. Existing Python3.10 host torch2.7; ROS Python3.12 separate.
- Handoff55files verified; sensorbag210926 present,5655messages/26.033s. Bag GridMap headers~.32s ahead of receipt, motion~.52s. Causal extractor must explicitly preserve both clocks.
- OldR4star105groups1470candidatebranches,29248020Hzrows,1467complete2sbranches; level4 only. No perstepcontact/jointtruth exceptterminal. Old2slabels exist atindex1; oldtraining hardcodesindex3 (10sec). New model must not reuse that loss contract.
- Ruling: use relative xyz and sin/cos(relativeyaw,roll,pitch),20samplesat.1sec; no old10sprogress in newpolicies. Reason: gravitypreserving,periodic, fixedhorizon; cost: additional orientation ambiguity nearEuler singularity must be evaluated withrotationmetrics too.
- Flipperzero/signcalibration asked asynchronously; not inferable from bagalone. Continue independent work, recordunresolved rather thanimpute.

- 2026-10-30 user confirms subcrawler physical angle reference: outward horizontal=0deg, vertical upward=+90deg. Recorded flipper-calibration.json. This resolves the intended physical reference, not by itself a numerical ROS/asset axis-offset proof.
- Reuse-only development CUDA minibatch run100epochs completed; checkpoint8431e620..., all4trainheads have positive/negative support, valtip/nearpositives0. ValidationADE.0663m,FDE.1144m are preliminary reused-only evidence, not the final expanded-data model.
- Production has not started: found shared done.any() settling/prefix abort in collector; must replace with independent per-env masks. Actual5/10env restore benchmarks pass, but does not establish this settling failure path.

- First production cohort completed:10groups×14branches×40intervals=5600rows. All14anchor restoration checks passed; current collector source SHA matches staged binding. Fresh115group integration dataset (10new+105reused) built; fixed missing legacy candidate-cycle metadata in R6 adapter without editing raw traces.
- Actual production continuation launched session70047,69remaining basejobs; outputs originalworkspace collection/, log base-production-continuation.log. Latest confirmed live; completed1/running1. Supplemental10jobs held for actualpositivecontact and geometry validation.
- Latest R6 foundation suite59tests passed11.29s. Oldtest_r4_star14 in dataset-free clone has2expected missingsealed-data failures; no fulllegacy regression claimed yet.
- Input physical flipper reference userconfirmed0outward/+90up. Pending simulation-axis numericalmapping audit and finalbagdistributionreport, no meanimputation.
- Agents stopped due workspace-owner spend cap. Main agent continues; final independentreview remains required and not yet satisfied. Do not claim finished or pushcomplete.
- Next work: while basecollectruns, verify R6 eval actual1/2perparentgeometry and dynamicbody collisionbounds (USD staticworldbbox maynottrackFabric); positive filteredfixturecontact test, exactcase60cm/1m supplementalcoverage; statistical thresholdrecord builder and resumableval/testpipeline. Plannedval13650/test6300 are NOT completed.
