# Radial anchor execution contract and fixture plan

Base commit: `0fa3cb5b7828c78ad74221a5f7ecf057333a449d`. Specification: attachment `a9177528-ebc7-430d-9b5e-aa838995093b`.

1. SHA-audit four selected source records, three original shard freezes/manifests, Git simulator/reset/terrain sources, container checkpoint/config and copied r5 manifest. Write A1 input-sufficiency table before code.
2. Freeze the radial contract, prior-source tolerances and fixture manifest in a new output root. Use the original three complete shard manifests so contact-view construction and pose order stay unchanged; gate only the four selected anchors.
3. TDD a pure adapter that rejects synthetic E1 prefix metadata, verifies shard/anchor identity, produces an output-only manifest overlay and evaluates marker/termination/DEM/Actor/source invariants.
4. Forecast three candidate-free 112-env, 200-step shard runs against P1 v2 caps with 20% margin. Run only if the forecast and source identity pass.
5. Execute each original settling runner in a fresh Docker process to a distinct subroot. Do not launch ROS, bridge or follower. Capture PID/start tick, resources, completion, native exit, cleanup and host/container SHA.
6. Compare the four selected outputs against immutable saved records using pre-frozen tolerances. Stop at the first physical or numerical failure; never relax thresholds.
7. Save decision, SHA ledger, report and small comparison figures. Keep raw artifacts outside Git and commit source/config/tests/small report files only.

Review focus: original source SHA remains exact; no synthetic episode prefix; no Path publication; Actor vector phase is not silently relabeled; output overlay changes no logical manifest field; missing marker cannot pass on exit code alone.
