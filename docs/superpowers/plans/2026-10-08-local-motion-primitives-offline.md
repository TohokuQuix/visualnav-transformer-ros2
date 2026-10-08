# Local Motion Primitive Bank Offline Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide whether five 0.75 m robot-local motion primitives with separate common navigation goals are source-valid, follower-compatible, terrain-supported, and ready for a later P1 rollout review.

**Architecture:** Reuse SHA-frozen anchors, source-parity checks, DEM corridor audit, and E1 gates from the completed 0.75 m review. A new pure module constructs analytic arcs and goal-relative labels; a new offline runner verifies provenance, audits runtime goal dependencies, performs geometry/terrain/relabeling analysis, and seals artifacts. It must never start a ROS or physics process.

**Tech Stack:** Python, NumPy, Matplotlib, pytest, SHA-256, archived E1 JSON traces.

**Spec:** `/home/kojima/.codex/attachments/994dbfc6-3342-4196-a066-86d7a0c8231a/Pasted text.txt`

## Global Constraints

- Base commit `74fea4003110d08f70f52bc1dc4a03526621f0d2`; prior roots and decisions are read-only.
- Five ordered endpoint bearings `[-20,-10,0,10,20]°`, 0.75 m chord, eight future base-frame XY points, forward tangent, 1.236 m^-1 screen.
- Common global goals are independently one of the same five 0.75 m bearings. Teacher remains 40 × 0.05 s; primary DEM arc is 0–0.60 m.
- Eight physical anchors produce 40 unique physical definitions and at most 200 correlated goal-conditioned label identities.
- No router, bridge, follower, simulator, rollout, collection, training, formal validation, or robot control.

## Review Focus

- A changed goal-metadata field must not enter follower command, Actor, reward or termination: test the exact source audit and stop on a physical dependency.
- At ±20°, analytic arc curvature and discrete eight-point curvature must both pass the unchanged screen: test formula and reconstructed endpoint.
- Changing common goal must change navigation labels but preserve locomotion targets and trace SHA: test archived-trace relabel fixture.
- Unknown DEM cells and arc after 0.60 m must not supply the primary terrain gate: test clipped known corridors.
- A repeated physical trajectory must not become five independent samples: test identity and split grouping.

### Task 1: Freeze provenance and source-valid goal dependency

**Files:** Create `scripts/eat_nav_local_primitive_review.py`; test `tests/test_local_motion_primitives.py`.

**Interfaces:** `verify_inputs(backup_root, goal075_archive)->dict`; `runtime_goal_dependency_audit(repo, backup)->dict`.

- [x] Write failing tests for archive/anchor/source SHA and for a synthetic physical goal dependency rejection.
- [x] Verify RED, implement frozen source graph with classified edges and source-exact evidence, verify GREEN.
- [x] Save source provenance, goal-dependency audit and contract SHA before any primitive generation.

### Task 2: Analytic primitive geometry and follower screen

**Files:** Create `src/visualnav_transformer/eat_nav/local_motion_primitives.py`; modify runner/test.

**Interfaces:** `constant_curvature_primitive(distance_m,bearing_deg)->dict`; `audit_primitive(curve,anchor,follower_config)->dict`.

- [x] Write failing tests for κ=2sinθ/d, signed endpoint, equal arc samples, exact five SHA, dense/discrete curvature, turn/reverse rejection.
- [x] Verify RED, implement pure arc construction and unchanged source-equivalent C0 audit, verify GREEN.
- [x] Audit all eight anchors with the same five Paths.

### Task 3: Terrain, identities, labels and archived-trace fixture

**Files:** Modify pure module/runner/test.

**Interfaces:** `terrain_support(anchors,primitives)->dict`; `goal_labels(trace,anchor_pose,global_goal)->dict`; `relabel_fixture(sealed_trace)->dict`.

- [x] Write failing tests for primary/full clipping, known-only terrain, three identity levels, goal-relative progress and trace SHA preservation.
- [x] Verify RED, compute 40 physical and 200 potential correlated labels, class/seed terrain gate and reserve-anchor suggestions from measured DEM only.
- [x] Verify one sealed archived trace and produce five goal relabels without new physics; verify GREEN.

### Task 4: Decision, conditional P1, comparison, forecast, artifacts

**Files:** Modify runner/test; create compact `docs/eat-nav/local-motion-primitives-20261008/*`.

- [x] Write failing tests for exact decision precedence and 28 physical P1 processes; verify ten required figures in the generated artifact set.
- [x] Verify RED, compare previous banks at StateGroup level, forecast measured resource distributions plus 20%, and generate only an unexecuted P1 proposal if every offline gate passes.
- [x] Run a new output root, verify every artifact SHA and relevant tests, commit the complete 1.1 MB offline artifact set, and request code review.

## Completion Boundary

End at one of M12’s five decisions. Report source, geometry, follower, terrain, coverage and fixture gates separately. A pass permits only explicit later P1 approval.
