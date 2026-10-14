# 14-Path Stage A feasibility protocol v1

The pilot uses one existing MI-A terrain StateGroup, `random_rough` level 2 seed
`2026101602`, both Front-folded and Rear-folded reset postures, four env tiles,
and the frozen bank order `(0, -30, +30, -60, +60, -90, +90)` for Forward then
Reverse. Per posture, two pre-Path fresh-process records precede 14 fresh-process
candidate executions. There are 32 processes and 112 candidate trial traces.
The existing C++ follower, RL checkpoint, 20-step settling, 20 Hz control,
40-step label, DEM, noise, physics and Gate P are unchanged. No goal is input.

The angle denotes the bank's tangent-turn geometry parameter, not measured body
yaw. Each Path has eight points on a 0.75 m reference circular arc. Direct and
±30° must match the sealed MI-A six-Path coordinates and SHA exactly. All
geometry audits complete before any simulator Path execution.

## Gate fixed before execution

- Offline geometry: fourteen finite, unique valid Paths; correct origin,
  initial tangent, waypoint spacing, arc length, curvature and mirror symmetry.
  Existing six Path byte SHA must match sealed MI-A records.
- Every pre-Path process and candidate process must pass host/inner seal,
  input/source identity, resource and Gate P checks. Four env tiles must each
  publish one candidate Path and yield forty finite control rows and an
  available finite 2-second body-frame XY/yaw label.
- Resource hard caps remain GPU device-total 6,656 MiB, host available RAM
  at least 2 GiB, batch storage below 800 MB and 600 seconds per process.
- Pass requires all 32 processes and all 112 trial traces protocol-valid.
  Verified invalid geometry, Gate P, label, Path publication, termination,
  source identity or seal is Fail. Missing evidence or infrastructure failure
  that prevents a scientific conclusion is Inconclusive. Stop at first failed
  process; keep sealed prior processes unchanged.
- There is **no** minimum XY displacement or forward speed gate. Low motion,
  stall or angularly capped motion is a scientific observation and must be
  reported, including for ±90°. No Path, follower, Actor or physics setting is
  changed in response to outcome.

After a protocol Pass, summarize 2-second XY/yaw, longitudinal velocity,
follower linear/angular command, contact, torque, termination and Forward versus
Reverse and Left versus Right by posture and angle. This single-seed feasibility
pilot does not establish terrain generalization.
