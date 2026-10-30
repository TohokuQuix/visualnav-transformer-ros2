# Observed data reuse audit

The immutable R4star fourteen-candidate dataset contains 105 StateGroups (81 train,
24 validation), 1,470 actual candidate branches, and 292,480 accepted 20 Hz rows.
Only level 4 was collected for its seven fields. k_rail_field has 9 train groups
from 5 terrain seeds and zero validation groups. Each other field has 12 train
and 4 validation groups from 8 seeds. 1,467 branches have a complete first 2 s.

The R6 artifact independently reconstructs future 0.1–2.0 s trajectories and
2-second events from actual root-state traces, rather than renaming 10-second
labels. Source candidate order is [0,1,2,7,3,4,5,6,8,9,10,11,12,13]. Both anchors
and all candidates sharing a field/seed retain the original development role.
Public batch01/02/03 reserved-test membership registries are excluded; no
reserved-test labels or raw traces were opened. Input DEM timestamps were
checked against each branch activation timestamp. Source trace hashes are saved.

R4star per-step root states retain xyz, wxyz quaternion and linear/angular world
velocity, commands and terminal snapshots. Per-step joints and contact are absent;
terminal-only contact forces cannot establish contact onset. Contact truth is
therefore unknown for reused R4star branches. Ordinary net terrain-support force
is not obstacle-contact truth.

2-second support [positive,negative,unknown] in event order tip,near,linear,turn:
train [[4,1130,0],[9,1125,0],[39,1092,3],[16,1115,3]];
validation [[0,336,0],[0,336,0],[6,330,0],[3,333,0]]. Validation currently has no
positive tip or near-fall examples; threshold selection requires new examples.
Observed trajectory samples: train 22,656; validation 6,720. Missing future samples
are masked and zero placeholders are excluded from loss.

Old batch01/02 train/validation traces retain per-step flipper position/velocity,
torque, joint velocity, contact net force and actor observations/actions in addition
to root and commands, but cover only four candidates and two fields at levels 1/4.
The cumulative development pool has 128 groups: 96 train/32 validation, 24/8 seeds.
The older motion14 collection manifest describes four fields at levels 1–3; it
requires a separate compatibility/provenance audit. There is no established
seven-field level0–4 dataset in the audited sources. New lower-level, wall,
narrow-obstacle, unilateral support and diagonal-approach collection is required.

## Collector host interface

`eat_nav_r6_dataset.pack_group(initial, candidates)` returns one unbatched array
dictionary and per-slot contact metadata. Initial arrays are dem(4,2,20,20),
actor_state(22), body_state(9). Each actually executed candidate dictionary has
slot 0..13, path(8,2), direction +/-1, and rows; unexecuted slots stay masked.
Rows follow the original 20 Hz schema with control_step starting 0,
logical_sim_time_ns, candidate_cycle, root_state, post_root_state,
follower_command, body_angular_velocity_radps, and terminal/reset/censor flags.
Call `validate_group_roles` before merging dataset groups to prevent field/seed
leakage. Call `validate_input_times(header, activation_ns)` for available headers.
The returned dict can be stacked directly into the standard R6 NPZ arrays.
