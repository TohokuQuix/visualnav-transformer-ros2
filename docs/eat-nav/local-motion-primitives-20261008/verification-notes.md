# Verification and location

Decision: `local_motion_primitive_terrain_support_fail`.

The 26 SHA-listed review artifacts and ten figures are committed in this directory. The identical generated root is `/tmp/eatnav-local-primitives-offline-20261008-r4`; its files were copied byte-for-byte and both copies passed the SHA ledger check. The frozen contract SHA is `4bbcdbb6d742a7d31645f2b0dd2e9558dabc00beec3ee755a1bfbea310d83d4a`.

Relevant geometry, prior-bank, provenance and fixture tests: **77 passed**. The whole repository suite could not collect in the available local Python environments because unrelated packages `diffusion_policy` and ROS `geometry_msgs` are absent. No simulator, follower, bridge, router, physics rollout or training was started.

The code review identified four gaps, all corrected before this result: exact archived anchor/Path frame checks, full Path-arc DEM diagnostics, numeric Bank A/B/G1–G3 StateGroup comparison, and SHA for the new generator/review runner. The primary 0–0.60 m terrain threshold was unchanged.
