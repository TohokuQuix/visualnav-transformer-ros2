# Stage B collection continuation after pre-Path reference error

The first Stage B root passed its 32-process, 112-trial first-seed smoke.
At assignment index 32, the first `pyramid_stairs` level-1 **new seed**
`2788966343` was compared against seed `1456788336`'s pre-Path reference.
The v1 coordinator selected the first reference with the same flipper posture
but omitted `state_group_id` from its lookup. The exact reference SHA in the
failed input manifest matches the old seed's pre-Path snapshot. Gate P failed
before Path activation, so index 32 has no candidate Outcome. Simulator native
exit was zero because the diagnostic exception was captured; the host decision
correctly marked the attempt invalid. Its inner/outer ledgers and seals remain
consistent and unchanged.

The corrected coordinator requires the same StateGroup **and** posture for
repeat-0 pre-Path reference. A regression test reproduces the old cross-seed
selection and validates the corrected lookup for index 32, its repeat and its
candidate. The original source, manifest, 32 valid attempts and failed index
32 are left intact in `motion-14-collection-20261014-01`.

Continuation uses new root `motion-14-collection-20261014-02` and frozen
`motion_14_collection_continuation_v2.json`. The 32 valid attempts were copied
byte for byte and reverified against their original outer-seal SHA. The failed
index 32 is excluded; its new attempt began with no cross-seed reference and
passed, as did the same-seed repeat. The assignment, seed, Path, Actor, C++
follower, physics, noise, material, Gate P and resource limits are unchanged.
Subsequent attempts run in the original manifest order. Dataset construction
will retain the original and continuation coordinator provenance separately
and reverify both the copied prefix and excluded failed seal.

This is an implementation correction to the reference lookup, not a change to
the scientific comparison or an outcome-based retry.
