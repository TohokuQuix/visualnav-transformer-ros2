# DEMO non-regression audit (MI-A v3)

The DEMO entrypoint `scripts/eat_nav_demo_isaac.py` and `demo_reset_policy.py` were not edited in MI-A v3. The existing `configure_demo_reset_policy` sets reset pose probabilities `(0,1,0)` and fold mode `(0,)`; its post-reset assertion checks `[RR,RL,FR,FL]=[0,0,π,π]` in the live robot. A fall-triggered reset repeats the same event. The Global Goal, ROS Path, follower, Actor and visualization pipeline source remains byte-identical to the MI-FI v1 base commit. The new collection allocation is read only by opt-in collection entrypoints, never by the DEMO entrypoint.

Static/unit evidence: `tests/test_eat_nav_demo_reset_policy.py`, `tests/test_mi_a_v3.py` and the prior sealed DEMO/MI-FI runs. A new Isaac DEMO reset trace was not produced in this phase; the live runtime result will not be inferred from static tests alone. Any observed Rear-folded DEMO reset is a protocol failure and stops this work.
