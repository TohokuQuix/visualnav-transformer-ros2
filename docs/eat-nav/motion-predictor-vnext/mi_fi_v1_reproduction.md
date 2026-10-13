# MI-FI v1 reproduction commands

Code: branch `exp/eatnav-mi-fi-v1`, commit `fe00798` plus the report-only follow-up commit. Frozen runtime source/input SHA values are in `mi_fi_v1_frozen_manifest.json`. The completed evidence root is read-only and must not be used as a new output root.

Run targeted static/synthetic checks:

```bash
cd /tmp/eatnav-mi-fi-v1
PYTHONPATH=src:. /home/kojima/anaconda3/envs/tinyvla/bin/python -m pytest tests/test_mi_fi_v1.py tests/test_mi_fi_v1_analysis.py tests/test_mi_a_v2.py tests/test_eat_nav_demo_reset_policy.py -q
```

For a separately approved reproduction, choose a fresh persistent root and keep the frozen 16-row assignment:

```bash
cd /tmp/eatnav-mi-fi-v1
PYTHONPATH=src:. /home/kojima/anaconda3/envs/tinyvla/bin/python scripts/eat_nav_mi_fi_v1_pilot.py --root /home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/NEW_MI_FI_V1_ROOT
PYTHONPATH=src:. /home/kojima/anaconda3/envs/tinyvla/bin/python scripts/eat_nav_mi_fi_v1_analyze.py --root /home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/NEW_MI_FI_V1_ROOT
```

The pilot stops on the first failed gate, never reruns a completed process and does not start formal MI-A or training.
