# Phase B-MI Stage 1: pre-Path stop

**MI-A: Inconclusive (candidate outcome not assessable).** The first of 72 frozen fresh-process runs stopped after 20 candidate-free settling steps and before Path activation. No candidate motion label exists. The frozen B-R1 Fail remains unchanged.

The first run used `pyramid_stairs`, terrain seed `2026101305`, level 2, four envs, repeat 0, Forward Direct. The mirror-control gate required actual tile-local |y|≤0.02 m, |yaw|≤0.02 rad, reflected robot-relative DEM mask IoU≥0.99 and reflected finite-height RMS≤0.005 m. All four envs violated yaw: absolute values were 0.0359, 0.0955, 0.0627 and 0.0358 rad. All 16 DEM frames had mask IoU 0.8182. Height RMS was at most 7.94×10⁻⁸ m, well below its limit. The observed mask contains a source-footprint clipping pattern, including 9 valid cells in each outer row, and is not a symmetric observation even on geometrically symmetric terrain. Thus terrain name and fixed reset yaw did not establish an actually mirrored pre-Path observation after settling. The body yaw drift is an independent reason for failure. Do not infer a Path or motion effect from this attempt.

The simulator wrote a diagnostic `failure.json` and closed with native exit 0. The host explicitly rejected the failure marker. Inner and outer ledgers match all listed file SHA values; neither seal has extra files. There were 14 resource samples: peak device-total GPU 5,704 MiB against 6,656 MiB cap, minimum host available RAM 84,498,584 KiB against 2,097,152 KiB floor, peak active host batch storage 259,378 bytes against 800,000,000 bytes cap. Wall time was 14.68 s. No candidate Path was published, so the schedule stopped at 0/72 and was not retried.

Evidence: `/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/mi-a-v1-20261013/`. The first attempt has `prepath.npz`, `prepath-identity.json`, `mirror-audit.json`, `failure.json`, `resource-series.jsonl`, `native-exit.json`, `decision.json`, and verified inner/outer SHA ledgers and seals. The attempt manifest SHA is `e8d73abd4f443d7e083794699b7d4046633ddd4af845b34f3e892d889fcf4107`; outer seal SHA is `444ffc985991a663f347cd997b026ccb10bed3e05a65564cb8d150f6525ee468`. Source, protocol and assignment SHAs were frozen in `frozen-provenance.json` before launch. Driver: NVIDIA 580.178.04, RTX 5070 Ti, GPU UUID `GPU-e9414d4c-dcc5-7554-bade-d7523dd94a26`.

The current design cannot pass its mirror-control evidence gate without a new protocol. A future design review should separate terrain geometric symmetry from observed DEM validity footprint symmetry, and decide how to handle natural yaw drift after settling. The v1 numeric gate is not relaxed or reapplied to a changed definition. Any revised pilot needs a new version, new threshold rationale fixed before another rollout, and fresh roots. No ±30 candidate run, ±60/±90 extension, predictor training, goal-selection evaluation, or Phase C collection followed this stop.

Reproduction command (requires the existing Isaac container, GPU and unchanged source/input):

```bash
cd /home/kojima/lib/visualnav-transformer-ros2-motion-predictor-vnext
PYTHONPATH=src:. /home/kojima/anaconda3/envs/tinyvla/bin/python scripts/eat_nav_motion_identifiability_schedule.py --root NEW_PERSISTENT_ROOT
```

The sealed root above must never be used as `NEW_PERSISTENT_ROOT`.
