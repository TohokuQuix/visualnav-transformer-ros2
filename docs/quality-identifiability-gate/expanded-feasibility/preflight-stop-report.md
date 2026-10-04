# Expanded candidate-feasibility pilot: preflight stop

**Status:** `X17_STOPPED_ROUTER_UNAVAILABLE`. No new anchor or candidate process started. The externally managed Zenoh router was absent and `127.0.0.1:7447` refused a host-side connection. The pilot does not own that router and did not start one.

The new branch `exp/eatnav-expanded-feasibility` was based on the clean corrected-addendum commit `ca6e112886469ce8af24f08b7654bc2d57ad7388`. The existing 18 E1 runs and corrected addendum were read-only. The complete-only train split, frozen candidate bank, original E1 thresholds, checkpoint and reference config were SHA checked before selecting new states.

| Terrain | Environment | Episode | Anchor step | Terrain seed |
| --- | ---: | ---: | ---: | ---: |
| boxes | 36 | 0 | 20 | 645838458 |
| hf_pyramid_slope | 54 | 0 | 65 | 2480648597 |
| k_rail_field | 6 | 0 | 15 | 2373776937 |
| k_rail_step_field | 73 | 0 | 75 | 964068328 |
| pyramid_stairs | 47 | 0 | 55 | 3361987328 |

Selection used the fixed SHA salt and only terrain, episode/window identity and terrain seed. All five are complete-only train windows with four finite DEM frames. Raw recorded frames agree with the selected window DEM SHA. The 35 frozen cubic Bezier Paths passed E0 Type A geometry gates; curvature and footprint coverage remain diagnostic. The 50-process manifest contains five anchors, 15 direct repeats and 30 lateral candidates, with 35 unique candidate outcomes. No new outcome was observed.

The old 18-process run took 2772.4 s wall time. For 50 processes, the forecast is 7117 s point, 7344 s P95 and 8813 s after the required 20% margin, below the 14400 s cap. Storage with margin is estimated at 0.518 GB, below the 1.0 GB cap. These are forecasts; the new terrain runs have not been timed.

The router failure occurred before anchor replay, duplicate controls, candidate rollout, X4 body-signal floor freeze, map Path creation from new live anchor poses, P/C/A/R/Y evaluation, safety-support evaluation, and final seven-terrain judgment. None of those results can be reported as pass. No scaled collection, evaluator learning, formal validation or robot work was started.

Artifacts are under `/tmp/eatnav-expanded-feasibility-20261004/`. The old E1 and corrected-addendum roots were not written. Restart requires external Zenoh readiness and a review of the stopped state; the script has zero automatic retry.
