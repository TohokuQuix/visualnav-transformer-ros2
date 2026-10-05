# Execution-aware Evaluator: architecture options

This is a design comparison. No Evaluator was trained.

| Option | Predicted quantity | Strength | Limitation from seven StateGroups | Recommendation |
| --- | --- | --- | --- | --- |
| A. Absolute ΔV regression | Candidate ΔV in frozen Critic units | Calibrated score and direct comparison are possible | Initial Critic spans 37.50–39.15; ΔV top-1 agrees with H-step advantage in 1/7 and reward in 0/7 groups. Checkpoint-specific scale and uncertain teacher semantics make a single absolute target weak. | Diagnostic baseline only after label revision |
| B. Within-group pairwise ranking | Relative candidate order at one anchor/goal | Removes group-level offsets; candidate selection matches the operational task | Same-group pairs are correlated. Direct repeat noise is exactly zero, so repeat-based ratios are undefined and cannot establish stochastic robustness. Label conflicts still require a quality definition. | Main comparison baseline after label revision |
| C. Multi-task ranking | Relative score, ΔV/advantage, 0–2 s progress, drawdown/tracking, uncertainty | Exposes label disagreement and can retain physical diagnostics instead of hiding them in one scalar | More heads and weight choices demand more independent StateGroups and prospective weight freeze. With no failures, a safety head has no positive support. | Preferred architecture concept; do not train yet |

The saved 6 s ΔV/advantage pair is sensitive to the four seconds after the 2 s Path horizon. An exploratory 0–2 s recomputation raises macro rank correlation from 0.209 to 0.684, while top-1 agreement remains 2/7. Horizon alignment must be settled prospectively before fitting any loss.

The portable primary path is **4-frame DEM + mask, base-frame body state, relative goal, and eight-point candidate Path**. Path/DEM features are deterministic and preserve unknown coverage. An optional checkpoint-dependent branch may add the **anchor-only Actor latent**; it must not use the post-Path live latent. Raw joint effort is excluded until an anchor-time capture and real-robot unit/order/timing parity are demonstrated. Candidate ID, family, amplitude, run order, absolute environment origin and all future response signals remain metadata or labels, never primary inputs.

Each training example is a StateGroup with a set of seven candidates. Pairwise loss can be computed inside it, but group weighting must sum to one group contribution rather than treating 21 pairs as independent samples. Validation and confidence intervals resample StateGroups or source environment/seed clusters. Ties use a prospectively fixed measurement floor; an observed zero duplicate floor is stored as zero, not silently replaced by epsilon.

Safety events require separate cause-specific labels and a separate head. Current 0 failure-positive live traces and 0 verified stacks make it untrainable. Timeout censoring and infrastructure failure must remain masks/causes, never safe negatives or robot failures.
