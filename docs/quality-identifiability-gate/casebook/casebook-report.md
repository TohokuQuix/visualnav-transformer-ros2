# Held-out trajectory casebook: findings

The model and split were frozen. 270 unique complete held-out windows were inferred with eight paired diffusion seeds and six q values.

## 1. Low-quality teacher imitation

No-progress (B): enabled teacher RMS 0.0369 m; 17/30 B cases meet the frozen low-teacher-reproduction category relative to their high-progress control. Tilt-before (A): 0.2067 m; all 67 complete windows belong to one tilt episode, so no episode-cluster CI can be estimated for A. These are static XY distances to teachers, not failure rates.

## 2. Effect of raising q

In B, teacher RMS at q=0.1 is 0.0780 m and at q=0.95 is 0.0714 m (high minus low -0.0066 m). In A, the corresponding change is +0.0047 m. Positive means less teacher imitation at high q. Thus, high q slightly **increases** imitation of no-progress B teachers, while it slightly reduces imitation of the one tilt episode's teachers. The B paired difference has episode-bootstrap 95% CI [-0.0074, -0.0057] m; A has no episode-level CI.

## 3. q versus sampling

Across all selected windows, median q-effect RMS is 0.0132 m and median sampling RMS is 0.1106 m; median ratio 0.140. The frozen sampling-dominance category is 270/270; the stricter q-ignore category is 0/270 because a small q trend can remain coherent. These categories must be read with the continuous effect sizes.

## 4. Alternative samples

At q=0.5, B has median across-seed forward-progress range 0.2778 m; 100.0% of B cases have both forward and backward endpoints among seeds. This is a static candidate-diversity measure. In cross-q matched teachers, only 1/50 anchors switch nearest teacher between q=0.1 and 0.95, and 0/50 select low at 0.1 then high at 0.95. A candidate near the high teacher is not established as physically safer.

## 5. Quality-enabled versus disabled

For B, mean enabled/disabled teacher RMS is 0.0369/0.0357 m. For A, it is 0.2067/0.1859 m. Smaller RMS means closer imitation of the teacher, not safer behavior. B enabled-minus-disabled 95% episode-bootstrap CI is [-0.0004, 0.0028] m, so this sample does not establish a difference. A is a single episode and provides no episode-level uncertainty estimate. The raw pose quaternion and q anchor timeline for that episode are plotted in `plots/tilt-episode-timeseries.png`.

## 6. Goal provenance and redundancy

Dataset goal is derived from the realized teacher endpoint (radius cap 0.8 m). Across all 6,208 complete held-out windows, goal distance/teacher endpoint displacement r=0.935; q/goal-distance r=-0.331. B and matched high-progress control mean goal lengths are 0.0164/0.3647 m. The nearest available B controls still have a mean goal-length gap of 0.3483 m; they are not goal-equivalent controls. This is a material limitation caused by the realized-endpoint goal definition. Raw command is velocity, so a direct distance gap is dimensionally invalid. Median command/goal direction difference is 8.3 degrees where both directions are defined.

## 7. Most plausible mechanism

The realized-endpoint goal encodes much of the observed progress and is a concrete source of redundancy. Teacher imitation, q response and diffusion variation must be interpreted together using the continuous metrics and case pages. Static inference alone cannot uniquely attribute causality to one component.

Checkpoint SHA-256: Quality-enabled `fcaf3b9397f3344a0017a52e324ad11ce879c29d9737e1b924d3800293e00987`; disabled `c8187d5b57baf6ddfb6ab36029de7b644b8dcfa699553bb6fbe260942c7465fb`. The longer disabled SHA in the request was a transcription error; the actual file matches the previous gate report.

## 8. Replay candidates

Candidate held-out complete-window indices (selection rule recorded in casebook-interpretation.json): [1791, 3822, 2622, 6165, 6164, 4052, 3944, 3665, 2569, 5706, 367, 3600, 4906, 4019, 321, 4619, 716, 6152, 2575, 3662, 5397, 4479, 2538, 6049, 3632, 3867, 314, 2215, 3329, 2833, 3586]. A separate 20–50-case closed-loop simulator replay would test contact and failure behavior.

## 9. Collection decision

Full collection remains stopped. A replay or explicit design decision is needed before changing dataset goal, model, loss, or collection scope.

## Support and limits

All five terrain types represented in the selected 270 windows have at least one sparse q bin under the preregistered 10-per-bin rule, so terrain-specific category estimates are NA. They are reported as insufficient support in casebook-summary.json. Episode-cluster bootstrap intervals for cohort category fractions and key paired differences are in casebook-interpretation.json. No claim is made for all seven terrain types or physical safety from XY trajectories.
