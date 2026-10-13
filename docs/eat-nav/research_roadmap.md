# EAT-Nav Evaluator research roadmap and known limits

## Current evidence boundary

The GUI demo proves the software path from a ROS2 goal to candidate scores,
C++ PureFollower command, RL Actor action, and Isaac motion for a small set of
terrains. The pilot Evaluator checkpoint predicts 2 s common-goal progress
from only **one independent StateGroup×seed** of 12 balanced crossover trials.
The 35 terrain conditions were checked for generation and origin selection;
they were not a 35-condition navigation benchmark. The current ROS2 GUI demo
uses the ROS-free C++ follower core in its control loop. It does not execute
the production ROS follower. Real-robot transfer is untested.

The current bank is Direct, Left +20°, Right −20°, each eight points. A
global RViz goal is converted into a local subgoal within 0.75 m; rotation
and scaling can produce candidate inputs outside the small pilot training
distribution. The current model input expects exactly eight Path points.

## Research questions and controlled comparisons

| Question | Concrete comparison | Outcome and limitation |
| --- | --- | --- |
| R1: teacher label | Train separate models for negative motion residual, 2 s common-goal progress, accumulated observed RL reward, and Pain after engineering allowables exist | Motion residual measures command tracking, not goal success; progress is goal aligned but sensitive to geometry; reward depends on the RL reward design; Pain needs validated joint/body limits and combination weights. Keep labels separate until downstream evidence justifies combination. |
| R2: inputs | DEM on/off; Actor state on/off; Path geometry on/off; Critic-derived information as label vs inference input | Removing Path geometry can make candidates indistinguishable. Only use an input at inference if the deployment pipeline can supply the same signal at the same time. Critic values are currently more defensible as offline teacher/diagnostic data than a portable inference input. |
| R3: architecture | Current CNN DEM encoder + state/Path MLPs; smaller MLP baseline; CNN changes; temporal Transformer | Compare score error/ranking, held-out terrain/seed behavior, parameter count, inference latency, and input support. The existing network has a CNN; a two-layer MLP is not its DEM encoder. |
| R4: Path bank | Wider turns, small-radius curves, switchbacks, reversing, variable-length Paths | First verify geometric feasibility and follower execution. New geometry changes the training distribution; variable length requires a new mask/input/checkpoint contract rather than silently feeding eight points. |
| R5: generalization | Hold out terrain type, terrain seed, StateGroup, Path geometry, and initial pose | Split by independent StateGroup/source scene, not by individual Path from the same pre-Path state. Report support and sample size for every stratum. |
| R6: stopping | Quantify residual motion under zero command, then evaluate an independently specified stop controller | Cancel currently clears the Path and commands zero velocity but does not force the RL policy or physics to hold position. Do not call it a safety stop. A future stop intervention must be a new control protocol, with real-robot safety review separately. |

## Suggested experiment order

1. Preserve current demo/config/checkpoint as a reproducible baseline. Run a
   new demo only in a fresh persistent root and verify SHA/seal before use.
2. Acquire multiple independent StateGroups, terrain seeds, and initial poses
   using paired candidate rollouts. Record pre-Path DEM/state, Path geometry,
   command, Actor, contact, torque, reward, root state, and termination with
   one common 50 ms logical timeline. Do not treat env siblings as independent
   without checking their shared randomization and material conditions.
3. Make a versioned derived dataset with group-level train/validation/test
   split. Fit normalization only on train. Keep missing or censored labels
   masked; preserve failure trajectories and termination reasons.
4. Compare the four teacher definitions **separately** on held-out groups.
   Establish physically meaningful Pain allowables before training Pain.
5. Test input ablations under identical splits and candidate sets. Check that
   Path geometry has a meaningful candidate-dependent signal and that DEM
   context adds value outside the training terrain.
6. Compare model families at a fixed data budget and evaluate calibration,
   pairwise ranking, downstream selected-Path outcome, and runtime latency.
7. Freeze a model and candidate bank before an independent simulator pilot.
   Evaluate common-goal progress, local arrival, tracking, recovery, contact,
   and joint load separately. Do not infer navigation success from local Path
   arrival alone because curved candidate endpoints differ from Direct.

## Open issues students should see first

- Zero follower command after Goal/Cancel can coexist with continued Actor
  action, joint motion, and root displacement. The
  [residual analysis](cancel-residual-report-20261013.md) separates what was
  measured from what remains causally unresolved.
- `k_rail_field` level 2 had low goal progress in the short demo despite a
  working pipeline; one run cannot distinguish terrain difficulty, Path bank
  limitations, and model choice.
- The pilot checkpoint has no independent validation/test support. A lower
  training loss or appealing GUI score does not show path-quality ranking on
  unseen terrain.
- The demo candidate geometry is rotated/scaled toward an RViz goal, so its
  exact input distribution can differ from the frozen P1 teacher bank.
- ROS2 clock/TF/goal transport has been exercised locally; the production
  follower, hardware latency, and real-robot DEM alignment are not validated
  by this GUI demo.

See [environment setup](environment_setup.md) for startup and
[experiment pipeline](experiment_pipeline.md) for source files, tensor
contracts, and the boundary between implemented scripts and future work.
