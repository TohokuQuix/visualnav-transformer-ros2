# Forward candidate evaluator

**Stopped before training:** the raw record lacks the exact 22-D pre-action actor observation. The collector consumed `obs["policy"]` to compute action but serialized only action, command, pose, twist, DEM, reward and Critic. Joint positions/velocities and observation noise realizations were not saved, so neither the observation nor a latent can be reconstructed exactly. Substituting twist/command would violate the specified body-state input and would not permit an action-consistency check.

Actor checkpoint SHA: `b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d`. Raw records scanned: 360,000. Observation statuses: {'missing': 360000}. Empirical normalization was disabled, but observation corruption was enabled. Actor latent was not saved.

E-state, E-traj, E-state-traj and E-latent-traj were not trained or evaluated. No predictive signal, candidate ranking ability or body-state benefit can be claimed from this Phase B. A future dataset needs the exact pre-action `obs[policy]` per session/env/episode/step and action, with checkpoint provenance. This would require separate collection authorization; none was performed here.
