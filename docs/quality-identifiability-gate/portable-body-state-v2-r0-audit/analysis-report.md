# Portable body-state v2: R0 deployment identity stop

## Decision

`onix_policy_bridge_deployment_identity_inconclusive`. The robot-side `onix_policy_bridge` version cannot be proved from this host. The user instruction names this as an R0 stop condition, so no aggregated-feature mapping, v2 schema, adapter, fixture, or simulator run was attempted.

## What is confirmed

- The new worktree `/tmp/eatnav-vnt-portable-body-v2` uses branch `exp/eatnav-portable-body-v2` at base commit `345608970ba879b207d77af5994f6a883cf43c9b`. The relevant baseline tests passed: **11 passed** using `/usr/bin/python3` with `PYTHONPATH=src:scripts` and plugin autoload disabled. The earlier attempts under the active Conda Python and without `PYTHONPATH` failed for environment/import reasons; no test code was modified.
- Prior stopped artifacts were SHA-frozen. Both M0–M13 ledgers still verify **41/41** files. The Isaac Lab source, RL checkpoint/config, known E1 trace, and three bag metadata files are recorded in [source-provenance-v2.json](source-provenance-v2.json).
- A local `onix_policy_bridge` package exists at `/home/kojima/onix/dep/onix_automation/onix_policy_bridge` (version `0.1.0`, branch `imu_failure_fallback`, HEAD `d951d4db4290464a03651866a2082c476923d3f5`). Its config files have uncommitted changes, `model_2000.pt` is deleted, and `model_8000.pt` is untracked. The local install space uses symlinks through the build tree to this mutable source. With the local Onix overlay sourced, ROS resolves the package prefix to `/home/kojima/onix/install/onix_policy_bridge`.
- The local `onix_joy` launch includes the package's launch file. The local `onix2_bringup.bash` runs it on the robot. The operator-side bringup script explicitly SSHes to `quix@onix2.local`, sources **that host's** `~/onix/install/setup.bash`, then runs `ros2 launch onix_joy onix_joy.launch.py ...`. Thus the local symlink target is only a configured candidate, not proof of the robot's deployed code. The local launch default would choose the existing `model_8000.pt` unless overridden; the robot's selected checkpoint is unknown.
- This host could not resolve `onix2.local`, including a separately attempted read-only name resolution. No remote package prefix, Git status, source hash, launch override, checkpoint, normalization config, process identity, or bag-era deployment version could be read. The sampled April/July bags do not themselves establish the bridge executable version; the local install was rebuilt in July and can differ from what recorded them.

## Evidence boundary

The local package source and install chain are documented in [onix-policy-bridge-deployment-audit.json](onix-policy-bridge-deployment-audit.json). Comparing that mutable local code to Isaac Lab would answer what the **local candidate** does, but the request requires the version actually deployed on the robot as the primary source. No `exact_semantic_match` classification or portable v2 field hash can be justified yet.

## Work needed to resume

Read-only access to the robot's current Onix workspace or a SHA-addressed deployment bundle is needed. The minimum bundle is: `onix_policy_bridge` source and launch/config SHA; package prefix and symlink targets; Git branch/HEAD/status; exact launch arguments; selected checkpoint and normalization config SHA; and a record tying the bag-era version to its recording date if bag equivalence is claimed. Then rerun R0 and proceed to R1 only if deployment identity is established. No hardware command or new bag is needed for this identification.

No production control files, prior artifacts, bags, checkpoints, or robot state were modified. No simulator, bridge, follower, monitor, or router process was started by this audit. There is no owned process to clean up.
