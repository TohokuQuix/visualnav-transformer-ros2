# P1 long-horizon G1/G2 code and static audit (2026-10-11)

**Decision: code/synthetic preparation complete; G1 remains inconclusive.** No new Isaac simulation, G2 runtime, or G3 pilot was started. The prior G1 attempt-0 host seal and decision were read only.

## G1 GPU monitor

`scripts/eat_nav_p1_longhorizon_gpu.py` uses the existing NVML compute+graphics per-PID bracket in `eat_nav_p1_gpu_attribution.py`. It records GPU UUID, physical device total/used memory, per-PID used memory and classification, host PID/start tick, `/proc` tables before/after, owned simulator identity, and the bracketed attribution gap. The G1-specific check rejects a gap over **256 MiB**, an absent owned GPU process, an unresolved PID/start tick, and any GPU PID that vanishes or changes start tick during a bracket. `capture_live` propagates NVML failures; the caller must fail closed. The exact source command and container cgroup must identify one simulator process. The tests use synthetic NVML/process data only.

`scripts/eat_nav_p1_g1_gpu_host.py` is a new host launcher prepared for a **new output root**. It obtains a prelaunch NVML baseline, identifies the simulator from its exact source command and container cgroup with PID/start tick, samples the live process through the G1 monitor, fsyncs each resource row, and checks native exit and owned residual. It refuses the `g1-attempt-0` root. It has **not been run**, so early identity acquisition and live NVML availability remain runtime checks. The current attempt-0 launcher used only `nvidia-smi` device total and must not be reused. Do not promote G1 attempt-0 to pass.

## G2 diagnostic runner

`scripts/eat_nav_p1_g2_diagnostic_runner.py` is a separate derivative of the historical P1 simulator source (SHA `1205147f7c63ed2e91d6981868f639f48af90f3ea6d692862f9f76713cca46b9`); the original P1 source and production follower are unchanged. `configs/eat-nav/p1-g2-contract.json` pins the diagnostic source SHA, `control_steps=600`, and `episode_length_s=40.0`. A G2 input root must contain a byte-identical copy named `g2-contract.json`; the runner checks its source SHA and refuses output outside the host-backed persistent root. All four diagnostic slots receive the frozen **0.75 m, 0° direct Path** and ordinary `A` pass-through command. It is a timing probe, not Path quality evidence. It uses the existing exact-stamp bridge and Actor inference path.

At every 50 ms logical control step, the bridge calls `clock`, then `poses` (TF), then waits for a fresh follower command. The exact-stamp bridge accepts a command only after same-stamp TF and Path publication. The simulator validates the bridge response, injects the accepted command, then calls `env.step`. The G2 trace records actual bridge clock/TF/accepted-command stamps, command acceptance steady time, physics start steady time, and episode counters. `SameEpisodeGate` rejects order/stamp differences, gaps, any termination/truncation/reset, and episode-counter discontinuity. Each slot's JSONL record is flushed and `fsync`ed after every step. A failed run retains its flushed prefix. The source does not claim independent wall-timestamp observation of every ROS publication; its order proof combines source order, exact-stamp acceptance, and command-before-physics steady time.

The runner refuses an environment other than 50 ms control step / 800-step timeout. G3's 640-step use of the same gate was tested synthetically; the G2 runner itself is fixed to 600 steps. The 20 candidate-free settling steps plus G2 600 or G3 640 remain below the 800-step timeout. Any physical termination still fails the single-episode timing gate.

## Episode-length source audit

- `velocity_env_cfg.py:405-424` sets decimation 10, physics dt 0.005 s, and episode length 20 s. The G2-only config assignment changes episode length to 40 s after `make_collection_cfg` returns, before `gym.make`. The original P1 runner/config is unchanged.
- `velocity_env_cfg.py:181-220` policy observation terms are velocity, projected gravity, command, joint state, and last action. No episode-length input appears. The G2 runner still checks live Actor observations; Isaac internal configuration is not yet runtime verified.
- `velocity_env_cfg.py:295-335` reward terms depend on motion/action/commands, not episode length directly. A longer trajectory can accumulate more reward; G2 does not use reward to judge Path quality.
- `velocity_env_cfg.py:338-351` keeps both `time_out` and nonfinite physical termination. The G2 change only moves the timeout; it does not disable physical termination.
- `envs.py:23-67` computes physics, termination, reward, then resets terminated envs and recomputes observation. A reset at step ~400 would mix episodes; G2 refuses it via done/reset and episode-counter checks.
- `curriculums.py:42-67` uses actual elapsed episode steps for terrain progression, evaluated on reset. The P1 source sets terrain generator curriculum false; its curriculum term remains configured. No reset is allowed during the G2 measurement, so the longer timeout should not trigger a curriculum transition during a valid trace. A physical early reset would invalidate the run. Runtime must verify the live manager configuration.
- `eat_nav_env_cfg.py:225-265` (`/tmp/eatnav-g1-eat-env-readonly.py` audit copy) preserves base config except EAT sensor/command/reset configuration; it does not set episode length.

## Verification and remaining gate

Synthetic/static suite: **52 passed** (`test_p1_longhorizon_protocol`, `test_p1_longhorizon_gpu`, `test_p1_g2_diagnostic_source`, and existing GPU/P1 tests). `compileall` passed for all new Python modules, including the host launcher. No actual GPU, Isaac Lab, ROS, TF, follower, or 600-step physics run was made. G1 source, production follower, bridge acceptance threshold, RL checkpoint, and prior P1 artifacts were not changed.

Before a G1 rerun, verify the new host launcher on the actual host in a new persistent root with exact PID/start-tick ownership, durable bracket logging, cleanup accounting, and the original resource caps. G2 must start only after G1 passes; its 600-step same-episode, continuous-trace, clock/TF/command/physics result remains unproven until runtime. G3 remains unauthorized.
