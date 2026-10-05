# Expanded E1 pilot: restored anchor config, X2 startup import stop

Date: 2026-10-05 JST. New output root:
`/tmp/eatnav-expanded-feasibility-config-restored-20261005`.

## Config restoration and frozen inputs

The SHA-verified original E1 anchor config contains
`velocity_max_abs: 0.0001` and `command_max_abs: 0.00001`. Those exact values
were added to all five new state configs. Parsed config comparison confirms
that removing only those two keys returns each previous config exactly. Every
other copied input file has the same SHA as the earlier failed root. The
failed anchor and all earlier output roots were not copied or reused.

The per-state config SHAs are recorded in `config-restoration-freeze.json`.
The 50 manifest rows and `input-freeze.json` were updated only where they
refer to those five config SHAs. New manifest SHA-256:
`b65868f9b3b6b7433be946f8385bb574d15574e2962f3aecd0b85f73231f016a`.
New input-freeze SHA-256:
`58f2ff2138647cd17cb2d2e7d390f27eb4e28413af28ba25f2a1b93ab192154d`.
The state, goal, local Path, seed, candidate ID, and run order are unchanged.
The runner checks these new SHAs and the complete original E1 anchor-limit
contract before launch. The focused suite passed 15 tests.

## Router and X2

X1 passed against the operator-owned router PID `2852834`, start tick
`178039512`, ZID `23466e323413a4e4c65e887017a57092`. The host and
container reached `127.0.0.1:7447`. The stdout SHA remained
`f5b955dc9df162161b2a4fcfc0586ae4bb01bd2b3b1588f43262f3926a7f9b85`;
stderr was empty with SHA `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
The same router PID/start tick and log SHAs were checked after the failed X2
invocation. The runner did not manage the router.

The first X2 command exited before constructing the bridge or launching a
follower. `eat_nav_command_delivery_bridge` could not import
`visualnav_transformer` because the command lacked the repository `src`
directory in `PYTHONPATH`. This is a diagnostic invocation environment error;
no readiness samples were acquired, so X2 is **not passed**. A later
import-only check with the repository `src` and `scripts` paths prepended to
the ROS environment passed. The candidate-free X2 diagnostic was not rerun.

The four-hour pilot clock was not started. There are zero new anchor,
duplicate, and candidate processes in this root. No candidate Path was
published. The external router remained present, and no follower process
remained after the import error. Automatic retry is disabled, so the pilot
stops at `X2_STARTUP_IMPORT_ERROR`. No result from this root is an Outcome.

## Evidence and next step

- `config-restoration-freeze.json`, the five `inputs/state-*/config-120.yaml`
  files, `run-manifest.json`, `input-freeze.json`
- `external-router-provenance.json`, `router.stdout.x1-snapshot.log`,
  `router.stderr.x1-snapshot.log`
- `x2-runner.log`, `x2-startup-failure.json`, `status.json`,
  `final-evidence-sha256.json`

A new X2 invocation requires explicit authorization under the zero automatic
retry rule. It must use the ROS environment with the repository `src` and
`scripts` paths prepended to `PYTHONPATH`, repeat X1 immediately beforehand,
and start no pilot clock until candidate-free X2 passes. The frozen model,
control law, Path, state, seed, order, and numeric thresholds remain unchanged.
