# P1 r4 execution plan

Base: r3 commit `3e9bd9db0f0ab65ce12e939600564e7a996fa257`. Authoritative spec: attachment `03cbe67e-2ce4-44d8-a8a2-fa9ddb956437`.

1. Verify the r3 SHA ledger and every source SHA; copy the frozen r3 anchor and manifest bytes into a new r4 root without regenerating them. Create a run-output overlay only if later gates pass.
2. Test-first implement resource contract v2 and a fresh NVML baseline. Known PID/start-tick other-process use counts in device total but does not consume the 256 MiB uncertainty allowance. Test the r3 measured 311.859 MiB case, unexpected compute, missing PID identity, unresolved bytes, gap, and device baseline.
3. Freeze v2 config and resource forecast; sample GPU immediately before X1. Stop if any contract gate fails.
4. Audit the operator-owned router. If absent or provenance unavailable, stop before X2. If valid, run candidate-free X2 and a second baseline. Start the 28-row process clock only when both pass.
5. If started, run frozen rows sequentially with no retry, seal each, then perform duplicate, P→C→A→R→Y, two-second Outcome and relabel analyses. Stop on the first frozen gate failure.
6. Save stage status, primary decision, SHA ledger and concise report. Run focused tests and verify final worktree state.

Review focus: ensure no r3 files change; known graphics memory is never treated as attribution error; no candidate Path or process starts before X1/X2; no missing input is silently substituted.
