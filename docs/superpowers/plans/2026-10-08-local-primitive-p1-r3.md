# Local Primitive P1 r3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze earlier-onset transition anchors and a versioned 7,680 MiB GPU contract, then execute the 28-process P1 only if every baseline, router, X2 and runtime gate passes.

**Architecture:** Preserve r2 immutable. A new offline freeze validates prior SHA ledgers and four measured anchors, applies v2 selection, and emits a new 28-row manifest. A separate hardware baseline uses the corrected bracketed NVML accounting; dependent X1/X2 and rollout start only after its gate passes.

**Tech Stack:** Python, pytest, SHA-256, NVML, existing ROS2/Zenoh/Isaac pipeline.

**Spec:** `/home/kojima/.codex/attachments/bcc6eb4b-1282-496f-9b06-913352c46544/Pasted text.txt`

## Tasks

1. [ ] Verify clean r2 commit, create isolated r3 worktree/root, and SHA-check r2 and durable source records.
2. [ ] Write RED tests, freeze `minimal_p1_gate_v2`, and select the two unchanged differential plus two 0.20 m transition anchors.
3. [ ] Write RED tests and freeze 28 unique process rows, Path SHA, source/config/checkpoint SHA, seed 123, retry zero and distinct global goal/local endpoint.
4. [ ] Freeze P1-only resource contract, query GPU UUID/total/driver/MIG, and collect bracketed PID/start-tick attribution with a ≤256 MiB other-process baseline gate.
5. [ ] Run static verification, compile and focused tests; if all pass, verify external router, candidate-free X2 and a second GPU baseline.
6. [ ] Only after all gates pass, execute 28 sequential sealed fresh processes; stop at first protocol or resource failure without retry.
7. [ ] Only after 28/28 valid, analyze duplicate, P→C→A→R→Y, two-second Outcomes, goal relabels and role-specific limits; write report and SHA ledger.

## Stop Rule

Any gate failure stops its dependent work. A failing baseline is preserved with timestamps and attribution; unrelated processes are never managed. No measured candidate Outcome enters anchor or Path selection.
