"""Write the stopped clock-domain root-cause decision and compact report."""
from __future__ import annotations
import json
from pathlib import Path
from eat_nav_clock_diagnostic_stage0 import OUT,OLD,atomic_json,sha
from eat_nav_clock_diagnostic_analyze import load_trace
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120

RUNS=OUT/'clock-diagnostic-traces'


def main():
    alignment=json.loads((OUT/'clock-alignment.json').read_text())
    motion=json.loads((OUT/'follower-motion-window-trace.json').read_text())
    divergence=json.loads((OUT/'first-divergence-clock.json').read_text())
    parity=json.loads((OUT/'first40-parity-clock.json').read_text())
    assert len(alignment['rows'])==360 and len(alignment['source'])==3
    summaries=[json.loads((RUNS/f'run-{i:03d}/live-0.json').read_text()) for i in range(3)]
    traces=[load_trace(RUNS/f'run-{i:03d}/live-0.trace.json.gz') for i in range(3)]
    duplicate=duplicate_120(summaries,traces,.0105)
    assert duplicate['pass'] and duplicate['xy_rms_m']==0.0
    assert all(not x['warning_events'] for x in alignment['source'])
    assert all(all(r['ready']==0 for r in motion['rows'] if r['run']==i) for i in range(3))
    assert divergence['original_frozen_120_step_pairs']['run-001_vs_run-000']['command']==39
    assert divergence['original_frozen_120_step_pairs']['run-001_vs_run-000']['root_state']==40
    assert all(all(v is None for v in row.values()) for row in divergence['new_diagnostic_120_step_pairs'].values())
    original_trace=load_trace(OLD/'long-horizon/runs/long-006/live-0.trace.json.gz')
    sim_commanded_m=sum(abs(row['command'][0])*.05 for row in original_trace[:39])
    source_audit=json.loads((OUT/'clock-source-audit.json').read_text())
    follower_diff=json.loads((OUT/'diagnostic-instrumentation-diff.json').read_text())
    exact=json.loads((OUT/'diagnostic-fixture-exact-parity.json').read_text())
    results=[]
    for i in range(3):
        rows=alignment['rows'][i*120:(i+1)*120]
        windows=[r for r in motion['rows'] if r['run']==i]
        marker=json.loads((RUNS/f'run-{i:03d}/run-complete.json').read_text())
        results.append({'repeat':i,'elapsed_wall_sec':marker['elapsed_sec'],
            'follower_callback_count_with_motion_snapshot':alignment['source'][i]['callback_count'],
            'bridge_sequence_for_command_step_39':rows[38]['bridge_command_sequence'],
            'bridge_response_wall_elapsed_for_command_step_39_sec':rows[38]['step_wall_elapsed_sec'],
            'bridge_response_follower_node_elapsed_for_command_step_39_sec':rows[38]['follower_node_elapsed_sec'],
            'sim_nominal_elapsed_before_command_step_39_sec':1.95,
            'motion_window_elapsed_for_command_step_39_sec':rows[38]['motion_window_elapsed_sec'],
            'motion_window_max_sec':max(r['window_elapsed_sec'] for r in windows),
            'motion_window_ready_count':sum(r['ready'] for r in windows),
            'commanded_distance_for_command_step_39_m':rows[38]['commanded_distance_m'],
            'actual_distance_for_command_step_39_m':rows[38]['actual_distance_m'],
            'tf_age_for_command_step_39_sec':rows[38]['tf_age_at_follower_sec'],
            'reverse_event_count':len(alignment['source'][i]['warning_events']),
            'first40_pilot_xy_rms_m':parity['rows'][i]['comparison']['pilot']['xy_rms_40_m'],
            'first40_original_long_xy_rms_m':parity['rows'][i]['comparison']['original_long']['xy_rms_40_m'],
            'trace_sha256':alignment['source'][i]['trace_sha256']})
    previous_decision=OUT/'revision-0/clock-hypothesis-decision.json'
    archive_bytes=(json.loads(previous_decision.read_text())['new_archive_bytes_before_report']
                   if previous_decision.exists() else
                   sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()))
    evidence={
        'status':'diagnostic_complete_no_stuck_recurrence',
        'state_id':'coverage-003','candidate_id':'bank-nominal','authorized_runs':3,'completed_runs':3,
        'new_duplicate':duplicate,'new_repeats':results,
        'original_failure':{'run_id':'long-006','first_command_difference_step':39,
            'first_root_difference_step':40,'first_contact_difference_step':40,
            'linear_command_after_reverse_mps':original_trace[39]['command'][0],
            'commanded_distance_follower_m':5.929,'actual_distance_follower_m':0.165,
            'follower_ratio':0.028,'sim_time_command_distance_before_step_39_m':sim_commanded_m,
            'follower_actual_to_sim_time_command_ratio':0.165/sim_commanded_m,
            'nominal_sim_elapsed_before_step_39_sec':1.95},
        'hypothesis_classification':{
            'H1_clock_domain_mismatch':'supported_as_causal_mechanism',
            'H1_evidence':'wall timer plus ROS node clock without /clock; 20s motion history advances while only ~2s sim physics elapses; commanded distance ~5.9m matches ~0.3m/s over ~20s; pre-event original physics/Path/anchor equal; prior reverse occurs before root/contact divergence',
            'H2_callback_scheduling_jitter':'possible_trigger_cofactor_not_proven',
            'H2_evidence':'callback sequence at sim step39 differs across diagnostic repeats, but commands and physics remain identical; original exact callback schedule was not logged',
            'H3_follower_state_contamination':'not_supported',
            'H4_physics_divergence':'not_supported',
            'exact_original_boundary_trigger':'unobserved_in_original_trace',
            'why_not_reproduced':'all three new runs had no motion-window-ready callback; maximum measured window spans remained just below 20s (prune uses >20s and readiness uses >=20s). Diagnostic logging can also change wall scheduling.',
        },
        'production_source_unchanged':sha(Path(source_audit['source_sha256']['follower_source']['path']))==source_audit['source_sha256']['follower_source']['sha256'],
        'production_binary_unchanged':sha(Path(source_audit['source_sha256']['follower_binary']['path']))==source_audit['source_sha256']['follower_binary']['sha256'],
        'fixture_exact_parity':all(x['exact_equal'] for x in exact['rows']),
        'instrumentation_patch_sha256':follower_diff['patch_sha256'],
        'full_alignment_sha256':sha(OUT/'clock-alignment.json'),
        'full_motion_trace_sha256':sha(OUT/'follower-motion-window-trace.json'),
        'new_archive_bytes_before_report':archive_bytes,
        'runtime_sum_sec':sum(r['elapsed_wall_sec'] for r in results),
        'changes_applied_to_control_law':False,'candidate_comparison_started':False,
        'stage_b_restarted':False,'formal_validation_started':False,
        'proposed_fix_only':'Synchronize simulation /clock, follower and bridge node clocks in a separate protocol; retain production controller law and validate TF/Path stamp domain, command freshness, 40-step parity, and frozen duplicate before any candidate comparison.',
        'approval_required_before_fix_or_more_runs':True,
    }
    atomic_json(OUT/'clock-hypothesis-decision.json',evidence)
    compact={'source_sha256':{'alignment':evidence['full_alignment_sha256'],
         'motion':evidence['full_motion_trace_sha256']},
         'selected_steps':[row for row in alignment['rows'] if row['sim_step'] in (0,20,38,39,40,119)],
         'missing_value_policy':'unavailable follower internal snapshots after early-return callbacks are null, not inferred'}
    atomic_json(OUT/'clock-alignment-summary.json',compact)
    lines=[
        '# coverage-003 clock-domain duplicate diagnostic',
        '',
        'The approved nominal Path was replayed in three fresh simulator processes for 120 control steps each. No candidate run or unfinished Stage B run was started.',
        '',
        '## Frozen inputs and instrumentation',
        '',
        f"- Starting branch `{source_audit['git']['branch']}` at `{source_audit['git']['head']}`; the worktree was clean before implementation.",
        f"- Path SHA-256: `{json.loads((OUT/'clock-diagnostic-run-identities.json').read_text())['runs'][0]['map_path_sha256']}`.",
        '- Production follower uses `create_wall_timer` at 20 Hz (`simple_trajectory_follower.cpp:154-155`); its motion history uses `get_clock()->now()` from the control callback (`:373`) and integrates `abs(vx) * dt` (`:716-748`). With `use_sim_time=false` and no `/clock` publisher, that elapsed time follows wall time.',
        '- A diagnostic-only follower copy added logging for Path, TF, callback sequence, clock, motion-window state and command. No control branch, parameter, timer, publisher or subscriber was changed. Seven frozen production bridge command fixtures matched exactly.',
        '- The bridge and simulator runner copies added timestamp and sequence fields. Source and patch SHA values are in the audit JSONs; the production source and binary SHA values still match D0.',
        '',
        '## Three nominal repeats',
        '',
        '| Repeat | 120-step wall time | Response wall time for command step 39 | Motion window for command step 39 | Max window | Ready callbacks | Bridge seq |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for x in results:
        lines.append(f"| {x['repeat']} | {x['elapsed_wall_sec']:.1f} s | {x['bridge_response_wall_elapsed_for_command_step_39_sec']:.2f} s | {x['motion_window_elapsed_for_command_step_39_sec']:.9f} s | {x['motion_window_max_sec']:.9f} s | {x['motion_window_ready_count']} | {x['bridge_sequence_for_command_step_39']} |")
    lines += [
        '',
        '- The bridge response during `env.step(38)` supplies the command recorded in trace step 39; sampled response values match the next trace row after float32 rounding. Nominal simulation time before command step 39 is 1.95 s. The follower had accumulated about 5.91–5.93 m of commanded distance around that response, matching roughly 0.3 m/s over about 20 s of wall-clock motion history.',
        '- All three new repeats had identical command, Actor input/latent/action, Critic, root, contact, DEM, and reward over all 120 steps; duplicate XY RMS was 0 m against the frozen 0.0105 m threshold.',
        '- All new first-40-step traces matched the original 40-step pilot exactly. Repeat 1 differed from its original failed 120-step trace first at step 39, as expected because the new run did not reverse.',
        '- No new `Linear stuck detected` event occurred. The motion window never became ready; its measured maxima were a few tens to hundreds of nanoseconds below 20 s. The original event cannot be dismissed as resolved: the logging build changes scheduling, and the original internal callback times were not recorded.',
        '',
        '## First divergence and causal assessment',
        '',
        '- In the original long-horizon triplet, run `long-006` first diverged at follower command step 39. Actor input/action and reward changed at the same step; root and contact first changed at step 40. Anchor, Path and prior control/physics signals matched the other nominal repeats.',
        f"- The original follower warning reported 5.929 m commanded, 0.165 m actual, ratio 0.028. Integrating the frozen simulator commands for the first 39 × 0.05 s before that command gives {sim_commanded_m:.3f} m; 0.165 m divided by this is {0.165/sim_commanded_m:.3f}, above the 0.25 stuck threshold. This is an illustrative time-domain comparison, since the follower's exact TF motion samples from that old event were not logged.",
        '- H1 clock-domain mismatch is supported as the causal mechanism. The wall-driven 20 s history and distance accumulation advance much faster than simulation physics. The controller prunes samples older than 20 s and calls the window ready at >=20 s, making the trigger sensitive to an exact callback boundary.',
        '- H2 callback scheduling is a possible trigger cofactor. New repeats differed in callback count and wall timing but did not differ in command or physics. The exact old callback boundary is unobserved. H3 state contamination and H4 physics-first divergence are not supported by the available traces.',
        '',
        '## Stop and proposed validation',
        '',
        '- The production controller, thresholds, timer, `use_sim_time`, and `/clock` setup were not changed. The bridge and follower diagnostic processes were stopped after the third run.',
        '- Before any fix or candidate comparison, obtain approval for a separate clock-domain protocol: synchronize simulator `/clock`, bridge/follower node clocks and TF/Path stamps; verify command freshness and the original production fixture, then repeat the frozen duplicate and 40-step parity checks. The proposal does not authorize implementation or further rollout.',
        '',
        f"- Raw alignment and callback trace SHA-256: `{evidence['full_alignment_sha256']}` and `{evidence['full_motion_trace_sha256']}`; stored under `{OUT}` outside Git.",
        f"- Three simulator repeats total {evidence['runtime_sum_sec']:.1f} s; archive size before report {archive_bytes/1e6:.1f} MB.",
        '',
    ]
    report=OUT/'clock-diagnostic-report.md'
    rendered='\n'.join(lines)
    if report.exists() and report.read_text()!=rendered:raise ValueError('report exists with different content')
    if not report.exists():report.write_text(rendered)
    atomic_json(OUT/'stage-d3-complete.json',{'status':'complete','decision_sha256':sha(OUT/'clock-hypothesis-decision.json'),
        'report_sha256':sha(report),'alignment_sha256':sha(OUT/'clock-alignment.json'),
        'motion_sha256':sha(OUT/'follower-motion-window-trace.json')})
    print('CLOCK_DIAGNOSTIC_FINALIZED',evidence['hypothesis_classification']['H1_clock_domain_mismatch'],evidence['runtime_sum_sec'])

if __name__=='__main__':main()
