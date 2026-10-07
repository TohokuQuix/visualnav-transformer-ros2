"""Pure validation of pre-reset physical termination evidence for radial inventory."""
from __future__ import annotations

import math


def _finite_tree(value):
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tree(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def build_reset_events(*, selected: dict[int, str], physics_step: int,
                       done_ids: list[int], terminated_ids: list[int], truncated_ids: list[int],
                       term_flags: dict[str, list[int]], timeout_terms: set[str],
                       pre_states: dict[int, dict], actor_values: dict[int, list[float]],
                       episode_lengths: dict[int, int], reset_counts: dict[int, int],
                       trace_steps: dict[int, int], dem_frame_ids: dict[int, list[str]],
                       contacts: dict[int, dict]) -> list[dict]:
    """Classify selected resets from flags captured before Onix `_reset_idx`.

    This function never uses post-reset observations. Unknown/timeout selected
    resets fail closed, while non-selected resets do not close selected streams.
    """
    done = set(done_ids)
    if len(done) != len(done_ids) or done != set(terminated_ids) | set(truncated_ids):
        raise ValueError('unknown termination: done/terminated/truncated mismatch')
    flagged = {i for ids in term_flags.values() for i in ids}
    if flagged != done:
        raise ValueError('unknown termination: term flags and done mismatch')
    if not 0 < physics_step <= 200 or physics_step % 10:
        raise ValueError('termination step is not a control boundary')
    events = []
    for env_id in done_ids:
        if env_id not in selected:
            continue
        terms = sorted(name for name, ids in term_flags.items() if env_id in ids)
        if env_id in truncated_ids or any(name in timeout_terms for name in terms):
            raise ValueError('timeout/truncated selected pose is not a physical termination')
        if env_id not in terminated_ids or not terms:
            raise ValueError('unknown selected termination')
        try:
            state = pre_states[env_id]
            actor = actor_values[env_id]
            episode_length = episode_lengths[env_id]
            reset_count = reset_counts[env_id]
            trace_step = trace_steps[env_id]
            frames = dem_frame_ids[env_id]
            contact = contacts[env_id]
        except KeyError as exc:
            raise ValueError('missing pre-reset evidence') from exc
        if (trace_step != physics_step or episode_length != physics_step // 10 or
            not isinstance(reset_count, int) or reset_count < 0 or
            not isinstance(actor, list) or len(actor) != 22 or
            not _finite_tree(state) or not _finite_tree(actor) or
            contact.get('bounds_valid') is not True or contact.get('finite') is not True or
            contact.get('buffer_full_or_truncation_suspected') is not False or
            contact.get('buffer_capacity') != 4096):
            raise ValueError('pre-reset state, Actor, contact, step, or counter invalid')
        events.append({
            'pose_id': selected[env_id], 'env_id': env_id,
            'episode_id': reset_count, 'reset_counter_before': reset_count,
            'reset_counter_after': reset_count + 1,
            'terminated': True, 'truncated': False, 'done': True,
            'term_flags': {name: env_id in ids for name, ids in term_flags.items()},
            'termination_reason': terms, 'termination_step': physics_step,
            'logical_timestamp_ns': physics_step * 5_000_000,
            'last_valid_step': trace_step, 'post_reset_state': False,
            'pre_reset_state': state, 'pre_reset_actor': actor,
            'pre_reset_contact': contact, 'dem_frame_ids': list(frames),
            'episode_length_at_termination': episode_length,
        })
    return events


def verify_pose_coverage(expected_pose_ids: list[str], results: list[dict]) -> None:
    """Require exact one-to-one coverage of the frozen pose manifest."""
    actual = [row['pose_id'] for row in results]
    if (len(expected_pose_ids) != len(set(expected_pose_ids)) or
        len(actual) != len(expected_pose_ids) or len(set(actual)) != len(actual) or
        set(actual) != set(expected_pose_ids)):
        raise ValueError('duplicate, missing, or extra pose result')
