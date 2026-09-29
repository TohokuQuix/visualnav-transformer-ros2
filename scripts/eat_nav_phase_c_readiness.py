"""Pure predicates for bounded follower startup and command freshness."""
from __future__ import annotations


def clock_progressed(samples_ns):
    return len(samples_ns) >= 3 and all(b > a for a,b in zip(samples_ns,samples_ns[1:]))


def prepath_ready(*, alive, subscribers, status_counts, tf_ready, clock_samples_ns):
    n=len(alive)
    if not n or not all(len(x)==n for x in (subscribers,status_counts,tf_ready)):
        return False
    return (all(alive) and all(x>0 for x in subscribers)
            and all(x>=2 for x in status_counts) and all(tf_ready)
            and clock_progressed(clock_samples_ns))


def command_delivery(before, after, commands):
    if len(before)!=len(after) or len(after)!=len(commands):
        raise ValueError('branch count mismatch')
    fresh=[a>b and cmd is not None for b,a,cmd in zip(before,after,commands)]
    if all(fresh):return 'fresh'
    if any(fresh):return 'partial'
    if any(a>b for b,a in zip(before,after)):return 'message_without_command'
    return 'stale'


def current_command_stamp(command_stamp_ns, path_stamp_ns):
    return path_stamp_ns is not None and command_stamp_ns >= path_stamp_ns
