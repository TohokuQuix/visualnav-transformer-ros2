"""Bounded forward-command intervention at the P1 command-manager boundary."""

from __future__ import annotations

import math


CONTROLLED = frozenset(('plus_controlled', 'minus_controlled'))


def applied_commands(normal, conditions, reference_forward, *, hard_limit=0.3):
    if len(normal) != len(conditions) or not math.isfinite(reference_forward) or not 0 <= reference_forward <= hard_limit:
        raise ValueError('invalid or out-of-limit reference schedule')
    result=[]
    for command, condition in zip(normal,conditions):
        if len(command)!=3 or not all(math.isfinite(v) for v in command):
            raise ValueError('invalid follower command')
        if condition not in CONTROLLED | {'direct','plus_normal','minus_normal'}:
            raise ValueError('unknown intervention condition')
        applied=list(command)
        # Preserve stop, turn-in-place, goal, and reverse commands. The
        # intervention only replaces positive final forward speed.
        if condition in CONTROLLED and applied[0]>0:
            applied[0]=reference_forward
        result.append(applied)
    return result
