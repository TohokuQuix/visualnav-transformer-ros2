"""Mock-only follower Path gateway and final command safety gate.

This module has no publisher for production cmd_vel or production follower topics.
"""

import math


class MockFollowerGateway:
    COMMAND_MAX_AGE_NS = 50_000_000
    HEARTBEAT_MAX_AGE_NS = 100_000_000

    def __init__(self):
        self.max_seen_generation = 0
        self.active_generation = None
        self.lease_deadline_ns = None
        self.last_output = (0.,0.)
        self.invalidation_ack = True
        self.events = []
        self.last_activation_stamp_ns = -1

    def activate(self, generation, stamp_ns, expires_ns):
        if generation < self.max_seen_generation or \
                (generation == self.max_seen_generation and
                 (self.active_generation != generation or
                  stamp_ns <= self.last_activation_stamp_ns)) or \
                expires_ns <= stamp_ns or \
                expires_ns-stamp_ns > 500_000_000:
            raise ValueError('old generation or invalid Path lease')
        self.max_seen_generation = generation
        self.active_generation = generation
        self.lease_deadline_ns = expires_ns
        self.last_activation_stamp_ns = stamp_ns
        self.invalidation_ack = False
        self.events.append(('path',generation,stamp_ns))

    def invalidate(self, generation):
        if generation != self.active_generation:
            raise ValueError('Path invalidation generation mismatch')
        self.max_seen_generation = max(self.max_seen_generation,generation)
        self.active_generation = None
        self.lease_deadline_ns = None
        self.last_output = (0.,0.)
        self.invalidation_ack = True
        self.events.append(('invalidated',generation))

    def follower_healthy(self, now_ns):
        return self.active_generation is not None and \
            now_ns <= self.lease_deadline_ns

    def tick(self, now_ns, follower_command, *, command_stamp_ns=None,
             supervisor_heartbeat_ns=None):
        stamp = now_ns if command_stamp_ns is None else command_stamp_ns
        heart = now_ns if supervisor_heartbeat_ns is None else supervisor_heartbeat_ns
        command = tuple(float(x) for x in follower_command)
        if len(command) != 2 or not all(map(math.isfinite,command)):
            self.last_output=(0.,0.)
            self.events.append(('invalid_follower_command',now_ns))
            return self.last_output
        allowed = (self.follower_healthy(now_ns) and
                   0 <= now_ns-stamp <= self.COMMAND_MAX_AGE_NS and
                   0 <= now_ns-heart <= self.HEARTBEAT_MAX_AGE_NS)
        self.last_output = command if allowed else (0.,0.)
        self.events.append(('final_command',now_ns,self.last_output))
        return self.last_output
