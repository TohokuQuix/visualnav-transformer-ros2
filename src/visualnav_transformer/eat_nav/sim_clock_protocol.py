"""Pure step/stamp guards for the simulation-only ROS clock bridge."""
from __future__ import annotations
from dataclasses import dataclass, field

STEP_NS = 50_000_000


@dataclass
class ClockProtocol:
    stamp_ns: int | None = None
    path_stamp_ns: int | None = None
    tf_stamp_ns: int | None = None
    commands: dict[int, list[tuple[float, float, float]]] = field(default_factory=dict)

    def advance(self, stamp_ns: int, *, reset: bool = False) -> None:
        if stamp_ns < 0:
            raise ValueError('negative simulation time')
        if self.stamp_ns is not None and stamp_ns <= self.stamp_ns:
            raise ValueError('simulation clock did not increase')
        if not reset and self.stamp_ns is not None and stamp_ns-self.stamp_ns != STEP_NS:
            raise ValueError('simulation clock jump')
        self.stamp_ns = stamp_ns
        self.tf_stamp_ns = None
        if reset:
            self.path_stamp_ns = None
        self.commands = {}

    def set_tf(self, stamp_ns: int) -> None:
        if stamp_ns != self.stamp_ns:
            raise ValueError('TF stamp mismatch')
        self.tf_stamp_ns = stamp_ns

    def set_path(self, stamp_ns: int) -> None:
        if stamp_ns != self.stamp_ns or self.tf_stamp_ns != stamp_ns:
            raise ValueError('Path/TF stamp mismatch')
        self.path_stamp_ns = stamp_ns

    def accept(self, branch: int, stamp_ns: int, command: tuple[float, float, float]) -> bool:
        if (stamp_ns != self.stamp_ns or self.tf_stamp_ns != stamp_ns
                or self.path_stamp_ns is None or self.path_stamp_ns > stamp_ns):
            return False
        previous = self.commands.setdefault(branch, [])
        if previous and any(abs(a-b)>1e-9 for a,b in zip(previous[0], command)):
            raise ValueError(f'different commands at same stamp {stamp_ns} branch {branch}')
        previous.append(command)
        return True

    def ready(self, branches: int) -> bool:
        return all(self.commands.get(i) for i in range(branches))

    def motion_elapsed_seconds(self, first_step: int, last_step: int) -> float:
        return (last_step-first_step)*STEP_NS/1e9
