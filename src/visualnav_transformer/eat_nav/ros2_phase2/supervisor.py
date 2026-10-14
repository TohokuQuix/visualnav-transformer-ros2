"""Goal generation and bounded DEM/Path lease policy."""


class Supervisor:
    SENSOR_FRESH_NS = 300_000_000
    SENSOR_STOP_NS = 500_000_000
    PATH_LEASE_NS = 500_000_000

    def __init__(self):
        self.generation = 0
        self.state = 'IDLE'
        self.goal_deadline_ns = None
        self.last_sensor_ns = None
        self.last_dem_stamp_ns = None
        self.path_deadline_ns = None
        self.new_input_ready = False

    def start_goal(self, now_ns, *, timeout_ns):
        if timeout_ns <= 0:
            raise ValueError('goal timeout')
        self.generation += 1
        self.state = 'WAITING_INPUT'
        self.goal_deadline_ns = now_ns + timeout_ns
        self.last_sensor_ns = None
        self.last_dem_stamp_ns = None
        self.path_deadline_ns = None
        self.new_input_ready = False
        return self.generation

    def observe_dem(self, stamp_ns, sensor_stamp_ns):
        if self.state not in ('WAITING_INPUT','ACTIVE') or sensor_stamp_ns > stamp_ns or \
                (self.last_sensor_ns is not None and sensor_stamp_ns <= self.last_sensor_ns) or \
                (self.last_dem_stamp_ns is not None and stamp_ns <= self.last_dem_stamp_ns):
            raise ValueError('stale/reversed DEM source')
        self.last_dem_stamp_ns = stamp_ns
        self.last_sensor_ns = sensor_stamp_ns
        self.new_input_ready = True

    def accept_path(self, generation, now_ns):
        if generation != self.generation or self.state not in ('WAITING_INPUT','ACTIVE') or \
                self.last_sensor_ns is None or now_ns-self.last_sensor_ns > self.SENSOR_FRESH_NS:
            raise ValueError('stale goal or inference')
        self.path_deadline_ns = now_ns+self.PATH_LEASE_NS
        self.state='ACTIVE'
        self.new_input_ready = False

    def cancel(self, generation):
        if generation != self.generation:
            raise ValueError('stale cancel generation')
        self.state='CANCELLED'
        self.path_deadline_ns=None
        self.generation += 1  # invalidate queued prediction/old transient Path

    def tick(self, now_ns, follower_healthy):
        if self.state not in ('WAITING_INPUT','ACTIVE'):
            return 'STOP'
        if now_ns > self.goal_deadline_ns:
            self.state='TIMEOUT'
        elif not follower_healthy and self.state == 'ACTIVE':
            self.state='FOLLOWER_LOST'
        elif self.state == 'ACTIVE' and (self.last_sensor_ns is None or
              now_ns-self.last_sensor_ns > self.SENSOR_STOP_NS or
              self.path_deadline_ns is None or now_ns > self.path_deadline_ns):
            self.state='STALE_INPUT'
        if self.state not in ('WAITING_INPUT','ACTIVE'):
            self.path_deadline_ns=None
            self.generation += 1
            return 'STOP'
        if self.state == 'WAITING_INPUT':
            return 'DEFER'
        return 'REPLAN' if self.new_input_ready and \
            now_ns-self.last_sensor_ns <= self.SENSOR_FRESH_NS else 'DEFER'
