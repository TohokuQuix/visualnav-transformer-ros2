import unittest

from visualnav_transformer.eat_nav.ros2_phase2.supervisor import Supervisor
from visualnav_transformer.eat_nav.ros2_phase2.gateway import MockFollowerGateway


MS = 1_000_000


class SafetyTest(unittest.TestCase):
    def test_jitter_defer_then_sensor_outage_stop(self):
        s = Supervisor()
        g = MockFollowerGateway()
        gen = s.start_goal(0, timeout_ns=10_000*MS)
        s.observe_dem(100*MS, 100*MS)
        s.accept_path(gen, 100*MS)
        g.activate(gen, 100*MS, 600*MS)
        self.assertEqual(s.tick(350*MS, g.follower_healthy(350*MS)),'DEFER')
        self.assertEqual(s.tick(550*MS, g.follower_healthy(550*MS)),'DEFER')
        self.assertEqual(s.tick(601*MS, g.follower_healthy(601*MS)),'STOP')
        g.invalidate(gen)
        self.assertEqual(g.tick(601*MS, (.4, .2)),(0.,0.))

    def test_duplicate_dem_does_not_renew_lease(self):
        s = Supervisor(); gen=s.start_goal(0,timeout_ns=5000*MS)
        s.observe_dem(100*MS,100*MS)
        s.accept_path(gen,100*MS)
        with self.assertRaises(ValueError):
            s.observe_dem(200*MS,100*MS)
        self.assertEqual(s.tick(601*MS,True),'STOP')

    def test_cancel_timeout_and_old_path(self):
        s=Supervisor(); g=MockFollowerGateway()
        gen=s.start_goal(0,timeout_ns=1000*MS)
        s.observe_dem(0,0); s.accept_path(gen,0)
        g.activate(gen,0,500*MS)
        self.assertEqual(g.tick(50*MS,(.4,0)),(.4,0.))
        s.cancel(gen); g.invalidate(gen)
        self.assertEqual(g.tick(100*MS,(.4,0)),(0.,0.))
        with self.assertRaises(ValueError):
            g.activate(gen,100*MS,600*MS)
        gen2=s.start_goal(200*MS,timeout_ns=100*MS)
        self.assertNotEqual(gen,gen2)
        self.assertEqual(s.tick(301*MS,True),'STOP')

    def test_heartbeat_follower_command_age_and_generation(self):
        g=MockFollowerGateway()
        g.activate(1,0,500*MS)
        self.assertEqual(g.tick(60*MS,(.2,0),command_stamp_ns=0),(0.,0.))
        g.activate(2,100*MS,600*MS)
        self.assertEqual(g.tick(120*MS,(.2,0),command_stamp_ns=120*MS),(.2,0.))
        self.assertEqual(g.tick(230*MS,(.2,0),command_stamp_ns=230*MS,
                                supervisor_heartbeat_ns=100*MS),(0.,0.))
        self.assertEqual(g.tick(240*MS,(float('nan'),0),command_stamp_ns=240*MS),
                         (0.,0.))


if __name__ == '__main__': unittest.main()
