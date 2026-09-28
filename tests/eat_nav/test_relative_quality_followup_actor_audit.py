"""Actor provenance rejects incomplete or absent observation vectors."""
import unittest
from visualnav_transformer.eat_nav.relative_quality_followup_actor import actor_observation_status


class ActorAuditTest(unittest.TestCase):
    def test_absent_obs_cannot_be_replaced_by_twist_and_action(self):
        row={'twist':[0.]*6,'action':[0.]*6,'command':[0.]*3}
        self.assertEqual(actor_observation_status(row,22),'missing')

    def test_complete_finite_vector_is_available(self):
        row={'actor_observation':[0.]*22}
        self.assertEqual(actor_observation_status(row,22),'available')
        row['actor_observation'][3]=float('nan')
        self.assertEqual(actor_observation_status(row,22),'invalid')
        row['actor_observation']=[0.]*21
        self.assertEqual(actor_observation_status(row,22),'invalid')

if __name__=='__main__':unittest.main()
