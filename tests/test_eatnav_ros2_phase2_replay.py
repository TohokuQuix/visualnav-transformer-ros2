import unittest
from pathlib import Path
import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.replay import load_sealed_prepath
from visualnav_transformer.eat_nav.ros2_phase2.wire import dem_from_message, actor_from_message


FIXTURE = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/motion-14-collection-20261014-02/0100-pyramid_stairs-level2-seed1653838991-front_folded-forward_+30')


class ReplayTest(unittest.TestCase):
    def test_sealed_prepath_transport_roundtrip(self):
        if not FIXTURE.exists():
            self.skipTest('sealed sample unavailable')
        replay=load_sealed_prepath(FIXTURE, env_id=0, episode_id=1)
        self.assertEqual(len(replay['dem_messages']),4)
        dem=np.stack([dem_from_message(msg)[0] for msg in replay['dem_messages']])
        state=actor_from_message(replay['actor_message'])[0]
        np.testing.assert_array_equal(dem,replay['dem_original'])
        np.testing.assert_array_equal(state,replay['actor_original'])
        self.assertEqual(replay['dem_sha256'],replay['ledger_dem_sha256'])
        self.assertEqual(replay['root_tile_local'].shape,(13,))


if __name__ == '__main__': unittest.main()
