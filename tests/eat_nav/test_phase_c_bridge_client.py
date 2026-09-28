import pytest
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient


def test_bridge_requires_live_server():
    with pytest.raises(OSError):
        FollowerBridgeClient(port=1, timeout=0.1)
