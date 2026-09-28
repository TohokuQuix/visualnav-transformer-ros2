"""Synchronous client for the real ROS2 follower branch bridge."""
from __future__ import annotations

import json
import socket


class FollowerBridgeClient:
    def __init__(self, host='127.0.0.1', port=47819, timeout=2.0):
        self.connection = socket.create_connection((host, port), timeout=timeout)
        self.connection.settimeout(timeout)
        self.stream = self.connection.makefile('rwb')

    def request(self, action, **fields):
        self.stream.write((json.dumps({'action': action, **fields})+'\n').encode())
        self.stream.flush()
        response = json.loads(self.stream.readline())
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'bridge request failed'))
        return response

    def close(self):
        self.stream.close()
        self.connection.close()
