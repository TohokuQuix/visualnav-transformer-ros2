"""P1 diagnostic bridge: attach exact follower branch records to accepted commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from eat_nav_command_delivery_bridge import DiagnosticServer
from visualnav_transformer.eat_nav.p1_follower_diagnostic import find_record


class PairedServer(DiagnosticServer):
    def wait_command(self, ns, timeout):
        response = super().wait_command(ns, timeout)
        response['follower_diagnostics'] = [
            find_record(self.log_dir / f'reset-{self.reset_number:04d}-branch-{index}.log',
                        ns, command)
            for index, command in enumerate(response['commands'])
        ]
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=47819)
    parser.add_argument('--branches', type=int, default=5)
    parser.add_argument('--follower', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--fault-drop-stamp-ns', type=int)
    PairedServer(parser.parse_args()).serve()


if __name__ == '__main__':
    main()
