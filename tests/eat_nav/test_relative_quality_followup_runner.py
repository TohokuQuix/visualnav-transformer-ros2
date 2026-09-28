"""Follow-up stage markers reject changed config, source inputs, and outputs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE=Path(__file__).resolve().parents[2]/'scripts/eat_nav_relative_followup_runner.py'
spec=importlib.util.spec_from_file_location('followup_runner',MODULE)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class MarkerTest(unittest.TestCase):
    def test_stage_marker_detects_corrupt_output_or_config(self):
        old=(runner.ROOT,runner.DOCS,runner.CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            try:
                root=Path(temp);runner.ROOT=root;runner.DOCS=root/'docs';runner.DOCS.mkdir()
                runner.CONFIG=runner.DOCS/'config.yaml';runner.CONFIG.write_text('seed: 11\n')
                result=runner.DOCS/'result.json';result.write_text('{"passed":true}\n')
                expected={'file_sha256':{'config':runner.sha(runner.CONFIG)},'previous_marker_sha256':None}
                marker=root/'markers/01.json'
                runner.write(marker,{'inputs':expected,'config_sha256':runner.sha(runner.CONFIG),
                                     'outputs':{'result.json':runner.sha(result)}})
                self.assertEqual(runner.check_marker('01',['result.json'],expected),runner.sha(marker))
                result.write_text('{"passed":false}\n')
                with self.assertRaisesRegex(ValueError,'output SHA'):
                    runner.check_marker('01',['result.json'],expected)
                result.write_text('{"passed":true}\n');runner.CONFIG.write_text('seed: 12\n')
                with self.assertRaisesRegex(ValueError,'input/config SHA'):
                    runner.check_marker('01',['result.json'],expected)
            finally:runner.ROOT,runner.DOCS,runner.CONFIG=old

if __name__=='__main__':unittest.main()
