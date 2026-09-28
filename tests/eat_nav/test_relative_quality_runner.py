"""Detached stage markers must reject stale inputs and output corruption."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE = Path(__file__).resolve().parents[2] / 'scripts/eat_nav_relative_quality_runner.py'
spec = importlib.util.spec_from_file_location('relative_runner', MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerMarkerTest(unittest.TestCase):
    def test_resume_only_with_same_input_and_output_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            old_root, old_docs = runner.ROOT, runner.DOCS
            try:
                runner.ROOT = Path(temp) / 'run'
                runner.DOCS = Path(temp) / 'docs'
                runner.DOCS.mkdir()
                f = runner.DOCS / 'result.json'
                f.write_text('{"ok":true}\n')
                marker = runner.ROOT / 'markers' / 'one.json'
                runner.write_json(marker, {'inputs': {'a':'1'},
                                           'outputs': {'result.json':runner.digest(f)}})
                self.assertEqual(runner.stage_done('one',['result.json'],{'a':'1'}),runner.digest(marker))
                with self.assertRaisesRegex(RuntimeError,'inputs changed'):
                    runner.stage_done('one',['result.json'],{'a':'2'})
                f.write_text('{"ok":false}\n')
                with self.assertRaisesRegex(RuntimeError,'output SHA mismatch'):
                    runner.stage_done('one',['result.json'],{'a':'1'})
            finally:
                runner.ROOT, runner.DOCS = old_root, old_docs


if __name__ == '__main__':
    unittest.main()
