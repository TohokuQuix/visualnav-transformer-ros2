"""Offline decoding and receive-clock causality regression tests."""
import importlib.util
from pathlib import Path
import struct
import sys
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/eat_nav_r6_extract_bag.py'


class BagExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('r6_bag_extract', SCRIPT)
        cls.mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.mod
        spec.loader.exec_module(cls.mod)

    def test_cdr_alignment_and_bounds(self):
        # CDR alignment starts after its four-byte encapsulation header.
        payload = b'\x00\x01\x00\x00' + struct.pack('<I', 4) + b'map\x00' + struct.pack('<d', .1)
        reader = self.mod.CDRReader(payload)
        self.assertEqual(reader.string(), 'map')
        self.assertAlmostEqual(reader.scalar('d', 8), .1)
        reader.finish()
        with self.assertRaises(ValueError):
            short = self.mod.CDRReader(payload[:-1])
            short.string()
            short.scalar('d', 8)

    def test_grid_cdr_embedded_schema_and_truncation(self):
        payload = bytearray(b'\x00\x01\x00\x00')
        def scalar(fmt, alignment, value):
            payload.extend(b'\x00' * ((4-len(payload)) % alignment))
            payload.extend(struct.pack('<'+fmt, value))
        def string(value):
            raw = value.encode()+b'\x00'
            scalar('I', 4, len(raw)); payload.extend(raw)
        scalar('i', 4, 123); scalar('I', 4, 456); string('map')
        for value in (.1, 2., 2., 0., 0., 0., 0., 0., 0., 1.): scalar('d', 8, value)
        scalar('I', 4, 1); string('elevation_inpainted')
        scalar('I', 4, 0)  # basic layers
        scalar('I', 4, 1)  # data arrays
        scalar('I', 4, 2)
        for label,size,stride in [('column_index',20,400),('row_index',20,20)]:
            string(label); scalar('I', 4, size); scalar('I', 4, stride)
        scalar('I', 4, 0); scalar('I', 4, 400)
        for i in range(400): scalar('f', 4, float(i))
        scalar('H', 2, 0); scalar('H', 2, 0)
        grid = self.mod.CDRReader(payload).grid()
        self.assertEqual(grid.header.stamp.sec, 123)
        self.assertEqual(grid.header.stamp.nanosec, 456)
        self.assertEqual(grid.layers, ['elevation_inpainted'])
        self.assertEqual(grid.data[0].data[-1], 399.)
        with self.assertRaises(ValueError): self.mod.CDRReader(payload[:-1]).grid()
        with self.assertRaises(ValueError): self.mod.CDRReader(payload+b'x').grid()

    def test_causal_index_uses_receipt_not_nearest_header(self):
        self.assertEqual(self.mod.causal_index([100, 200, 300], 250), 1)
        self.assertIsNone(self.mod.causal_index([100, 200], 99))
        self.assertEqual(self.mod.causal_index([100, 200], 200), 1)

    def test_distinct_history_excludes_future_and_requires_four(self):
        self.assertEqual(self.mod.history_indices([0, 330, 660, 990, 1320], 1000, 333), [0, 1, 2, 3])
        self.assertIsNone(self.mod.history_indices([0, 330, 660, 990], 900, 333))

    def test_goal_is_initial_yaw_relative(self):
        import numpy as np
        q = [0., 0., np.sin(np.pi / 4), np.cos(np.pi / 4)]
        np.testing.assert_allclose(self.mod.initial_goal([1., 2., 0.], q), [1., 4.], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
