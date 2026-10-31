import json
import math
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from spot import assess, reference, screen

class SpotTests(unittest.TestCase):
    def test_ordinary_and_unusual(self):
        refs = reference([1.0, 1.1, 1.2, 1.1])
        score, flagged, reason = assess(10.0, refs)
        self.assertTrue(flagged); self.assertGreater(score, 3.5); self.assertIn("flagged", reason)
    def test_missing_nonfinite_and_json(self):
        score, flagged, reason = assess(float("nan"), reference([1.0, 2.0]))
        self.assertIsNone(score); self.assertFalse(flagged); self.assertIn("undefined", reason)
        self.assertEqual(json.loads(json.dumps({"score": score}, allow_nan=False))["score"], None)
    def test_small_empty_constants_zero_scale(self):
        self.assertIsNone(reference([])["median"])
        self.assertEqual(assess(2.0, reference([2.0]))[0], 0.0)
        self.assertIsNone(assess(3.0, reference([2.0]))[0])
        self.assertEqual(reference([4.0, 4.0])["scale"], 0.0)
    def test_identifiers_and_counts(self):
        fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("MRE-,ACC+,PRED40+\n1,1,1\n1.1,1.1,1.1\n20,1.2,1.2\n")
            result = screen(path)
            self.assertEqual(result["flagged_record_ids"], ["3"])
            self.assertEqual(sum(x["flag_count"] for x in result["outcomes"]), 1)
            self.assertEqual(result["outcomes"][0]["records"][2]["record_id"], "3")
        finally:
            os.unlink(path)

if __name__ == "__main__": unittest.main()
