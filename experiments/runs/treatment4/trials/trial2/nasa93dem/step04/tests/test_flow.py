import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from source.nasa93_summary import assess_arrivals


class FlowTests(unittest.TestCase):
    def fixture(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write(text)
        handle.close()
        path = Path(handle.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_warmup_and_cumulative_arithmetic(self):
        path = self.fixture("idX,kind,Effort-,Kloc+\n1,a,10,1\n2,b,20,2\n3,a,30,3\n4,c,40,4\n")
        result = assess_arrivals(path)
        self.assertEqual(result["warm_up_count"], 2)
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["rows"][0]["prior_training_count"], 2)
        self.assertAlmostEqual(result["rows"][0]["cumulative_mae"], result["rows"][0]["absolute_error"])
        self.assertAlmostEqual(result["rows"][1]["cumulative_mae"],
                               (result["rows"][0]["absolute_error"] + result["rows"][1]["absolute_error"]) / 2)

    def test_empty_and_short_inputs(self):
        self.assertEqual(assess_arrivals(self.fixture("idX,Effort-\n"))["rows"], [])
        result = assess_arrivals(self.fixture("idX,Effort-\n1,10\n"))
        self.assertEqual(result["warm_up_count"], 1)
        self.assertEqual(result["rows"], [])

    def test_missing_unseen_and_outcome_isolation(self):
        original = self.fixture("idX,kind,Effort-\n1,a,10\n2,b,20\n3,?,30\n4,never,40\n")
        altered = self.fixture("idX,kind,Effort-\n1,a,10\n2,b,20\n3,?,999\n4,never,888\n")
        with self.assertRaises(ValueError):
            assess_arrivals(self.fixture("idX,kind,Effort-\n1,a,10\n2,b,\n"))
        first = assess_arrivals(original)["rows"][0]
        changed = assess_arrivals(altered)["rows"][0]
        self.assertTrue(np.isfinite(first["prediction"]))
        self.assertEqual(first["prediction"], changed["prediction"])
        self.assertEqual(first["prior_training_count"], changed["prior_training_count"])


if __name__ == "__main__":
    unittest.main()
