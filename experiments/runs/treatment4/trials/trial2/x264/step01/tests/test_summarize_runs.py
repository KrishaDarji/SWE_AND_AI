import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from summarize_runs import load_dataset, summarize


class SummarizeRunsTests(unittest.TestCase):
    def setUp(self):
        self.path = Path(tempfile.mkstemp(suffix=".csv")[1])
        self.path.write_text(
            "a,b,Performance-\n1,0,10.0\n0,1,20.0\n1,1,15.0\n", encoding="utf-8"
        )

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_loads_numeric_settings_and_measure(self):
        names, x, y = load_dataset(self.path)
        self.assertEqual(names, ["a", "b"])
        np.testing.assert_array_equal(x, [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(y, [10, 20, 15])

    def test_summary_has_expected_statistics_and_extrema(self):
        result = summarize(self.path)
        self.assertEqual(result["runs"], 3)
        self.assertEqual(result["setting_count"], 2)
        self.assertEqual(result["best_run"]["value"], 10.0)
        self.assertEqual(result["worst_run"]["value"], 20.0)
        self.assertAlmostEqual(result["measure_summary"]["mean"], 15.0)
        self.assertEqual(set(result["setting_on_counts"]), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
