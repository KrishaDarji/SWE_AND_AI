import tempfile
import unittest
from pathlib import Path

from source.nasa93_summary import compare_tools


class ComparisonTests(unittest.TestCase):
    def fixture(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        handle.write("tool,Effort-\n" + "".join(f"{tool},{value}\n" for tool, value in rows))
        handle.close(); self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return Path(handle.name)

    def test_groups_direction_and_threshold_crossing(self):
        result = compare_tools(self.fixture([("n", 10), ("n", 12), ("n", 14), ("h", 1), ("h", 2), ("h", 3)]), 8)
        self.assertEqual(result["groups"]["n"]["n"], 3)
        self.assertGreater(result["difference_mean_a_minus_mean_b"], 0)
        self.assertTrue(result["point_estimate_exceeds_threshold"])
        self.assertFalse(result["entire_interval_exceeds_threshold"])

    def test_unequal_variance_interval_and_p_value(self):
        result = compare_tools(self.fixture([("n", 1), ("n", 2), ("n", 3), ("n", 20), ("h", 2), ("h", 3), ("h", 4)]))
        self.assertGreater(result["welch_df"], 1)
        self.assertLess(result["confidence_interval"][0], result["difference_mean_a_minus_mean_b"])
        self.assertLessEqual(result["two_sided_p_value"], 1)

    def test_insufficient_and_zero_variance(self):
        with self.assertRaises(ValueError): compare_tools(self.fixture([("n", 1), ("h", 2), ("h", 3)]))
        with self.assertRaises(ValueError): compare_tools(self.fixture([("n", 1), ("n", 1), ("h", 2), ("h", 3)]))

    def test_threshold_validation(self):
        with self.assertRaises(ValueError): compare_tools(self.fixture([("n", 1), ("n", 2), ("h", 3), ("h", 4)]), -1)


if __name__ == "__main__": unittest.main()
