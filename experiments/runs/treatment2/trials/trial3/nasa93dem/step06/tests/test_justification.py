import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))

from project_history.justification import justify_estimate, justification_lines
from project_history.model import Project


class JustificationTests(unittest.TestCase):
    def test_justification_reuses_median_estimate_and_shows_spread(self):
        projects = [
            Project("a", 2000, 10.0, 20.0, None, None, {}),
            Project("b", 2001, 10.0, 40.0, None, None, {}),
            Project("c", 2002, 10.0, 60.0, None, None, {}),
            Project("missing", 2003, None, 99.0, None, None, {}),
        ]
        result = justify_estimate(projects, 10.0)
        self.assertEqual(result["benchmark_count"], 3)
        self.assertEqual(result["estimate"], 40.0)
        self.assertEqual(result["lower_effort"], 20.0)
        self.assertEqual(result["upper_effort"], 40.0)
        self.assertIn("historical spread", justification_lines(result)[-1])

    def test_justification_handles_no_usable_history(self):
        result = justify_estimate([Project("x", None, None, None, None, None, {})], 5)
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["confidence"], "not supportable")
        self.assertIn("unavailable", justification_lines(result)[0])

    def test_justification_rejects_non_positive_size(self):
        with self.assertRaises(ValueError):
            justify_estimate([], 0)


if __name__ == "__main__":
    unittest.main()
