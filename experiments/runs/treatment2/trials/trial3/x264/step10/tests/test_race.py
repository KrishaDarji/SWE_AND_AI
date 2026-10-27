import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.race import compare_searches


class RaceTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_equal_budget_and_same_objective(self):
        path = self.write("mode,time\na,10\nb,4\nc,8\nd,2\n")
        result = compare_searches(path, budget=2, seed=1)
        self.assertEqual(result.objective_measure, "time")
        self.assertEqual(result.system.evaluations, result.baseline.evaluations)
        self.assertEqual(result.system.objective, 4.0)

    def test_seed_makes_baseline_reproducible_and_validates_budget(self):
        path = self.write("mode,time\na,10\nb,4\nc,8\n")
        first = compare_searches(path, budget=20, seed=7)
        second = compare_searches(path, budget=20, seed=7)
        self.assertEqual(first, second)
        with self.assertRaisesRegex(ValueError, "positive"):
            compare_searches(path, budget=0)


if __name__ == "__main__":
    unittest.main()
