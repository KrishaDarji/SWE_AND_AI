import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.choose import choose_configuration


class ChoiceTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_selects_fastest_feasible_configuration(self):
        path = self.write("mode,quality,time\na,90,10\na,92,12\nb,80,4\nb,82,6\nc,95,20\n")
        result = choose_configuration(path, "quality", minimum_quality=90, objective_measure="time")
        self.assertEqual(result.configuration, {"mode": "a"})
        self.assertEqual(result.quality_mean, 91.0)
        self.assertEqual(result.feasible_configurations, 2)

    def test_supports_upper_bound_and_rejects_empty_set(self):
        path = self.write("mode,quality,time\na,2,10\nb,8,4\n")
        result = choose_configuration(path, "quality", maximum_quality=3, objective_measure="time")
        self.assertEqual(result.configuration, {"mode": "a"})
        with self.assertRaisesRegex(ValueError, "no configuration"):
            choose_configuration(path, "quality", minimum_quality=9, objective_measure="time")


if __name__ == "__main__":
    unittest.main()
