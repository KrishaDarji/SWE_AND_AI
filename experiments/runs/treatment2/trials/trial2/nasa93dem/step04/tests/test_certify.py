import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/Users/RadhaKrishna/Desktop/SWE_AND_AI/moot/optimize/process/nasa93dem.csv")
sys.path.insert(0, str(ROOT / "source"))
from project_summary import certify_less_costly, load_projects


class CertifyTests(unittest.TestCase):
    def test_strongly_cheaper_group_is_certified(self):
        projects = load_projects(str(DATA))
        a = [p for p in projects if p.year >= 1984]
        b = [p for p in projects if p.year <= 1979]
        result = certify_less_costly(a, b, "late", "early", resamples=500)
        self.assertTrue(result.genuinely_less_costly)
        self.assertLess(result.difference_mean_a_minus_b, 0)

    def test_inconclusive_identical_groups_are_not_certified(self):
        projects = load_projects(str(DATA))
        result = certify_less_costly(projects[:20], projects[:20], resamples=300)
        self.assertFalse(result.genuinely_less_costly)

    def test_cli_selects_groups_and_emits_json(self):
        output = subprocess.check_output(
            [sys.executable, str(ROOT / "source" / "certify.py"), str(DATA),
             "sced", "l", "n", "--resamples", "300"], text=True)
        parsed = json.loads(output)
        self.assertIn("genuinely_less_costly", parsed)


if __name__ == "__main__":
    unittest.main()
