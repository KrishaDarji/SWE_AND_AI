import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.certify import certify_improvement


class CertificationTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_lower_candidate_is_credible(self):
        path = self.write("mode,time\nbase,100\nbase,101\nbase,99\nnew,80\nnew,81\nnew,79\n")
        result = certify_improvement(path, {"mode": "base"}, {"mode": "new"})
        self.assertTrue(result.credible)
        self.assertGreater(result.improvement, 0)
        self.assertLess(result.p_value, 0.05)

    def test_single_observation_is_not_credible(self):
        path = self.write("mode,time\nbase,10\nnew,5\n")
        self.assertFalse(certify_improvement(path, {"mode": "base"}, {"mode": "new"}).credible)

    def test_missing_configuration_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "match"):
            certify_improvement(self.write("mode,time\nbase,10\n"), {"mode": "base"}, {"mode": "new"})


if __name__ == "__main__":
    unittest.main()
