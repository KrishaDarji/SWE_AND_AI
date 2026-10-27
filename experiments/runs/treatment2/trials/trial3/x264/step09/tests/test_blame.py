import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from x264_summary.blame import blame_slow_encodes


class BlameTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", delete=False)
        handle.write(text); handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_ranks_positive_slow_penalty_first(self):
        path = self.write("a,b,time\n0,0,10\n0,1,20\n1,0,30\n1,1,40\n")
        result = blame_slow_encodes(path)
        self.assertEqual(result[0].setting, "a")
        self.assertEqual(result[0].penalty, 20.0)
        self.assertEqual(result[1].penalty, 10.0)

    def test_skips_unobserved_state_and_rejects_nonbinary(self):
        self.assertEqual(len(blame_slow_encodes(self.write("a,time\n1,4\n1,5\n"))), 0)
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            blame_slow_encodes(self.write("a,time\n2,4\n0,5\n"))


if __name__ == "__main__":
    unittest.main()
