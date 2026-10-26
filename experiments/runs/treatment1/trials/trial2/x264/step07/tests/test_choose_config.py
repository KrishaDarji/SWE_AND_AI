import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source" / "choose_config.py"


class ChooseConfigTests(unittest.TestCase):
    def write_csv(self, text):
        handle = tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", delete=False)
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_selects_lowest_feasible_configuration(self):
        path = self.write_csv("a,Performance-\n1,10\n0,3\n1,7\n")
        result = subprocess.run([sys.executable, str(SOURCE), path, "--max-performance", "7"],
                                check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["best"]["a"], "0")
        self.assertEqual(payload["feasible_count"], 2)

    def test_default_constraint_is_median(self):
        path = self.write_csv("Performance-\n1\n4\n9\n")
        result = subprocess.run([sys.executable, str(SOURCE), path],
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)["constraint"], 4.0)

    def test_rejects_constraint_with_no_feasible_rows(self):
        path = self.write_csv("Performance-\n2\n3\n")
        result = subprocess.run([sys.executable, str(SOURCE), path, "--max-performance", "1"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
