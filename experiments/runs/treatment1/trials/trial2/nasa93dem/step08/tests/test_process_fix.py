import csv
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
import process_fix


class ProcessFixTests(unittest.TestCase):
    def test_real_dataset_has_recommendation(self):
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../../../moot/optimize/process/nasa93dem.csv"))
        result = process_fix.recommend(process_fix.read_rows(path))
        self.assertIn(result["factor"], process_fix.TARGETS)
        self.assertGreater(result["estimated_risk_reduction"], 0)

    def test_missing_dataset_is_rejected(self):
        with self.assertRaises(ValueError):
            process_fix.read_rows(self._tiny_path([{"Kloc+": "1", "Defects-": "1"}]))

    def _tiny_path(self, rows):
        fd, path = tempfile.mkstemp(text=True)
        os.close(fd)
        fields = list(rows[0])
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
        self.addCleanup(lambda: os.unlink(path))
        return path


if __name__ == "__main__":
    unittest.main()
