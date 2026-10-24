import csv
import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from health_closedissues import main, summarize


class SummaryTests(unittest.TestCase):
    def make_csv(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_metrics_and_best_direction(self):
        path = self.make_csv("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n0.4,0.7,0.5,10,squared_error,2,0.1,3\n0.2,0.6,0.8,20,absolute_error,1,0.2,5\n")
        result = summarize(path)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["best"]["MRE-"]["value"], 0.2)
        self.assertEqual(result["best"]["PRED40+"]["configuration"]["N_estimators"], "20")
        self.assertEqual(result["configurations"]["criterion"]["unique"], 2)

    def test_cli_emits_json(self):
        path = self.make_csv("MRE-,ACC+,PRED40+,N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth\n0.4,0.7,0.5,10,squared_error,2,0.1,3\n")
        self.assertEqual(main([path]), 0)


if __name__ == "__main__":
    unittest.main()
