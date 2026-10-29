import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from summarizer import summarize_csv

FIELDS = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]


class SummarizerTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.DictWriter(f, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows); f.close()
        self.addCleanup(lambda: Path(f.name).unlink(missing_ok=True)); return f.name

    def test_summaries_roles_and_frequencies(self):
        rows = [{"MRE-":"1", "ACC+":"2", "PRED40+":"3", "N_estimators":"10", "criterion":"a", "Min_sample_leaves":"1", "Min_impurity_decrease":"0.1", "Max_depth":"2"}, {"MRE-":"2", "ACC+":"3", "PRED40+":"4", "N_estimators":"10", "criterion":"a", "Min_sample_leaves":"1", "Min_impurity_decrease":"0.1", "Max_depth":"2"}]
        result = summarize_csv(self.write(rows))
        self.assertEqual(result["record_count"], 2); self.assertEqual(result["numeric_statistics"]["MRE-"]["mean"], 1.5)
        self.assertEqual(result["categorical_frequencies"]["criterion"], {"a": 2}); self.assertEqual(result["configuration_fields"], ["N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"])
        self.assertEqual(result["outcome_fields"], ["MRE-", "ACC+", "PRED40+"]); self.assertEqual(result["missing_values"]["MRE-"], 0)

    def test_missing_duplicates_and_repeated_configurations(self):
        row = {"MRE-":"", "ACC+":"2", "PRED40+":"3", "N_estimators":"10", "criterion":"a", "Min_sample_leaves":"1", "Min_impurity_decrease":"0.1", "Max_depth":"2"}
        result = summarize_csv(self.write([row, row]))
        self.assertEqual(result["missing_values"]["MRE-"], 2); self.assertEqual(result["duplicate_row_count"], 1)
        self.assertEqual(len(result["repeated_configurations"]), 1); self.assertEqual(next(iter(result["repeated_configurations"].values())), 2)

    def test_empty_input(self):
        result = summarize_csv(self.write([]))
        self.assertEqual(result["record_count"], 0); self.assertEqual(result["distinct_configuration_count"], 0)


if __name__ == "__main__": unittest.main()
