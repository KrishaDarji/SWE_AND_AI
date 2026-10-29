import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).parents[1] / "source"
sys.path.insert(0, str(SOURCE))
from flow import CONFIGURATION_FIELDS, OUTCOME_FIELDS, final_summary, flow_csv, snapshots


def row(**values):
    result = {field: "x" for field in CONFIGURATION_FIELDS}
    result.update({field: "" for field in OUTCOME_FIELDS})
    result.update(values)
    return result


class FlowTests(unittest.TestCase):
    def test_directions_statistics_best_and_ties(self):
        rows = [row(record_id="a", **{"MRE-": "3", "ACC+": "1", "PRED40+": "2", "N_estimators": "1"}),
                row(record_id="b", **{"MRE-": "2", "ACC+": "2", "PRED40+": "2", "N_estimators": "2"})]
        result = list(snapshots(rows))
        final = result[-1]["outcomes"]
        self.assertEqual(final["MRE-"]["best_record_id"], "b")
        self.assertEqual(final["ACC+"]["best_record_id"], "b")
        self.assertEqual(final["PRED40+"]["best_record_id"], "a")
        self.assertAlmostEqual(final["MRE-"]["sample_variance"], .5)
        self.assertEqual(result[0]["outcomes"]["MRE-"]["valid_count"], 1)

    def test_empty_one_missing_nonfinite_and_defensive_copy(self):
        empty = list(snapshots(iter(())))
        self.assertEqual(empty, [])
        first = row(record_id="a", **{"MRE-": "nan", "ACC+": "", "PRED40+": "inf"})
        snaps = list(snapshots(iter([first])))
        self.assertEqual(snaps[0]["outcomes"]["MRE-"]["valid_count"], 0)
        first["N_estimators"] = "changed"
        self.assertIsNone(snaps[0]["outcomes"]["MRE-"]["best_configuration"])

    def test_csv_integration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["MRE-", "ACC+", "PRED40+"] + CONFIGURATION_FIELDS)
                writer.writeheader()
                writer.writerow(row(**{"MRE-": "4", "ACC+": "1", "PRED40+": "0"}))
            output = list(flow_csv(path))
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["outcomes"]["MRE-"]["best_value"], 4.0)

    def test_iterator_consumed_once_and_snapshots_are_independent(self):
        class Once:
            def __init__(self): self.used = False
            def __iter__(self):
                if self.used: raise AssertionError("iterated twice")
                self.used = True
                yield row(record_id="a", **{"MRE-": "1", "ACC+": "2", "PRED40+": "3"})
        result = list(snapshots(Once()))
        result[0]["outcomes"]["MRE-"]["mean"] = 99
        self.assertNotEqual(result[0]["outcomes"]["ACC+"]["mean"], 99)


if __name__ == "__main__":
    unittest.main()
