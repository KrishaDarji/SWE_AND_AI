import math
import sys
import csv
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
from flow import running_summaries


def rec(i, **values):
    return {"record_id": f"r{i}", "configuration": {"x": i}, **values}


class FlowTests(unittest.TestCase):
    def test_directions_batch_stats_and_best(self):
        rows = [rec(1, **{"MRE-": "4", "ACC+": "0.2", "PRED40+": "0.3"}), rec(2, **{"MRE-": "2", "ACC+": "0.8", "PRED40+": "0.3"})]
        got = running_summaries(rows)[-1]["summaries"]
        self.assertEqual(got["MRE-"]["best_record_id"], "r2")
        self.assertEqual(got["ACC+"]["best_record_id"], "r2")
        self.assertEqual(got["PRED40+"]["best_record_id"], "r1")
        self.assertAlmostEqual(got["MRE-"]["mean"], 3); self.assertAlmostEqual(got["MRE-"]["sample_variance"], 2)

    def test_empty_and_one_observation(self):
        self.assertEqual(running_summaries([]), [])
        state = running_summaries([rec(1, **{"MRE-": "1"})])[0]["summaries"]["MRE-"]
        self.assertEqual(state["valid_count"], 1); self.assertIsNone(state["sample_variance"])

    def test_missing_nonfinite_ties_and_prefix_immutability(self):
        rows = [rec(1, **{"MRE-": "nan", "ACC+": "1"}), rec(2, **{"MRE-": "2", "ACC+": "1"}), rec(3, **{"MRE-": "1", "ACC+": "inf"})]
        snaps = running_summaries(rows)
        self.assertEqual(snaps[0]["summaries"]["MRE-"]["valid_count"], 0)
        self.assertEqual(snaps[1]["summaries"]["ACC+"]["best_record_id"], "r1")
        self.assertEqual(snaps[1]["summaries"]["MRE-"]["best_record_id"], "r2")
        self.assertEqual(snaps[1]["summaries"]["MRE-"]["best_value"], 2)
        self.assertEqual(snaps[0]["summaries"]["ACC+"]["valid_count"], 1)

    def test_csv_integration_maps_exact_best_configurations(self):
        fields = ["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion", "Min_sample_leaves", "Min_impurity_decrease", "Max_depth"]
        rows = [{"MRE-": "3", "ACC+": ".2", "PRED40+": ".9", "N_estimators": "10", "criterion": "a", "Min_sample_leaves": "1", "Min_impurity_decrease": "0", "Max_depth": "2"},
                {"MRE-": "1", "ACC+": ".8", "PRED40+": ".4", "N_estimators": "20", "criterion": "b", "Min_sample_leaves": "2", "Min_impurity_decrease": ".1", "Max_depth": "3"},
                {"MRE-": "2", "ACC+": ".7", "PRED40+": "1", "N_estimators": "30", "criterion": "c", "Min_sample_leaves": "3", "Min_impurity_decrease": ".2", "Max_depth": "4"}]
        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows); handle.flush()
            from flow import running_summaries_csv
            best = running_summaries_csv(handle.name)[-1]["summaries"]
        self.assertEqual(best["MRE-"]["best_record_id"], 2); self.assertEqual(best["ACC+"]["best_record_id"], 2); self.assertEqual(best["PRED40+"]["best_record_id"], 3)
        self.assertEqual(best["MRE-"]["best_configuration"], {k: rows[1][k] for k in fields[3:]})

    def test_best_configuration_is_defensively_copied(self):
        configuration = {"x": 1}; rows = [{"record_id": "r1", "configuration": configuration, "MRE-": "1"}]
        snapshots = running_summaries(rows); configuration["x"] = 99
        self.assertEqual(snapshots[0]["summaries"]["MRE-"]["best_configuration"], {"x": 1})

    def test_iterator_processes_record_before_requesting_next(self):
        first = {"record_id": "r1", "configuration": {"x": 1}, "MRE-": "1"}
        def records():
            yield first
            first["configuration"]["x"] = 99
            yield {"record_id": "r2", "configuration": {"x": 2}, "MRE-": "2"}
        snapshots = running_summaries(records())
        self.assertEqual(snapshots[0]["summaries"]["MRE-"]["best_configuration"], {"x": 1})

if __name__ == "__main__": unittest.main()
