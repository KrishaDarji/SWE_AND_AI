import csv
import tempfile
import unittest
from pathlib import Path

from source.nasa93_summary import repair_proposal, repair_demonstration, prepare_learning_data


class RepairTests(unittest.TestCase):
    def fixture(self, rows):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        handle.write("idX,setting,team,Kloc+,Effort-,Defects-\n")
        for row in rows:
            handle.write(",".join(map(str, row)) + "\n")
        handle.close(); path = Path(handle.name)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        return path

    def test_invalid_and_already_feasible(self):
        path = self.fixture([("a", "x", "1", "10", "1"), ("b", "y", "1", "20", "2"),
                             ("c", "z", "1", "30", "3"), ("d", "w", "1", "40", "4")])
        with self.assertRaises(ValueError): repair_proposal(path, {"setting": "x"}, 10)
        result = repair_proposal(path, {"setting": "x", "team": "1"}, 100)
        self.assertEqual(result["status"], "no-repair-needed")
        self.assertEqual(result["change_count"], 0)

    def test_minimum_change_and_training_only_candidates(self):
        path = self.fixture([("a", "x", "1", "10", "10"), ("b", "x", "2", "20", "20"),
                             ("c", "y", "1", "30", "30"), ("d", "y", "2", "40", "40"),
                             ("e", "z", "1", "50", "50"), ("f", "z", "2", "60", "60")])
        result = repair_demonstration(path, 35)
        self.assertEqual(result["status"], "repair-found")
        self.assertLessEqual(result["candidate_estimated_effort"], 35)
        self.assertIn(result["candidate_training_csv_record_index"], result["training_indices"])
        self.assertEqual(result["change_count"], len(result["changed_fields"]))

    def test_no_feasible_repair_and_outcome_isolation(self):
        path = self.fixture([("a", "x", "1", "10", "10"), ("b", "y", "1", "20", "20"),
                             ("c", "z", "1", "30", "30"), ("d", "w", "1", "40", "40")])
        proposal = {"setting": "never", "team": "never"}
        first = repair_proposal(path, proposal, 0)
        self.assertEqual(first["status"], "no-feasible-repair")
        with open(path, newline="") as handle:
            rows = list(csv.reader(handle))
        for row in rows[1:]: row[4] = "999999"
        altered = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
        csv.writer(altered).writerows(rows); altered.close(); altered_path = Path(altered.name)
        self.addCleanup(lambda: altered_path.unlink(missing_ok=True))
        second = repair_proposal(altered_path, proposal, 0)
        self.assertEqual(first["status"], second["status"])
        self.assertEqual(first["candidate_descriptors"], second["candidate_descriptors"])


if __name__ == "__main__": unittest.main()
