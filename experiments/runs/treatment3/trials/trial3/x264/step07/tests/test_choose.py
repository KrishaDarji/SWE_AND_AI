import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "source"))
import choose


class ChooseTests(unittest.TestCase):
    def test_real_dataset_has_a_feasible_winner(self):
        rows = choose.load_dataset(choose.DATASET)
        selected, count = choose.choose(rows)
        self.assertGreater(count, 0)
        self.assertIsNotNone(selected)
        self.assertTrue(choose.is_feasible(selected))
        self.assertEqual(float(selected[choose.OBJECTIVE]), 268.72)

    def test_no_feasible_configuration(self):
        rows = [{c: "1" for c in choose.REQUIRED_COLUMNS[:-1]}]
        rows[0][choose.OBJECTIVE] = "1"
        self.assertEqual(choose.choose(rows), (None, 0))

    def test_validation_rejects_bad_value(self):
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=choose.REQUIRED_COLUMNS)
            writer.writeheader()
            row = {c: "0" for c in choose.REQUIRED_COLUMNS}
            row[choose.OBJECTIVE] = "bad"
            writer.writerow(row)
            path = handle.name
        try:
            with self.assertRaises(ValueError):
                choose.load_dataset(path)
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
