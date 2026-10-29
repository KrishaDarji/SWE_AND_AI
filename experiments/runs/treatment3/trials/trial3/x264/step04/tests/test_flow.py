import csv
import tempfile
import unittest
from pathlib import Path

from flow import evaluate


class FlowTests(unittest.TestCase):
    def write(self, rows):
        file = tempfile.NamedTemporaryFile(mode="w", newline="", delete=False)
        writer = csv.writer(file)
        writer.writerows(rows)
        file.close()
        self.addCleanup(lambda: Path(file.name).unlink(missing_ok=True))
        return file.name

    def test_real_dataset_is_sequential_and_reproducible(self):
        path = Path(__file__).parents[8] / "moot/optimize/config/X264_AllMeasurements.csv"
        first, second = evaluate(path), evaluate(path)
        self.assertEqual(first, second)
        self.assertEqual((first.rows_seen, first.valid_rows, first.invalid_rows), (1152, 1152, 0))
        self.assertIsNotNone(first.best_configuration)

    def test_invalid_records_and_tie_break(self):
        header = ["b", "a", "Performance-"]
        path = self.write([header, ["1", "0", "2"], ["0", "1", "2"], ["x", "0", "1"], ["0", "0"]])
        result = evaluate(path)
        self.assertEqual(result.invalid_rows, 2)
        self.assertEqual(result.best_configuration, (0, 1))

    def test_schema_validation(self):
        with self.assertRaises(ValueError):
            evaluate(self.write([["x", "score"], ["0", "1"]]))


if __name__ == "__main__":
    unittest.main()
