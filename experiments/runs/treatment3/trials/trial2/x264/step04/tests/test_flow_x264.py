import csv
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "source"))
from flow_x264 import EvaluationState, evaluate, parse_record  # noqa: E402


class FlowTests(unittest.TestCase):
    def test_invalid_and_incomplete_records_are_counted(self):
        state = EvaluationState()
        state.ingest(["2"] * 16 + ["3.0"], 2)
        state.ingest(["0"] * 15 + ["3.0"], 3)
        state.ingest(["1"] * 16 + ["2.0"], 4)
        self.assertEqual(state.invalid_records, 2)
        self.assertEqual(state.best.row_number, 4)

    def test_tie_keeps_earliest_configuration(self):
        state = EvaluationState()
        state.ingest(["1"] * 16 + ["1.0"], 2)
        state.ingest(["0"] * 16 + ["1.0"], 3)
        self.assertEqual(state.best.row_number, 2)
        self.assertEqual(state.best.configuration, (1,) * 16)

    def test_real_dataset_is_sequentially_evaluable(self):
        dataset = HERE.parents[6] / "moot/optimize/config/X264_AllMeasurements.csv"
        first = evaluate(dataset)
        second = evaluate(dataset)
        self.assertEqual(first.summary(), second.summary())
        self.assertEqual(first.valid_observations, 1152)
        self.assertEqual(first.invalid_records, 0)
        self.assertEqual(first.best.row_number, 72)

    def test_header_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("wrong\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate(path)


if __name__ == "__main__":
    unittest.main()
