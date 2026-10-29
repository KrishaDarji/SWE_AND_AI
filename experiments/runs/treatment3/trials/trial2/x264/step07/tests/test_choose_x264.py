import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("choose_x264", ROOT / "source" / "choose_x264.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ChooseTests(unittest.TestCase):
    def test_load_and_select_real_dataset(self):
        rows, settings = module.load_dataset(Path("moot/optimize/config/X264_AllMeasurements.csv"))
        selected = module.choose(rows)
        self.assertEqual(len(rows), 1152)
        self.assertEqual(len(settings), 16)
        self.assertEqual(selected[module.PERFORMANCE_COLUMN], 254.587)
        self.assertTrue(all(selected[k] == v for k, v in module.QUALITY_POLICY.items()))

    def test_no_feasible_configuration(self):
        with self.assertRaises(LookupError):
            module.choose([{**{k: 1 for k in module.QUALITY_POLICY}, module.PERFORMANCE_COLUMN: 1.0}])

    def test_validation_rejects_missing_quality_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("no_cabac,Performance-\n0,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.load_dataset(path)


if __name__ == "__main__":
    unittest.main()
