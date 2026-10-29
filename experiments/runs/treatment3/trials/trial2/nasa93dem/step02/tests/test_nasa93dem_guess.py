import json, subprocess, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[8]
SRC = Path(__file__).resolve().parents[1] / "source"
sys.path.insert(0, str(SRC))
from nasa93dem_guess import TARGET, evaluate_model, load_dataset, predict_effort, train_model
DATA = ROOT / "moot/optimize/process/nasa93dem.csv"

class Tests(unittest.TestCase):
    def test_load(self):
        rows, fields = load_dataset(DATA); self.assertEqual(len(rows), 93); self.assertIn(TARGET, fields); self.assertEqual(rows[0][TARGET], 117.6)
    def test_metrics_reproducible(self):
        rows, fields = load_dataset(DATA); a = evaluate_model(rows, fields); b = evaluate_model(rows, fields)
        self.assertEqual(a, b); self.assertEqual(a["test_rows"], 19); self.assertGreaterEqual(a["mae"], 0); self.assertGreaterEqual(a["rmse"], 0); self.assertTrue(-1 <= a["r2"] <= 1)
    def test_api(self):
        rows, fields = load_dataset(DATA); model, names = train_model(rows, fields)
        self.assertGreater(predict_effort(model, {n: rows[0][n] for n in names}), 0)
    def test_cli(self):
        out = subprocess.run([sys.executable, str(SRC / "nasa93dem_guess.py"), str(DATA)], capture_output=True, text=True, check=True)
        self.assertIn("rmse", json.loads(out.stdout))

if __name__ == "__main__": unittest.main()
