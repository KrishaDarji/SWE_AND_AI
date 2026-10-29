import sys, pathlib, unittest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
from race_selector import load_dataset, evaluate, DATASET

class RaceTests(unittest.TestCase):
  def test_load_real_dataset(self):
    data = load_dataset(DATASET)
    self.assertEqual(len(data), 10000); self.assertTrue(0 <= data[0][1] <= 1)

  def test_evaluate_small_valid_stream(self):
    data = []
    for i in range(60):
        data.append(({"N_estimators": float(i % 3), "criterion": "a" if i % 2 else "b", "Min_sample_leaves": 1., "Min_impurity_decrease": 0., "Max_depth": 3.}, float(i % 5) / 4))
    out = evaluate(data, warmup=20, block=10)
    self.assertEqual(out["n_decisions"], 4)
    self.assertTrue(0 <= out["learned_selector"]["selected_mean_acc"] <= 1)

  def test_rejects_bad_acc(self):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv") as f:
      f.write("N_estimators,criterion,Min_sample_leaves,Min_impurity_decrease,Max_depth,ACC+\n1,a,1,0,2,2\n"); f.flush()
      with self.assertRaises(ValueError): load_dataset(f.name)

if __name__ == "__main__": unittest.main()
