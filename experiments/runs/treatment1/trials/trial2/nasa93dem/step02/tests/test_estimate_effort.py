import json, pathlib, sys, tempfile, unittest
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "source"))
from estimate_effort import estimate, load_rows, main

DATA = pathlib.Path("moot/optimize/process/nasa93dem.csv")

class EstimateTests(unittest.TestCase):
    def setUp(self):
        self.rows = load_rows(DATA)
        self.proposal = dict(self.rows[0])

    def test_exact_project_is_close_to_its_neighbors(self):
        result = estimate(self.rows, self.proposal, k=3)
        self.assertEqual(result["neighbors"], 3)
        self.assertGreater(result["estimated_effort_pm"], 0)

    def test_missing_and_unknown_values_do_not_crash(self):
        proposal = {"prec": "x", "flex": "?", "team": "new"}
        result = estimate(self.rows, proposal, k=5)
        self.assertEqual(result["neighbors"], 5)

    def test_cli_emits_approval_json(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "proposal.json"
            output.write_text(json.dumps(self.proposal))
            self.assertEqual(main([str(DATA), str(output), "--max-effort", "0"]), 0)

if __name__ == "__main__":
    unittest.main()
