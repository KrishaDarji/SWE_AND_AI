import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TreatmentManifestTests(unittest.TestCase):
    def test_manifest_matches_required_seven_treatments(self):
        manifest = json.loads((ROOT / "experiments/treatments.json").read_text())
        treatments = manifest["treatments"]
        self.assertEqual([item["id"] for item in treatments], list(range(1, 8)))
        self.assertEqual(manifest["domains"], ["x264", "nasa93dem", "health_closedissues"])
        self.assertEqual(manifest["ladder_steps"], 10)

    def test_reuse_pairs_are_present(self):
        manifest = json.loads((ROOT / "experiments/treatments.json").read_text())
        treatments = manifest["treatments"]
        self.assertEqual(treatments[0]["reuse_instructed"], False)
        self.assertEqual(treatments[1]["reuse_instructed"], True)
        self.assertEqual(treatments[2]["reuse_instructed"], False)
        self.assertEqual(treatments[3]["reuse_instructed"], True)
        self.assertEqual(treatments[4]["reuse_instructed"], False)
        self.assertEqual(treatments[5]["reuse_instructed"], True)
        self.assertEqual(treatments[6]["writer"], "human_team")


if __name__ == "__main__":
    unittest.main()
