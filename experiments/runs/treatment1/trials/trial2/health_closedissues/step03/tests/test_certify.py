import csv
import importlib.util
import pathlib
import tempfile
import unittest

ROOT = pathlib.Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("certify", ROOT / "source" / "certify.py")
certify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(certify)


class CertifyTests(unittest.TestCase):
    def test_normalization_respects_directions(self):
        rows = [{"criterion": "a", "MRE-": 0, "ACC+": 1, "PRED40+": 1},
                {"criterion": "b", "MRE-": 1, "ACC+": 0, "PRED40+": 0}]
        self.assertEqual(certify.normalized_scores(rows), [1.0, 0.0])

    def test_reliable_difference_on_separated_groups(self):
        rows = []
        for family, base in (("a", 0.9), ("b", 0.1)):
            for _ in range(20):
                rows.append({"criterion": family, "MRE-": 1 - base,
                             "ACC+": base, "PRED40+": base})
        groups, comparisons, best = certify.certify(rows, iterations=200)
        self.assertEqual(best, "a")
        self.assertTrue(comparisons[0]["reliable"])
        self.assertGreater(comparisons[0]["difference"], 0)

    def test_load_rows(self):
        with tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv") as handle:
            writer = csv.DictWriter(handle, fieldnames=["criterion", "MRE-", "ACC+", "PRED40+"])
            writer.writeheader()
            writer.writerow({"criterion": "x", "MRE-": "1", "ACC+": "2", "PRED40+": "3"})
            handle.flush()
            self.assertEqual(len(certify.load_rows(handle.name)), 1)


if __name__ == "__main__":
    unittest.main()
