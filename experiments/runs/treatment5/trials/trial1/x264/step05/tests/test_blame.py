import csv, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from blame import analyze, rank_settings


class BlameTests(unittest.TestCase):
    def test_association_constants_missing_and_multivalue(self):
        rows = [("strong", "constant", "multi", "corr", "Performance-"),
                ("a", "x", "1", "a", "1"), ("a", "x", "2", "a", "2"),
                ("b", "x", "3", "b", "10"), ("b", "x", "?", "b", "11"),
                ("?", "x", "2", "?", "9"), ("a", "x", "1", "a", "1")]
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            csv.writer(f).writerows(rows); path = f.name
        try:
            settings, usable, overall, ranked, duplicates = analyze(path)
            self.assertEqual(len(usable), 6); self.assertEqual(ranked[0]["setting"], "corr")
            self.assertIn("multi", [item["setting"] for item in ranked])
            self.assertEqual(ranked[-1]["score"], 0.0)
            self.assertEqual(ranked[0]["groups"], 3)
            self.assertTrue(any(a == "strong" and b == "corr" for a, b in duplicates))
        finally: os.unlink(path)

    def test_deterministic_ties_and_primary_score(self):
        data = [("b", "a", "Performance-"), ("x", "x", "1"), ("y", "y", "3"),
                ("x", "x", "1"), ("y", "y", "3")]
        ranked1 = rank_settings(["b", "a"], [(dict(zip(data[0], row)), float(row[2])) for row in data[1:]])[1]
        ranked2 = rank_settings(["b", "a"], [(dict(zip(data[0], row)), float(row[2])) for row in data[1:]])[1]
        self.assertEqual([x["setting"] for x in ranked1], [x["setting"] for x in ranked2])
        self.assertEqual(ranked1[0]["score"], 1.0)

    def test_nonfinite_outcome_is_excluded(self):
        rows = [("x", "Performance-"), ("a", "1"), ("b", "nan")]
        with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
            csv.writer(f).writerows(rows); path = f.name
        try: self.assertEqual(len(analyze(path)[1]), 1)
        finally: os.unlink(path)


if __name__ == "__main__": unittest.main()
