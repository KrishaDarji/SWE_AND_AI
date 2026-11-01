import math, pathlib, tempfile, unittest, sys
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / "source"))
import nasa93_summary as ns

class ChooseTests(unittest.TestCase):
    def write(self, tags_efforts):
        f = tempfile.NamedTemporaryFile("w", delete=False, newline="")
        f.write("idX," + ",".join(ns.CATEGORICAL) + ",Kloc+,Effort-,Defects-\n")
        for ident, tag, effort in tags_efforts:
            vals = [tag] * 22 if isinstance(tag, str) else list(tag)
            f.write(",".join([ident] + vals + ["10", str(effort), "1"]) + "\n")
        f.close(); return f.name
    def test_minimum_change_and_complete_report(self):
        near = ["c"] * 22; near[0] = "b"
        p = self.write([("q", "c", 200), ("one", near, 10), ("two", "c", 200)])
        original = ns.split_groups
        fit_original = ns._fit_with_evidence
        ns.split_groups = lambda projects, seed=42: (projects[1:], projects[:1])
        ns._fit_with_evidence = lambda projects, neighbors=5: fit_original(projects, 1)
        r = ns.choose(p, 100)
        ns.split_groups = original
        ns._fit_with_evidence = fit_original
        self.assertEqual(r["status"], "successful recommendation")
        self.assertEqual(r["recommendation"]["changed_fields"], 1)
        self.assertEqual(len(r["baseline"]["descriptors"]), 22)
        self.assertEqual(r["recommendation"]["predicted_improvement"], r["baseline"]["predicted_effort"] - r["recommendation"]["predicted_effort"])
    def test_exact_cap_and_already_feasible(self):
        p = self.write([("q", "a", 50), ("other", "b", 100), ("third", "c", 150)])
        self.assertEqual(ns.choose(p, 100)["status"], "already feasible")
    def test_invalid_caps_and_insufficient_data(self):
        p = self.write([("q", "a", 50), ("other", "b", 100)])
        for cap in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError): ns.choose(p, cap)
        with self.assertRaises(ValueError): ns.choose(self.write([]), 1)
    def test_missing_baseline(self):
        p = self.write([("q", "?", 200), ("other", "b", 10), ("third", "c", 20)])
        original = ns.split_groups
        ns.split_groups = lambda projects, seed=42: (projects[1:], projects[:1])
        self.assertEqual(ns.choose(p, 100)["status"], "invalid/incomplete baseline")
        ns.split_groups = original

if __name__ == "__main__": unittest.main()
