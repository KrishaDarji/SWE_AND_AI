import csv, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from source.spot_quality import analyze

class SpotTests(unittest.TestCase):
    def make(self, rows):
        fd, path = tempfile.mkstemp(suffix=".csv"); os.close(fd)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        self.addCleanup(lambda: os.unlink(path)); return path

    def test_range_error(self):
        rows = [{"MRE-":"0.1","ACC+":"0.5","PRED40+":"0.5","x":"1"},
                {"MRE-":"0.2","ACC+":"1.2","PRED40+":"0.4","x":"2"}]
        report = analyze(self.make(rows))
        flagged = next(item for item in report["flagged"] if item["row"] == 3)
        self.assertTrue(any("outside expected" in x for x in flagged["reasons"]))

    def test_bad_number(self):
        report = analyze(self.make([{"MRE-":"bad","ACC+":"0.5","PRED40+":"0.5","x":"1"}]))
        self.assertEqual(len(report["parse_errors"]), 1)
        self.assertEqual(report["valid_rows"], 0)

if __name__ == "__main__":
    unittest.main()
