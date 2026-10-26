import os
import tempfile
import unittest

from source.detect_anomaly import analyze, read_rows


HEADER = "a,b,Performance-\n"


class DetectorTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile(mode="w", delete=False, newline="")
        handle.write(text)
        handle.close()
        self.addCleanup(lambda: os.unlink(handle.name))
        return handle.name

    def test_reads_trimmed_numeric_values(self):
        features, rows = read_rows(self.write(HEADER + "0,0,10.0 \n"))
        self.assertEqual(features, ["a", "b"])
        self.assertEqual(rows[0][1], 10.0)

    def test_flags_isolated_large_residual(self):
        text = HEADER + "0,0,10\n0,1,10\n1,0,10\n1,1,100\n"
        _, records, _, _ = analyze(self.write(text), threshold=3)
        self.assertEqual([r["row"] for r in records if r["anomalous"]], [5])

    def test_rejects_non_binary_feature(self):
        with self.assertRaises(ValueError):
            read_rows(self.write(HEADER + "2,0,10\n"))


if __name__ == "__main__":
    unittest.main()
