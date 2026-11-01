import csv, os, tempfile, unittest
from source.x264_certify import certify, _t_cdf, _t_quantile

class CertifyTests(unittest.TestCase):
    def fixture(self, rows):
        h=tempfile.NamedTemporaryFile("w", newline="", delete=False); w=csv.writer(h); w.writerow(["no_8x8dct","Performance-"]); w.writerows(rows); h.close(); self.addCleanup(lambda: os.unlink(h.name)); return h.name
    def test_direction_and_threshold_uncertainty(self):
        r=certify(self.fixture([[0,10],[0,11],[0,12],[1,20],[1,21],[1,22]]),5)
        self.assertEqual(r["mean_difference_A_minus_B"],-10); self.assertTrue(r["point_estimate_exceeds_threshold"]); self.assertTrue(r["interval_entirely_beyond_threshold"])
        r=certify(self.fixture([[0,10],[0,11],[1,15.5],[1,16.5]]),5)
        self.assertTrue(r["point_estimate_exceeds_threshold"]); self.assertTrue(r["interval_crosses_practical_threshold"])
    def test_welch_reference_and_large_statistic(self):
        r=certify(self.fixture([[0,1],[0,2],[0,3],[1,4],[1,5],[1,6]]),0)
        self.assertAlmostEqual(r["degrees_of_freedom"],4,places=12); self.assertAlmostEqual(_t_quantile(.975,4),2.7764451051978,places=8); self.assertAlmostEqual(_t_cdf(0,4),.5,places=14); self.assertGreater(r["p_value_two_sided"],0)
    def test_invalid_missing_insufficient_zero_and_threshold(self):
        for rows in ([[0,1],[0,2],[1,"?"]], [[0,1],[1,2]], [[0,1],[0,1],[1,2],[1,2]]):
            with self.assertRaises(ValueError): certify(self.fixture(rows))
        with self.assertRaises(ValueError): certify(self.fixture([[0,1],[0,2],[1,3],[1,4]]),-1)

if __name__ == "__main__": unittest.main()
