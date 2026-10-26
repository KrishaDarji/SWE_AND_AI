import csv, os, tempfile, unittest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "source"))
from certify import analyze

class CertifyTests(unittest.TestCase):
    def test_detects_clear_improvement(self):
        fd, p = tempfile.mkstemp(text=True); os.close(fd)
        try:
            with open(p, "w", newline="") as f:
                w=csv.writer(f); w.writerow(["g","y"])
                for n in [1,2,3,4,5,6]: w.writerow([0,n])
                for n in [11,12,13,14,15,16]: w.writerow([1,n])
            x=analyze(p,"g","0","1","y")
            self.assertTrue(x["credible"]); self.assertGreater(x["difference"],0)
        finally: os.unlink(p)
    def test_rejects_empty_group(self):
        fd, p = tempfile.mkstemp(text=True); os.close(fd)
        try:
            with open(p, "w") as f: f.write("g,y\n0,1\n0,2\n")
            with self.assertRaises(ValueError): analyze(p,"g","0","1","y")
        finally: os.unlink(p)

if __name__ == "__main__": unittest.main()
