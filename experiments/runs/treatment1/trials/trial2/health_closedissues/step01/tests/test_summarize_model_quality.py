import csv
import importlib.util
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("summarizer", ROOT / "source" / "summarize_model_quality.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SummarizerTests(unittest.TestCase):
    def make_csv(self):
        file = tempfile.NamedTemporaryFile(mode="w", newline="", suffix=".csv", delete=False)
        writer = csv.writer(file)
        writer.writerow(["MRE-", "ACC+", "PRED40+", "N_estimators", "criterion"])
        writer.writerow(["0.2", "0.7", "0.4", "10", "squared_error"])
        writer.writerow(["0.1", "0.6", "0.9", "20", "absolute_error"])
        file.close()
        self.addCleanup(pathlib.Path(file.name).unlink)
        return file.name

    def test_quality_directions_and_configuration(self):
        result = MODULE.summarize(self.make_csv())
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["quality_metrics"]["MRE-"]["direction"], "minimize")
        self.assertEqual(result["quality_metrics"]["ACC+"]["maximum"], 0.7)
        self.assertEqual(result["configuration"]["N_estimators"]["unique_values"], 2)
        self.assertEqual(result["configuration"]["criterion"]["type"], "categorical")
        self.assertEqual(result["best_observed"]["PRED40+"]["N_estimators"], "20")

    def test_bad_number_fails(self):
        path = self.make_csv()
        text = pathlib.Path(path).read_text().replace("0.1,0.6", "bad,0.6")
        pathlib.Path(path).write_text(text)
        with self.assertRaises(ValueError):
            MODULE.summarize(path)


if __name__ == "__main__":
    unittest.main()
