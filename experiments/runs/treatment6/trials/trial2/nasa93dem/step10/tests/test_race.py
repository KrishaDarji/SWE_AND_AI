import csv, math, pathlib, tempfile, unittest, sys
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / 'source'))
import nasa93_summary as ns

class RaceTests(unittest.TestCase):
    def dataset(self):
        return str(pathlib.Path(__file__).parents[8] / 'moot/optimize/process/nasa93dem.csv')

    def test_ten_reproducible_grouped_splits_and_summary(self):
        a, b = ns.race(self.dataset()), ns.race(self.dataset())
        self.assertEqual(a['splits'], b['splits'])
        self.assertEqual(len(a['splits']), 10)
        for row in a['splits']:
            self.assertTrue(row['train_groups'] > 0 and row['test_groups'] > 0)
            self.assertGreaterEqual(row['evaluated_count'], 0)
        self.assertEqual(a['summary']['comparable_split_count'], 10)

    def test_classification_and_tolerance_validation(self):
        self.assertEqual(ns.classify_difference(2), 'win')
        self.assertEqual(ns.classify_difference(-2), 'loss')
        self.assertEqual(ns.classify_difference(0.1, 0.1), 'tie')
        self.assertEqual(ns.classify_difference(None), 'undefined')
        for x in (-1, math.inf, math.nan):
            with self.assertRaises(ValueError): ns.race(self.dataset(), tolerance=x)

    def test_training_only_baseline_and_positions(self):
        result = ns.race(self.dataset(), seeds=[42])
        row = result['splits'][0]
        self.assertEqual(row['evaluated_count'] + row['excluded_test_targets'], row['test_rows'])
        self.assertTrue(all('position' in x and 'idX' in x for x in row['records']))
        _, projects, _, _ = ns.load(self.dataset()); train, _ = ns.split_groups(projects, 42)
        mean = sum(ns._target(p)[0] for p in train if ns._target(p)[1] == 'valid') / sum(ns._target(p)[1] == 'valid' for p in train)
        self.assertEqual(row['records'][0]['baseline_prediction'], mean)

    def test_edge_cases(self):
        f = tempfile.NamedTemporaryFile('w', delete=False, newline='')
        fields = ['idX'] + ns.CATEGORICAL + ['Effort-']; csv.writer(f).writerow(fields)
        for i, value in enumerate(['0', '?', 'bad', 'NaN'], 1): csv.writer(f).writerow([str(i), *(['a'] * 22), value])
        f.close()
        with self.assertRaises(ValueError): ns.race(f.name, seeds=[42])
        self.assertEqual(ns.classify_difference(float('nan')), 'undefined')

if __name__ == '__main__': unittest.main()
