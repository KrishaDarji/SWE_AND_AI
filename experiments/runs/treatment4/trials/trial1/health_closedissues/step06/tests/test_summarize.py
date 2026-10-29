"""Hand-calculated fixtures plus independent calculations on the actual CSV."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from summarize import summarize

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')
HEADERS = ['MRE-', 'ACC+', 'PRED40+', 'N_estimators', 'criterion',
           'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth']
CONFIG = ['N_estimators', 'criterion', 'Min_sample_leaves', 'Min_impurity_decrease', 'Max_depth']


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.csv'

    def write(self, rows, headers=HEADERS):
        with self.path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows(rows)
        return self.path

    def test_hand_calculated_numeric_statistics(self):
        self.write([[x, x, x, x, 'squared_error', x, x, x] for x in [1, 2, 3, 4]])
        report = summarize(self.path)
        expected = {'count': 4, 'mean': 2.5, 'std_sample': (5 / 3) ** .5,
                    'min': 1, 'q25': 1.75, 'median': 2.5, 'q75': 3.25, 'max': 4}
        for name in HEADERS:
            if name != 'criterion':
                for key, value in expected.items():
                    self.assertAlmostEqual(report['columns'][name]['statistics'][key], value)
        self.assertEqual(report['columns']['MRE-']['best_observed'], 1)
        self.assertEqual(report['columns']['ACC+']['best_observed'], 4)
        self.assertEqual(report['columns']['PRED40+']['direction'], 'maximize')
        self.assertEqual(report['configuration_fields'], CONFIG)

    def test_whitespace_missing_and_duplicates(self):
        a = [1, 2, 3, 10, ' squared_error ', 1, 0, 4]
        b = ['1.0', 2, 3, '10.0', 'squared_error', 1, 0, 4]
        c = [9, 8, 7, 10, 'squared_error ', 1, 0, 4]
        d = ['', ' ', '', '', '   ', '', '', '']
        report = summarize(self.write([a, b, c, d]))
        self.assertEqual(report['record_count'], 4)
        self.assertEqual(report['column_count'], 8)
        self.assertEqual(report['columns']['criterion']['value_counts'], {'squared_error': 3})
        self.assertEqual(report['columns']['criterion']['distinct_values'], ['squared_error'])
        for info in report['columns'].values():
            self.assertEqual(info['missing_count'], 1)
        self.assertEqual(report['duplicate_records'], {'unique_count': 3, 'repeated_group_count': 1,
                         'records_in_repeated_groups': 2, 'extra_record_count': 1})
        self.assertEqual(report['repeated_configurations'], {'unique_count': 2, 'repeated_group_count': 1,
                         'records_in_repeated_groups': 3, 'extra_record_count': 2})

    def test_all_missing_and_single_observation(self):
        report = summarize(self.write([[''] * 8]))
        self.assertIsNone(report['columns']['MRE-']['statistics'])
        self.assertIsNone(report['columns']['MRE-']['best_observed'])
        self.assertEqual(report['columns']['MRE-']['best_observed_count'], 0)
        report = summarize(self.write([[1, 2, 3, 10, 'poisson', 1, 0, 4]]))
        self.assertIsNone(report['columns']['MRE-']['statistics']['std_sample'])
        self.assertEqual(report['columns']['MRE-']['statistics']['q25'], 1)

    def test_invalid_inputs(self):
        for headers in [[], HEADERS[:-1], HEADERS + ['extra'], HEADERS[:-1] + ['MRE-']]:
            with self.subTest(headers=headers):
                with self.assertRaises(ValueError):
                    summarize(self.write([], headers))
        with self.assertRaises(ValueError):
            summarize(self.write([]))
        for row in [[1], [1] * 9, ['bad'] + [1] * 7,
                    ['nan'] + [1] * 7, ['inf'] + [1] * 7, ['-inf'] + [1] * 7]:
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    summarize(self.write([row]))
        with self.assertRaises(FileNotFoundError):
            summarize(Path(self.temp.name) / 'missing.csv')

    def test_reordered_columns(self):
        report = summarize(self.write([[4, 0, 1, 'poisson', 10, 3, 2, 1]], HEADERS[::-1]))
        self.assertEqual(report['column_names'], HEADERS[::-1])
        self.assertEqual(report['columns']['MRE-']['best_observed'], 1)

    def test_cli_success_and_expected_failure(self):
        self.write([[1, 2, 3, 10, 'poisson', 1, 0, 4]])
        output = Path(self.temp.name) / 'report'
        command = [sys.executable, str(ROOT / 'source/summarize.py'), str(self.path), '--output-dir', str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads((output / 'summary.json').read_text())['record_count'], 1)
        self.assertEqual((output / 'summary.txt').read_text(), result.stdout)
        self.path.unlink()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Error:', result.stderr)


class ActualDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = summarize(DATASET)
        with DATASET.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            cls.headers = reader.fieldnames
            cls.rows = [{k: v.strip() for k, v in row.items()} for row in reader]

    def test_schema_and_every_column_against_independent_csv_calculations(self):
        self.assertEqual(len(self.rows), 10000)
        self.assertEqual(self.headers, HEADERS)
        self.assertEqual(self.report['record_count'], len(self.rows))
        self.assertEqual(self.report['column_count'], len(HEADERS))
        for name in self.headers:
            info = self.report['columns'][name]
            raw = [row[name] for row in self.rows if row[name] != '']
            self.assertEqual(info['missing_count'], len(self.rows) - len(raw))
            if name == 'criterion':
                self.assertEqual(info['distinct_values'], sorted(set(raw)))
                self.assertEqual(info['distinct_count'], len(set(raw)))
                self.assertEqual(info['value_counts'], dict(Counter(raw)))
                continue
            values = np.asarray(raw, dtype=float)
            self.assertEqual(info['distinct_values'], sorted(set(values.tolist())))
            self.assertEqual(info['distinct_count'], len(set(values.tolist())))
            expected = dict(zip(['min', 'q25', 'median', 'q75', 'max'], np.quantile(values, [0, .25, .5, .75, 1])))
            expected.update(count=len(values), mean=np.mean(values), std_sample=np.std(values, ddof=1))
            for key, value in expected.items():
                self.assertAlmostEqual(info['statistics'][key], value, places=12, msg=f'{name}.{key}')
            if name in ['MRE-', 'ACC+', 'PRED40+']:
                best = np.min(values) if name == 'MRE-' else np.max(values)
                self.assertEqual(info['best_observed'], best)
                self.assertEqual(info['best_observed_count'], int(np.sum(values == best)))

    def test_duplicate_counts_independently(self):
        normalized = [{k: (v if k == 'criterion' else float(v)) if v else None
                       for k, v in row.items()} for row in self.rows]
        for key, names in [('duplicate_records', self.headers), ('repeated_configurations', CONFIG)]:
            groups = Counter(tuple(row[name] for name in names) for row in normalized)
            repeated = [count for count in groups.values() if count >= 2]
            self.assertEqual(self.report[key], {'unique_count': len(groups),
                             'repeated_group_count': len(repeated),
                             'records_in_repeated_groups': sum(repeated),
                             'extra_record_count': len(normalized) - len(groups)})


if __name__ == '__main__':
    unittest.main(verbosity=2)
