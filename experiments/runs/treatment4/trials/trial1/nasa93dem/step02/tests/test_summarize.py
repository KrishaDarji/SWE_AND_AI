"""Independent fixtures and CSV arithmetic checks; uses only this dataset."""
import csv
import math
import os
import sys
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from summarize import read_projects, render_text, summarize

# Deliberately independent schema and numeric declarations.
HEADER = 'idX centerX YearX prec flex resl team pmat rely data cplx ruse docu time stor pvol acap pcap pcon apex plex ltex tool site sced Kloc+ Effort- Defects- MonthsX'.split()
NUMBERS = ['idX', 'YearX', 'Kloc+', 'Effort-', 'Defects-', 'MonthsX']
DATASET = Path(os.environ.get('NASA93_DATASET', '/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/process/nasa93dem.csv'))


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'fixture.csv'

    def write(self, rows, header=HEADER):
        with self.path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def row(self, number='1'):
        return [number if name in NUMBERS else ' n ' for name in HEADER]

    def test_hand_calculated_numeric_summary(self):
        self.write([self.row(str(x)) for x in [1, 2, 3, 4]])
        result = summarize(self.path)['columns']['Effort-']['statistics']
        expected = {'count': 4, 'sum': 10, 'mean': 2.5, 'std_sample': math.sqrt(5 / 3),
                    'min': 1, 'q25': 1.75, 'median': 2.5, 'q75': 3.25, 'max': 4}
        for key, value in expected.items():
            self.assertAlmostEqual(result[key], value)

    def test_categorical_whitespace_and_metadata(self):
        rows = [self.row() for _ in range(3)]
        for row, value in zip(rows, [' h ', 'h', ' l\t']):
            row[HEADER.index('prec')] = value
        self.write(rows)
        report = summarize(self.path)
        self.assertEqual(report['columns']['prec']['value_counts'], {'h': 2, 'l': 1})
        self.assertEqual(report['columns']['centerX']['value_counts'], {'n': 3})
        self.assertTrue(all(not x.endswith('X') for x in report['candidate_predictors']))
        self.assertEqual(report['objectives'], {'Kloc+': 'maximize', 'Effort-': 'minimize', 'Defects-': 'minimize'})
        self.assertNotIn('Effort-', report['candidate_predictors'])
        self.assertEqual(report['columns']['MonthsX']['role'], 'metadata')
        self.assertIn('MonthsX is descriptive metadata', render_text(report))

    def test_missing_values_and_small_samples(self):
        rows = [self.row() for _ in range(5)]
        for row, token in zip(rows, ['', '?', 'NA', 'N/A', 'null']):
            row[HEADER.index('Effort-')] = token
            row[HEADER.index('prec')] = ' ' + token + ' '
        for row in rows[1:]:
            row[HEADER.index('Kloc+')] = ''
        self.write(rows)
        result = summarize(self.path)
        self.assertEqual(result['total_missing_values'], 14)
        self.assertEqual(result['columns']['prec']['value_counts'], {})
        self.assertEqual(result['columns']['Effort-']['missing_count'], 5)
        stats = result['columns']['Effort-']['statistics']
        self.assertEqual(stats['count'], 0)
        self.assertTrue(all(value is None for key, value in stats.items() if key != 'count'))
        self.assertIsNone(result['columns']['Kloc+']['statistics']['std_sample'])
        self.assertEqual(result['columns']['Kloc+']['statistics']['mean'], 1)

    def test_invalid_numeric_and_nonfinite(self):
        for value in ['oops', 'NaN', 'inf', '-inf', '1e999']:
            with self.subTest(value=value):
                self.write([self.row(value)])
                with self.assertRaises(ValueError):
                    summarize(self.path)

    def test_invalid_schema_rows_empty_and_missing_file(self):
        for header, rows in [([], []), (HEADER, []), (HEADER[:-1], []),
                             (HEADER[:-1] + ['idX'], []), (HEADER, [self.row()[:-1]]),
                             (HEADER, [self.row() + ['extra']])]:
            with self.subTest(header=header, rows=rows):
                self.write(rows, header)
                with self.assertRaises(ValueError):
                    summarize(self.path)
        self.path.write_text('')
        with self.assertRaises(ValueError):
            read_projects(self.path)
        with self.assertRaises(FileNotFoundError):
            read_projects(self.path.parent / 'missing.csv')

    def test_malformed_csv(self):
        self.path.write_text(','.join(HEADER) + '\n"unterminated')
        with self.assertRaises(csv.Error):
            summarize(self.path)


class ActualDatasetTests(unittest.TestCase):
    def test_independent_csv_calculations(self):
        report = summarize(DATASET)
        with DATASET.open(newline='') as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, HEADER)
            rows = list(reader)
        self.assertEqual(len(rows), 93)
        self.assertEqual(report['project_count'], len(rows))
        self.assertEqual(report['column_count'], 29)
        self.assertEqual(report['numeric_column_count'], 6)
        self.assertEqual(report['categorical_column_count'], 23)
        self.assertEqual(len(report['candidate_predictors']), 22)
        missing_total = 0
        for name in HEADER:
            raw = [row[name].strip() for row in rows]
            present = [v for v in raw if v not in ('', '?', 'NA', 'N/A', 'null')]
            item = report['columns'][name]
            missing_total += len(raw) - len(present)
            self.assertEqual(item['missing_count'], len(raw) - len(present))
            self.assertEqual(item['nonmissing_count'], len(present))
            self.assertEqual(item['unique_count'], len(set(present)))
            if name not in NUMBERS:
                self.assertEqual(item['value_counts'], dict(Counter(present)))
                continue
            values = sorted(Decimal(value) for value in present)
            n = len(values)
            avg = sum(values) / n
            expected = {'count': n, 'sum': sum(values), 'mean': avg, 'min': values[0], 'max': values[-1],
                        'std_sample': (sum((v - avg) ** 2 for v in values) / (n - 1)).sqrt()}
            for label, numerator in [('q25', 1), ('median', 2), ('q75', 3)]:
                index = Decimal(n - 1) * numerator / 4
                low = int(index)
                fraction = index - low
                expected[label] = values[low] * (1 - fraction) + values[min(low + 1, n - 1)] * fraction
            for key, value in expected.items():
                with self.subTest(column=name, statistic=key):
                    self.assertTrue(math.isclose(item['statistics'][key], float(value), rel_tol=1e-12, abs_tol=1e-12))
        self.assertEqual(missing_total, 0)
        self.assertEqual(report['total_missing_values'], missing_total)


if __name__ == '__main__':
    unittest.main(verbosity=2)
