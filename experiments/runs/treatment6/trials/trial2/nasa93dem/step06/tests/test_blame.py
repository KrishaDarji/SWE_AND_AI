import csv, pathlib, sys, tempfile, unittest
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / 'source'))
import nasa93_summary as ns

class BlameTests(unittest.TestCase):
    def write(self, rows):
        f = tempfile.NamedTemporaryFile('w', delete=False, newline='')
        fields = ['idX'] + ns.CATEGORICAL + ['Effort-', 'Kloc+', 'Defects-']
        w = csv.writer(f); w.writerow(fields)
        for i, (prec, effort) in enumerate(rows, 1):
            vals = [str(i)] + [prec] + ['x'] * 21 + [str(effort), '9', '1']
            w.writerow(vals)
        f.close(); return f.name

    def test_known_eta_means_ss_and_directions(self):
        r = ns.blame(self.write([(' a ', 1), ('a', 3), ('b', 5), ('b', 7)]))['reports'][0]
        self.assertEqual(r['descriptor'], 'prec'); self.assertAlmostEqual(r['reference_mean'], 4)
        self.assertAlmostEqual(r['between_ss'], 16); self.assertAlmostEqual(r['total_ss'], 20)
        self.assertAlmostEqual(r['eta_squared'], .8)
        self.assertEqual(r['categories'], {'a': {'count': 2, 'mean_effort': 2.0}, 'b': {'count': 2, 'mean_effort': 6.0}})
        self.assertEqual(r['higher_mean_effort'], ['b']); self.assertEqual(r['lower_mean_effort'], ['a'])

    def test_missing_effort_and_descriptor_have_per_field_populations(self):
        path = self.write([('a', 1), ('?', 3), ('a', 'NaN'), ('b', 5)])
        reports = {x['descriptor']: x for x in ns.blame(path)['reports']}
        self.assertEqual(reports['prec']['eligible_rows'], 2)
        self.assertEqual(reports['prec']['excluded'], {'missing_descriptor': 1, 'invalid_or_missing_effort': 1})
        self.assertEqual(reports['flex']['eligible_rows'], 3)

    def test_constant_outcome_undefined_and_ties_deterministic(self):
        a = ns.blame(self.write([('a', 2), ('b', 2)]))['reports']
        b = ns.blame(self.write([('a', 2), ('b', 2)]))['reports']
        self.assertEqual([x['descriptor'] for x in a], [x['descriptor'] for x in b])
        self.assertTrue(all(x['eta_squared'] is None for x in a))

    def test_exact_predictors_and_exclusions(self):
        self.assertEqual(ns.CATEGORICAL, 'prec flex resl team pmat rely data cplx ruse docu time stor pvol acap pcap pcon apex plex ltex tool site sced'.split())
        self.assertTrue(not set(ns.CATEGORICAL) & set(ns.METADATA) and not set(ns.CATEGORICAL) & set(ns.OUTCOMES))

if __name__ == '__main__': unittest.main()
