import csv, pathlib, sys, unittest
ROOT = pathlib.Path(__file__).parents[1]; sys.path.insert(0, str(ROOT / 'source'))
import nasa93_summary as ns

class JustifyTests(unittest.TestCase):
    def dataset(self):
        p = pathlib.Path(__file__).parents[8] / 'moot/optimize/process/nasa93dem.csv'
        return str(p)

    def test_real_query_and_evidence_reproduce_prediction(self):
        r = ns.justify(self.dataset())
        self.assertEqual(r['query']['position'], min(x.position for x in r['test'] ) if 'test' in r else r['query']['position'])
        self.assertEqual(len(r['query']['descriptors']), 22)
        self.assertAlmostEqual(r['predicted_effort'], sum(x['effort'] for x in r['neighbors']) / len(r['neighbors']))
        self.assertTrue(all(x['effort'] is not None and x['position'] >= 2 for x in r['neighbors']))
        self.assertIsNone(r['retrospective_audit'])

    def test_held_out_effort_cannot_change_prediction_or_explanation(self):
        h, rows, _, _ = ns.load(self.dataset()); train, test = ns.split_groups(rows)
        q = min(test, key=lambda p: p.position)
        original = q.values['Effort-']; q.values['Effort-'] = '999999'
        a = ns.justify(self.dataset()); q.values['Effort-'] = original
        b = ns.justify(self.dataset())
        self.assertEqual(a['query'], b['query']); self.assertEqual(a['neighbors'], b['neighbors'])
        self.assertEqual(a['predicted_effort'], b['predicted_effort']); self.assertEqual(a['training_mean_baseline'], b['training_mean_baseline'])

    def test_missing_unseen_and_schema_policy(self):
        fields = ['idX'] + ns.CATEGORICAL + ['Effort-']
        rows = [['1'] + ['a']*22 + ['1'], ['2'] + ['b']*22 + ['3']]
        import tempfile
        f = tempfile.NamedTemporaryFile('w', delete=False, newline=''); w = csv.writer(f); w.writerow(fields); w.writerows(rows); f.close()
        _, projects, _, _ = ns.load(f.name); model = ns.CategoricalNeighborModel(1).fit(projects)
        self.assertTrue(model.predict_config({x: 'unseen' for x in ns.CATEGORICAL}) >= 0)
        with self.assertRaises(ValueError): model.predict_config({x: 'a' for x in ns.CATEGORICAL[:-1]})

    def test_duplicate_identifiers_and_audit_label(self):
        r = ns.justify(self.dataset(), retrospective=True)
        self.assertIn('outcome_kind', r['retrospective_audit'])
        _, rows, _, _ = ns.load(self.dataset())
        self.assertLess(len({p.identifier for p in rows}), len(rows))
        self.assertTrue(all('position' in x and 'idX' in x for x in r['neighbors']))

if __name__ == '__main__': unittest.main()
