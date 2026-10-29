"""Step 10 independent arithmetic, outcome blindness and regression integration."""
import copy
import csv
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'source'))
from race import compare_records, race, race_rows, render_text, uniform_identity
from choose import choose_rows
from summarize import CONFIGURATION, OBJECTIVES

DATASET = Path('/Users/vivek/D Drive/vs code/SWEAI_Project/SWE_AND_AI/moot/optimize/hpo/Health-ClosedIssues0000.csv')


def row(mre=1, acc=2, pred=3, estimators=100):
    return dict(zip([*CONFIGURATION, *OBJECTIVES],
                    [estimators, 'squared_error', 3, 0, 5, mre, acc, pred]))


def stable(report):
    result = copy.deepcopy(report)
    result.pop('timing')
    for trial in result['trials']:
        trial.pop('baseline_runtime_seconds')
    return result


class RaceTests(unittest.TestCase):
    def test_exact_step7_eligibility_all_missing_fields_cap_and_identity(self):
        rows = [row(estimators=100), row(estimators=101), row(estimators=99.5)]
        for name in [*CONFIGURATION, *OBJECTIVES]:
            for missing in (None, '', '  '):
                r = row()
                r[name] = missing
                rows.append(r)
        result = race_rows(rows, seeds=[42, 43])
        old = choose_rows(rows)
        self.assertEqual(result['candidate_record'], old['selected_record'])
        self.assertEqual(result['eligibility']['counts'], old['counts'])
        self.assertEqual(result['eligibility']['excluded_records'], old['excluded_records'])
        self.assertEqual(result['eligibility']['feasible_row_ids'], [1, 3])
        self.assertEqual(result['eligibility']['baseline_feasible_count'], 2)
        for t in result['trials']:
            self.assertIn(t['baseline_record']['row_id'], [1, 3])
        self.assertFalse(result['cap']['measured_compute_budget'])

    def test_fresh_rng_reproducible_and_duplicate_seeds(self):
        rows = [row(mre=i) for i in range(20)]
        seeds = [42, 51, 42, -1, 0]
        result = race_rows(rows, seeds=seeds)
        self.assertEqual(stable(result), stable(race_rows(rows, seeds=seeds)))
        ids = result['eligibility']['feasible_row_ids']
        for trial in result['trials']:
            self.assertEqual(trial['baseline_record']['row_id'], random.Random(trial['seed']).choice(ids))
        self.assertEqual(result['trials'][0]['baseline_record'], result['trials'][2]['baseline_record'])
        with patch('race.random.Random', wraps=random.Random) as rng:
            race_rows(rows, seeds=seeds)
            self.assertEqual([c.args[0] for c in rng.call_args_list], seeds)

    def test_nonmissing_outcome_values_do_not_change_baseline_identities(self):
        rows = [row(mre=i, acc=10-i, pred=i*2) for i in range(30)]
        rows[4]['MRE-'] = None
        changed = copy.deepcopy(rows)
        for i, r in enumerate(changed):
            for name in OBJECTIVES:
                if r[name] is not None:
                    r[name] = (-1 if i % 2 else 1) * (10000 + 23*i)
        before = race_rows(rows)
        after = race_rows(changed)
        self.assertEqual(before['eligibility'], after['eligibility'])
        self.assertEqual([t['baseline_record']['row_id'] for t in before['trials']],
                         [t['baseline_record']['row_id'] for t in after['trials']])
        # The RNG-facing interface contains only identities and a seed.
        self.assertIn(uniform_identity([1, 7, 8], 42), [1, 7, 8])

    def test_directions_secondary_losses_and_outcome_tie_excludes_identity(self):
        report = race_rows([row(1, -50, -50), row(2, 100, 100)], seeds=list(range(20)))
        trials = [t for t in report['trials'] if t['baseline_record']['row_id'] == 2]
        self.assertTrue(trials)
        for t in trials:
            c = t['comparison']
            self.assertEqual(c['lexicographic_relation'], 'win')
            self.assertEqual(c['secondary_objective_losses'], ['ACC+', 'PRED40+'])
            self.assertEqual(c['per_objective']['MRE-']['observed_relation'], 'win')
            self.assertEqual(c['per_objective']['ACC+']['candidate_minus_baseline'], -150)
            self.assertEqual(c['per_objective']['PRED40+']['observed_relation'], 'loss')
        tied = race_rows([row(), row()], seeds=list(range(20)))
        self.assertTrue(any(t['baseline_record']['row_id'] == 2 for t in tied['trials']))
        self.assertEqual(tied['descriptive_summaries']['lexicographic'], {'win': 0, 'tie': 20, 'loss': 0})

    def test_priority_acc_then_pred_and_loss_branch(self):
        def rec(i, mre, acc, pred):
            return {'row_id': i, 'observed_objectives': {'MRE-': mre, 'ACC+': acc, 'PRED40+': pred}}
        self.assertEqual(compare_records(rec(1, 1, 3, 0), rec(2, 1, 2, 100))['lexicographic_relation'], 'win')
        self.assertEqual(compare_records(rec(1, 1, 2, 4), rec(2, 1, 2, 3))['lexicographic_relation'], 'win')
        self.assertEqual(compare_records(rec(1, 2, 100, 100), rec(2, 1, 0, 0))['lexicographic_relation'], 'loss')
        with self.assertRaises(ValueError):
            compare_records(rec(1, 1e308, 0, 0), rec(2, -1e308, 0, 0))

    def test_empty_feasible_set(self):
        for rows in ([], (), [{}], [row(mre=None)], [row(estimators=101)]):
            result = race_rows(rows)
            self.assertEqual(result['status'], 'no_feasible_candidate')
            self.assertIsNone(result['candidate_record'])
            self.assertEqual(result['descriptive_summaries']['compared_seed_count'], 0)
            for t in result['trials']:
                self.assertIsNone(t['baseline_record'])
                self.assertIsNone(t['comparison'])
            self.assertIsNone(result['descriptive_summaries']['per_objective']['MRE-']['mean_candidate_minus_baseline'])

    def test_invalid_inputs(self):
        for cap in (0, -1, True, None, '100', float('nan'), float('inf'), 10**1000):
            with self.subTest(cap=repr(cap)), self.assertRaises(ValueError):
                race_rows([row()], cap)
        for seeds in ([], (), None, '42', [True], [42.0], ['42']):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                race_rows([row()], seeds=seeds)
        for rows in (None, {}, 'rows', [None], [row(mre=float('nan'))],
                     [row(acc=True)], [dict(row(), unknown=1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                race_rows(rows)

    def test_actual_csv_independent_counts_differences_and_summaries(self):
        with DATASET.open(newline='', encoding='utf-8-sig') as handle:
            raw = list(csv.DictReader(handle))
        result = race(DATASET)
        ids = [i for i, r in enumerate(raw, 1)
               if all(r[n].strip() for n in [*CONFIGURATION, *OBJECTIVES]) and float(r['N_estimators']) <= 100]
        self.assertEqual(result['eligibility']['feasible_row_ids'], ids)
        candidate_id = min(ids, key=lambda i: (float(raw[i-1]['MRE-']), -float(raw[i-1]['ACC+']), -float(raw[i-1]['PRED40+']), i))
        self.assertEqual(candidate_id, result['candidate_record']['row_id'])
        deltas = {n: [] for n in OBJECTIVES}
        for t in result['trials']:
            baseline_id = random.Random(t['seed']).choice(ids)
            self.assertEqual(t['baseline_record']['row_id'], baseline_id)
            for name, direction in OBJECTIVES.items():
                delta = float(raw[candidate_id-1][name]) - float(raw[baseline_id-1][name])
                deltas[name].append(delta)
                detail = t['comparison']['per_objective'][name]
                self.assertEqual(detail['candidate_minus_baseline'], delta)
                self.assertEqual(detail['improvement_in_objective_direction'], -delta if direction == 'minimize' else delta)
        for name, values in deltas.items():
            self.assertAlmostEqual(result['descriptive_summaries']['per_objective'][name]['mean_candidate_minus_baseline'], sum(values)/len(values))
        self.assertEqual(result['statistical_analysis']['status'], 'pending')
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)
        for value in result['timing'].values():
            if isinstance(value, float):
                self.assertGreaterEqual(value, 0)
        for phrase in ('not measured compute cost', 'not independent LLM trials', 'expected by construction', 'secondary_objective_losses'):
            self.assertIn(phrase, render_text(result))

    def test_cli_valid_and_expected_errors(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests') as tmp:
            p = Path(tmp)
            output = p / 'race.json'
            cmd = [sys.executable, str(ROOT / 'source/race.py'), str(DATASET), '--output', str(output)]
            result = subprocess.run(cmd + ['--max-estimators', '100', '--seeds', '42', '51'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['seeds'], [42, 51])
            self.assertEqual(result.stdout, render_text(report))
            for extra in (['--max-estimators', 'nan'], ['--max-estimators', '0'], ['--seeds'], ['--seeds', '1.5']):
                self.assertEqual(subprocess.run(cmd + extra, capture_output=True).returncode, 2)
            for text in ('wrong\n1\n', ','.join([*CONFIGURATION, *OBJECTIVES])+'\n'):
                fixture = p / 'invalid.csv'
                fixture.write_text(text)
                self.assertEqual(subprocess.run([sys.executable, str(ROOT/'source/race.py'), str(fixture), '--output', str(output)], capture_output=True).returncode, 2)
            self.assertEqual(subprocess.run([sys.executable, str(ROOT/'source/race.py'), str(p/'missing.csv'), '--output', str(output)], capture_output=True).returncode, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
