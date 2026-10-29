"""Retrospective constrained selection of recorded Health configurations."""
import argparse
import csv
import json
import math
from pathlib import Path

import sklearn

from justify import difference, normalize_records, selection_key
from summarize import CONFIGURATION, OBJECTIVES, load_rows

RULE = ('Minimize MRE-, then maximize ACC+, then maximize PRED40+, then earliest '
        'one-based CSV data-row number. Ties use exact normalized float equality.')
LIMITATIONS = [
    'The dataset contains no measured compute cost. A real compute-budget guarantee is unavailable. N_estimators is only a configuration constraint for this illustrative demo; other settings and hardware also affect actual compute.',
    'Full-data retrospective selection, not a budgeted search over hidden outcomes or an independent validation result. Selection optimism can overstate future performance; freeze the configuration and validate on independent representative data.',
    'Optimality is limited to the complete recorded feasible rows under the stated lexicographic rule. No unseen hyperparameter combinations are generated. Duplicate configurations remain separate observations; earliest row is not a time claim.',
    'MRE- takes priority over any secondary-objective loss. There are no supplied practical thresholds or overall scores. Observed metric formulas, units and estimator semantics are unspecified.',
    'Missing configuration fields or objectives exclude a row without imputation; missingness may be selective. Finite numeric configuration values are retained without inventing estimator parameter domains.',
    'A positive finite numeric cap is a user constraint, not a compute measurement or a guarantee of deployability. Exact float comparisons may be sensitive to recorded precision.'
]


def validate_cap(value):
    """Accept positive finite numeric caps; strings are parsed only by the CLI."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('max_estimators must be a positive finite number')
    try:
        valid = math.isfinite(value) and value > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('max_estimators must be a positive finite number')
    return value


def record(row_id, row):
    return {'row_id': row_id, 'csv_line': row_id + 1,
            'configuration': {n: row[n] for n in CONFIGURATION},
            'observed_objectives': {n: row[n] for n in OBJECTIVES}}


def choose_rows(rows, max_estimators=100):
    """Select only fully observed recorded configurations, retaining original IDs."""
    cap = validate_cap(max_estimators)
    if not isinstance(rows, (list, tuple)):
        raise ValueError('rows must be a list or tuple')
    rows = normalize_records(rows) if rows else []
    complete, feasible, excluded = [], [], []
    incomplete_count = over_cap_count = 0
    for row_id, row in enumerate(rows, 1):
        missing_config = [n for n in CONFIGURATION if row[n] is None]
        missing_objectives = [n for n in OBJECTIVES if row[n] is None]
        above = row['N_estimators'] is not None and row['N_estimators'] > cap
        if not missing_config and not missing_objectives:
            complete.append((row_id, row))
            if not above:
                feasible.append((row_id, row))
            else:
                over_cap_count += 1
        else:
            incomplete_count += 1
        if missing_config or missing_objectives or above:
            excluded.append({'row_id': row_id, 'missing_configuration': missing_config,
                             'missing_objectives': missing_objectives, 'above_cap': above})
    report = {'status': 'selected' if feasible else 'no_feasible_candidate',
              'constraint': {'field': 'N_estimators', 'operator': '<=', 'cap': cap,
                             'measured_compute_budget': False, 'real_compute_budget_guarantee': False},
              'selection_rule': RULE, 'objective_directions': dict(OBJECTIVES),
              'counts': {'candidate': len(rows), 'complete': len(complete),
                         'excluded': len(excluded), 'feasible': len(feasible),
                         'incomplete': incomplete_count, 'complete_above_cap': over_cap_count},
              'count_convention': 'Candidate means every input row; excluded = incomplete + complete_above_cap; candidate = excluded + feasible. Missing reasons may overlap within excluded_records.',
              'excluded_records': excluded, 'selected_record': None, 'cap_evidence': None,
              'tie_evidence': None, 'tradeoffs': None, 'limitations': LIMITATIONS.copy(),
              'reuse': 'Steps 1–6 source and tests retained unchanged. Reuses Step 1 CSV loading/schema/directions, Step 6 record normalization, lexicographic selection key and direction-aware differences. Adds complete-configuration filtering, inclusive cap, explicit empty result, count/exclusion audit and feasible-set comparisons.',
              'environment': {'scikit_learn_version': sklearn.__version__}}
    if feasible:
        row_id, selected = min(feasible, key=selection_key)
        report['selected_record'] = record(row_id, selected)
        report['cap_evidence'] = {'recorded_N_estimators': selected['N_estimators'],
                                  'cap': cap, 'satisfies_inclusive_cap': selected['N_estimators'] <= cap}
        mre = [(i, r) for i, r in feasible if r['MRE-'] == selected['MRE-']]
        acc = [(i, r) for i, r in mre if r['ACC+'] == selected['ACC+']]
        pred = [(i, r) for i, r in acc if r['PRED40+'] == selected['PRED40+']]
        report['tie_evidence'] = {'minimum_mre_count': len(mre), 'after_acc_count': len(acc),
                                  'after_pred40_count': len(pred), 'final_tied_row_ids': [i for i, _ in pred]}
        champions = {}
        for name, direction in OBJECTIVES.items():
            i, r = min(feasible, key=lambda item: (
                item[1][name] if direction == 'minimize' else -item[1][name], item[0]))
            champions[name] = {**record(i, r), 'selected_difference': difference(selected[name], r[name], direction)}
        i, r = min(complete, key=selection_key)
        report['tradeoffs'] = {'feasible_objective_champions': champions,
                              'unconstrained_complete_winner': record(i, r),
                              'unconstrained_winner_satisfies_cap': r['N_estimators'] <= cap,
                              'scope': 'Champions use complete feasible rows, with earliest-row champion ties. Unconstrained reference uses complete recorded rows only; neither reference is independent validation.'}
    json.dumps(report, allow_nan=False)
    return report


def choose(dataset, max_estimators=100):
    _, rows = load_rows(dataset)
    report = choose_rows(rows, max_estimators)
    report['dataset_path'] = str(Path(dataset).resolve())
    return report


def render_text(report):
    lines = ['Retrospective recorded-configuration selection',
             f"Constraint: N_estimators <= {report['constraint']['cap']}",
             report['selection_rule'], 'Counts: ' + json.dumps(report['counts']),
             report['count_convention'], 'Status: ' + report['status']]
    if report['selected_record'] is None:
        lines.append('No feasible candidate; no configuration is selected.')
    else:
        lines.extend(['Selected recorded configuration: ' + json.dumps(report['selected_record']),
                      'Cap evidence: ' + json.dumps(report['cap_evidence']),
                      'Tie evidence: ' + json.dumps(report['tie_evidence']),
                      'Trade-offs: ' + json.dumps(report['tradeoffs'])])
    lines.extend(report['limitations'])
    lines.append(report['reuse'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--max-estimators', type=float, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        report = choose(args.dataset, args.max_estimators)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(render_text(report), end='')


if __name__ == '__main__':
    main()
