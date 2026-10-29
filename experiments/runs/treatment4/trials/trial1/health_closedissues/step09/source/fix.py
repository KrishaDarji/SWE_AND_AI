"""Find the smallest recorded setting change with strictly lower observed MRE-."""
import argparse
import csv
import json
from pathlib import Path

import sklearn

from choose import record
from justify import difference, normalize_records
from summarize import CONFIGURATION, OBJECTIVES, load_rows

RULE = ('Among complete settings with observed MRE- strictly below baseline, '
        'minimize differing normalized configuration fields, then lower MRE-, '
        'then earlier one-based CSV data-row number. Exact normalized equality; '
        'secondary objectives do not break ties.')
LIMITATIONS = [
    'Offline search over recorded outcomes, not prediction of an unseen intervention or independent validation.',
    'Observed MRE reduction is not causal, repeatable, or statistically certified improvement. Selection over recorded outcomes can be optimistic.',
    'Change cost counts differing normalized fields as a proxy for change size, not measured intervention cost. Each numeric field counts once regardless of magnitude; criterion has no invented ordering.',
    'Duplicate settings remain separate observations. A lower outcome at identical settings identifies no setting change and does not establish repeatability.',
    'Missing settings or MRE exclude candidates; missing secondary outcomes remain null without imputation. Missingness may be selective.',
    'No estimator cap, compute cost, metric formula, operational cost model, or deployment guarantee is inferred. Recorded finite settings are not validated as estimator constructor parameters.',
    'CSV data-row numbers denote record positions, not chronology. Comparisons use finite normalized floats and exact equality; recorded precision can affect ties.'
]


def repair_rows(rows, baseline_row):
    """Return an auditable recorded repair for a one-based baseline row."""
    rows = normalize_records(rows)
    if isinstance(baseline_row, bool) or not isinstance(baseline_row, int):
        raise ValueError('baseline_row must be a one-based integer data-row number')
    if not 1 <= baseline_row <= len(rows):
        raise ValueError('baseline_row is outside the one-based data-row range')
    baseline = rows[baseline_row - 1]
    missing = [n for n in CONFIGURATION if baseline[n] is None]
    if missing or baseline['MRE-'] is None:
        raise ValueError('Baseline requires complete settings and observed MRE-; '
                         f'missing settings={missing}, missing MRE-={baseline["MRE-"] is None}')
    excluded, improving = [], []
    missing_settings = missing_mre = complete = 0
    for row_id, row in enumerate(rows, 1):
        absent = [n for n in CONFIGURATION if row[n] is None]
        no_mre = row['MRE-'] is None
        missing_settings += bool(absent)
        missing_mre += no_mre
        if absent or no_mre:
            excluded.append({'row_id': row_id, 'missing_configuration': absent,
                             'missing_MRE-': no_mre})
            continue
        complete += 1
        if row['MRE-'] < baseline['MRE-']:
            cost = sum(row[n] != baseline[n] for n in CONFIGURATION)
            improving.append((cost, row['MRE-'], row_id, row))
    report = {
        'status': 'recommended' if improving else 'no_improving_candidate',
        'baseline': record(baseline_row, baseline), 'recommendation': None,
        'changed_fields': [], 'change_cost': None, 'observed_MRE_reduction': None,
        'secondary_differences': None, 'setting_change_identified': False,
        'setting_change_statement': 'No improving recorded candidate; no setting change identified.',
        'selection_rule': RULE,
        'search_scope': 'All CSV data rows, including the baseline and duplicate settings; no estimator cap. Candidates require all five settings and observed MRE- only.',
        'counts': {'recorded_rows': len(rows), 'complete_settings_and_mre': complete,
                   'excluded': len(excluded), 'missing_settings': missing_settings,
                   'missing_mre': missing_mre, 'strictly_improving': len(improving),
                   'complete_not_improving': complete - len(improving)},
        'count_convention': 'recorded_rows = excluded + complete_settings_and_mre; complete_settings_and_mre = strictly_improving + complete_not_improving. Missing-reason counts overlap.',
        'excluded_records': excluded, 'tie_evidence': None,
        'objective_directions': dict(OBJECTIVES), 'limitations': LIMITATIONS.copy(),
        'reuse': 'Steps 1–7 modules and tests retained unchanged. Reuses CSV schema/loading, finite-float and categorical normalization, record rendering, and direction-aware objective differences. Adds baseline validation, minimal-field-change search and exclusion/tie audit.',
        'environment': {'scikit_learn_version': sklearn.__version__}}
    if improving:
        cost, mre, row_id, selected = min(improving, key=lambda item: item[:3])
        report['recommendation'] = record(row_id, selected)
        report['changed_fields'] = [{'field': n, 'old': baseline[n], 'new': selected[n]}
                                    for n in CONFIGURATION if baseline[n] != selected[n]]
        report['change_cost'] = cost
        report['observed_MRE_reduction'] = baseline['MRE-'] - mre
        report['setting_change_identified'] = cost > 0
        report['setting_change_statement'] = (
            'Recorded candidate changes the listed settings.' if cost else
            'Cost zero: identical settings have a lower recorded MRE-; no setting change was identified.')
        secondary = {}
        for name in ('ACC+', 'PRED40+'):
            old, new = baseline[name], selected[name]
            delta = (difference(new, old, OBJECTIVES[name]) if old is not None and new is not None
                     else {'selected_minus_reference': None,
                           'improvement_in_objective_direction': None, 'observed_relation': 'unavailable'})
            secondary[name] = {'baseline': old, 'recommended': new, **delta}
        report['secondary_differences'] = secondary
        min_cost = [item for item in improving if item[0] == cost]
        final_ties = [item[2] for item in min_cost if item[1] == mre]
        report['tie_evidence'] = {'minimum_change_cost': cost, 'minimum_cost_count': len(min_cost),
                                  'lowest_MRE_at_minimum_cost': mre, 'final_tied_row_ids': final_ties}
    json.dumps(report, allow_nan=False)
    return report


def repair(dataset, baseline_row):
    """Load the supplied dataset; never read a saved recommendation."""
    _, rows = load_rows(dataset)
    report = repair_rows(rows, baseline_row)
    report['dataset_path'] = str(Path(dataset).resolve())
    return report


def render_text(report):
    return json.dumps(report, indent=2, allow_nan=False) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--baseline-row', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        report = repair(args.dataset, args.baseline_row)
        rendered = render_text(report)
        args.output.write_text(rendered, encoding='utf-8')
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(rendered, end='')


if __name__ == '__main__':
    main()
