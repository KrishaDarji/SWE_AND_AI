"""Explain an MRE-first recorded configuration using descriptive evidence only."""
import argparse
import csv
import json
from pathlib import Path

import sklearn

from blame import METHOD, LIMITATIONS as DIAGNOSIS_LIMITATIONS, diagnose, partition
from certify import observed_summary
from flow import normalize_outcomes
from guess import validate_configuration
from summarize import CONFIGURATION, OBJECTIVES, load_rows, summarize

PREFERENCE = ('Illustrative MRE-first preference rule, not a universally best '
              'multi-objective recommendation. Minimize observed MRE-, then '
              'maximize ACC+, then maximize PRED40+, then earliest one-based CSV data-row number.')
LIMITATIONS = [
    'Selection optimism: choosing an extreme after examining all recorded outcomes can overstate future performance. The selected observation is not an independent validation result.',
    'A single best observation establishes neither statistical certainty nor repeatability. Repeated configurations are rows, not established independent replications.',
    'Marginal associations do not causally explain this individual outcome and must not be added as prediction contributions. Confounding, correlated settings, interactions and sparse groups limit interpretation.',
    'No predictions, overall score, significance tests or uncertainty intervals are used in this justification. All numerical evidence is observed or descriptive arithmetic.',
    'Metric formulas, units, theoretical ranges, compute costs and deployment guarantees are unspecified. A recorded zero MRE- is not proof of perfect future performance.',
    'Available-outcome historical distributions include the selected record and may use different subsets. Complete-outcome selection excludes any record missing an objective; missing hyperparameters remain explicit and require resolution.',
    'Historical means and medians are descriptive reference vectors, not runnable configurations. Historical objective champions are outcome-selected references, not independent controls.',
    'CSV row number identifies record order, not time. Finite floats and exact normalized equality define ties; extreme arithmetic that is not finite is rejected.',
    'The rule prioritizes MRE- before any loss in secondary objectives; platform preferences and acceptable trade-offs have not been supplied.'
]
CHECKS = [
    'Confirm metric definitions, directions, scales and the provenance of zero or unusual values; audit evaluation splits and leakage.',
    'Agree with the platform team on objective priorities, minimum acceptable ACC+ and PRED40+, and practical improvement thresholds before validation.',
    'Freeze the proposed configuration and evaluate it and agreed baselines on independent representative data with repeated evaluations and appropriate uncertainty assessment.',
    'Confirm hyperparameter names, supported values and missing settings against the actual estimator and deployed library version; the CSV is not an estimator constructor contract.',
    'Measure latency, memory and compute cost in the target environment, then assess data shift, monitoring, rollback and operational acceptance before deployment.'
]


def normalize_records(rows):
    """Normalize reusable inputs; omitted schema fields mean missing."""
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError('rows must be a nonempty list or tuple')
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - (set(CONFIGURATION) | set(OBJECTIVES)):
            raise ValueError('Records must be dictionaries containing only Health fields')
        normalized.append({**validate_configuration({n: row.get(n) for n in CONFIGURATION}),
                           **normalize_outcomes({n: row.get(n) for n in OBJECTIVES})})
    return normalized


def selection_key(item):
    row_id, row = item
    return row['MRE-'], -row['ACC+'], -row['PRED40+'], row_id


def difference(value, reference, direction):
    """Keep raw difference and direction-aware improvement separate."""
    delta = value - reference
    benefit = -delta if direction == 'minimize' else delta
    return {'selected_minus_reference': delta, 'improvement_in_objective_direction': benefit,
            'observed_relation': 'equal' if benefit == 0 else 'better' if benefit > 0 else 'worse'}


def justify_rows(rows):
    """Return traceable historical evidence without fitting or causal attribution."""
    rows = normalize_records(rows)
    complete = [(i, row) for i, row in enumerate(rows, 1)
                if all(row[n] is not None for n in OBJECTIVES)]
    excluded = [{'row_id': i, 'missing_objectives': [n for n in OBJECTIVES if r[n] is None]}
                for i, r in enumerate(rows, 1) if any(r[n] is None for n in OBJECTIVES)]
    if not complete:
        raise ValueError('No record has observed values for all three objectives')
    selected_id, selected = min(complete, key=selection_key)
    config = {n: selected[n] for n in CONFIGURATION}
    outcomes = {n: selected[n] for n in OBJECTIVES}
    mre_ties = [(i, r) for i, r in complete if r['MRE-'] == selected['MRE-']]
    acc_ties = [(i, r) for i, r in mre_ties if r['ACC+'] == selected['ACC+']]
    pred_ties = [(i, r) for i, r in acc_ties if r['PRED40+'] == selected['PRED40+']]
    positions = {}
    for name, direction in OBJECTIVES.items():
        values = [r[name] for r in rows if r[name] is not None]
        value = selected[name]
        below, equal, above = (sum(v < value for v in values), sum(v == value for v in values),
                               sum(v > value for v in values))
        stats = observed_summary(values, len(rows))
        best = min(values) if direction == 'minimize' else max(values)
        positions[name] = {'direction': direction, 'selected_value': value, 'summary': stats,
                           'strictly_below_count': below, 'equal_count': equal,
                           'strictly_above_count': above,
                           'rank_from_best_competition': 1 + (below if direction == 'minimize' else above),
                           'strictly_better_count': below if direction == 'minimize' else above,
                           'lower_tail_midrank_percentile': 100 * (below + equal / 2) / len(values),
                           'references': {key: {'value': ref, **difference(value, ref, direction)}
                                          for key, ref in [('historical_mean', stats['mean']),
                                                           ('historical_median', stats['median']),
                                                           ('historical_best', best)]}}
    reference_records = {}
    first_id, first = complete[0]
    references = [('earliest_complete_record', first_id, first)]
    for name, direction in OBJECTIVES.items():
        champion_id, champion = min(complete, key=lambda item: (
            item[1][name] if direction == 'minimize' else -item[1][name], item[0]))
        references.append((name + '_champion', champion_id, champion))
    for label, row_id, row in references:
        reference_records[label] = {'row_id': row_id, 'configuration': {n: row[n] for n in CONFIGURATION},
                                    'observed_objectives': {n: row[n] for n in OBJECTIVES},
                                    'differences': {n: difference(selected[n], row[n], d)
                                                    for n, d in OBJECTIVES.items()}}
    diagnosis = diagnose(rows)
    by_name = {f['field']: f for f in diagnosis['ranked_findings'] + diagnosis['unranked_fields']}
    settings = {}
    for name in CONFIGURATION:
        field = by_name[name]
        _, memberships = partition([r[name] for r in rows], name == 'criterion')
        group_id = memberships[selected_id - 1]
        settings[name] = {'selected_value': selected[name], 'rank': field['rank'],
                          'eta_squared': field['eta_squared'], 'status': field['status'],
                          'grouping': field['grouping'], 'included_mre': field['included_mre'],
                          'selected_group': None if group_id is None else field['groups'][group_id],
                          'missing_predictor_mre': field['missing_predictor_mre']}
    same_ids = [i for i, r in enumerate(rows, 1) if all(r[n] == selected[n] for n in CONFIGURATION)]
    dominators = []
    for i, r in complete:
        benefits = [difference(r[n], selected[n], d)['improvement_in_objective_direction']
                    for n, d in OBJECTIVES.items()]
        if all(b >= 0 for b in benefits) and any(b > 0 for b in benefits):
            dominators.append(i)
    report = {'preference': PREFERENCE, 'record_count': len(rows),
              'objective_directions': dict(OBJECTIVES),
              'selection': {'eligible_count': len(complete), 'excluded_count': len(excluded),
                            'excluded_records': excluded,
                            'minimum_mre_tie_count': len(mre_ties),
                            'remaining_after_acc_tie_count': len(acc_ties),
                            'remaining_after_pred40_tie_count': len(pred_ties),
                            'final_tied_row_ids': [i for i, _ in pred_ties],
                            'ties': 'Exact equality of normalized finite floats; final tie uses earliest data row.'},
              'recommended_record': {'row_id': selected_id, 'csv_line': selected_id + 1,
                                     'configuration': config, 'observed_objectives': outcomes,
                                     'missing_configuration_fields': [n for n in CONFIGURATION if config[n] is None]},
              'historical_positions': positions, 'reference_records': reference_records,
              'reference_conventions': {
                  'distribution': 'All nonmissing values per objective, including selected and otherwise excluded records; no imputation.',
                  'mean_median': 'All available outcomes per objective; vectors are not runnable configurations.',
                  'record_references': 'Earliest complete row and each objective champion among complete rows; champion ties use earliest row.',
                  'percentile': '100*(strictly below + 0.5*equal)/nonmissing count; lower-tail midrank, irrespective of objective direction.',
                  'rank': '1 + strictly better count in the objective direction; equality shares competition rank.',
                  'differences': 'Raw selected minus reference; improvement reverses sign only for minimized objectives.'},
              'setting_evidence': settings, 'diagnosis_method': METHOD,
              'diagnosis_limitations': DIAGNOSIS_LIMITATIONS,
              'same_configuration_evidence': {'row_ids': same_ids, 'record_count': len(same_ids),
                   'objectives': {n: observed_summary([rows[i-1][n] for i in same_ids], len(same_ids))
                                  for n in OBJECTIVES}},
              'tradeoffs': {'dominating_complete_row_ids': dominators,
                            'nondominated_among_complete_records': not dominators,
                            'scope': 'Finite complete-outcome historical records only; nondominance does not establish universal optimality.'},
              'evidence_scope': {'observed': 'Recorded outcomes, distribution summaries, differences and marginal diagnosis.',
                                 'predictions': None, 'assumptions': 'MRE-first priorities are illustrative; metric semantics and representativeness need external confirmation.'},
              'limitations': LIMITATIONS, 'deployment_checks': CHECKS,
              'environment': {'scikit_learn_version': sklearn.__version__},
              'reuse': 'Steps 1–5 source/tests unchanged. Reuses loading and file summaries, configuration/outcome normalization, observed comparison summaries and marginal diagnosis.'}
    json.dumps(report, allow_nan=False)
    return report


def justify(dataset):
    """Use the preserved CSV loader and summary on the supplied dataset path."""
    _, rows = load_rows(dataset)
    report = justify_rows(rows)
    historical = summarize(dataset)
    report['dataset_path'] = historical['dataset_path']
    report['historical_data_quality'] = {key: historical[key] for key in (
        'record_count', 'duplicate_records', 'repeated_configurations')}
    return report


def render_text(report):
    """Render all numeric claims directly from structured evidence."""
    record = report['recommended_record']
    selection = report['selection']
    lines = ['Platform-team configuration justification', report['preference'],
             f"Recommend recorded data row {record['row_id']} (CSV line {record['csv_line']}) for independent validation.",
             'Hyperparameters: ' + json.dumps(record['configuration'], sort_keys=True),
             'Recorded objectives: ' + json.dumps(record['observed_objectives'], sort_keys=True),
             f"Eligible {selection['eligible_count']} of {report['record_count']}; excluded {selection['excluded_count']} for missing objectives.",
             'Excluded records and missing outcomes: ' + json.dumps(selection['excluded_records']),
             f"Tied at minimum MRE-: {selection['minimum_mre_tie_count']}; after higher ACC+: {selection['remaining_after_acc_tie_count']}; after higher PRED40+: {selection['remaining_after_pred40_tie_count']}; final row IDs: {selection['final_tied_row_ids']}.",
             selection['ties'], 'Missing hyperparameters: ' + json.dumps(record['missing_configuration_fields']),
             '\nHistorical distributions and references (all available values per objective, including selected):']
    for name, pos in report['historical_positions'].items():
        lines.append(f"{name} ({pos['direction']}): observed={pos['selected_value']}; summary={json.dumps(pos['summary'])}; best-first rank={pos['rank_from_best_competition']}; below/equal/above={pos['strictly_below_count']}/{pos['equal_count']}/{pos['strictly_above_count']}; lower-tail midrank percentile={pos['lower_tail_midrank_percentile']}.")
        for label, ref in pos['references'].items():
            lines.append(f"  {label}: {json.dumps(ref)}")
    lines.extend(report['reference_conventions'].values())
    lines.append('\nHistorical record comparisons (positive improvement means better in that objective direction):')
    for label, ref in report['reference_records'].items():
        lines.append(f"{label}: row {ref['row_id']}; recorded objectives={json.dumps(ref['observed_objectives'])}; selected differences={json.dumps(ref['differences'])}.")
    lines.append('\nRelevant setting-level diagnosis of observed MRE-:')
    lines.append(METHOD['grouping'])
    for name, setting in report['setting_evidence'].items():
        lines.append(f"{name}={setting['selected_value']}; marginal rank={setting['rank']}; eta-squared={setting['eta_squared']}; status={setting['status']}; selected group={json.dumps(setting['selected_group'])}; field included MRE summary={json.dumps(setting['included_mre'])}.")
        if setting['selected_group'] is None:
            lines.append('  Missing selected setting; no nonmissing group attribution. Missing-setting MRE: ' + json.dumps(setting['missing_predictor_mre']))
    lines.append('These are marginal associations, not causal explanations or additive prediction contributions; no predictions are used.')
    lines.append('Same configuration evidence: ' + json.dumps(report['same_configuration_evidence']))
    lines.append('Historical Pareto check: ' + json.dumps(report['tradeoffs']))
    lines.append('Secondary-objective losses are shown explicitly against their historical best and champion records above. No overall score or practical acceptance threshold is assumed.')
    lines.append('\nLimitations:')
    lines.extend('- ' + item for item in report['limitations'])
    lines.append('\nChecks before deployment:')
    lines.extend('- ' + item for item in report['deployment_checks'])
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    try:
        if sklearn.__version__ != '1.6.0':
            raise ValueError('Use scikit-learn 1.6.0')
        report = justify(args.dataset)
        rendered = render_text(report)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'justification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        (args.output_dir / 'justification.txt').write_text(rendered, encoding='utf-8')
    except (OSError, ValueError, OverflowError, csv.Error) as exc:
        parser.exit(2, f'Error: {exc}\n')
    print(rendered, end='')


if __name__ == '__main__':
    main()
