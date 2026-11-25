"""Justify a retrospective encoder recommendation using recorded evidence only."""

import argparse
import hashlib
import json
from pathlib import Path

from blame import diagnose
from remember import OBJECTIVE, load_measurements, summarize

TIE_RULE = ('Reuse Step 1 ascending Performance- ranking; exact numeric ties select '
            'the earliest CSV data row (1-based). No tolerance or setting preference.')
REFERENCE_RULE = ('Median Performance- of every historical measurement row, including '
                  'duplicates; average the two middle values for an even count. '
                  'This is a distribution reference, not an encoder configuration.')
LIMITATIONS = [
    'This is a retrospective recommendation among recorded rows, not an optimum over unseen configurations.',
    'Selecting the minimum from the same history creates selection optimism; a single best observation supplies no statistical certainty or repeatability guarantee.',
    'Historical rows may differ in workload, hardware, other settings, or unrecorded conditions. Rows are equally weighted, including duplicates; counts do not establish independent replication.',
    'Setting-level group means and eta squared are marginal associations, not explanations of this individual outcome or causal effects. Correlated scores are never added.',
    'Encoded setting names and values are preserved without inferring flag semantics or validity of unobserved combinations.',
    'Performance- is minimized; physical units are unspecified. No output-quality, memory, or resource-use measurements are inferred.',
    'No model predictions, p-values, confidence intervals, or significance claims are used in this justification. The historical median is not a randomized control.',
    'Validate the candidate under the administrator\'s workload with repeated controlled measurements and required quality/resource checks before operational adoption.',
    'Exact floating-point equality defines ties; near ties and rounding can affect ordering. Unsupported nonfinite computed statistics are rejected.',
]


def justify(data):
    """Return auditable observations, distribution context and marginal evidence.

    diagnose validates direct API inputs and supplies the existing Step 4 evidence.
    summarize supplies the preserved Step 1 ranking and historical statistics.
    """
    diagnosis = diagnose(data)
    summary = summarize(data)
    chosen = summary['top_runs'][0]
    value = chosen[OBJECTIVE]
    targets = [row[-1] for row in data.rows]
    n = len(targets)
    median = summary['objective']['statistics']['median']
    delta = value - median
    same_config = [i for i, row in enumerate(data.rows, 1)
                   if row[:-1] == data.rows[chosen['data_row'] - 1][:-1]]
    evidence = []
    for finding in diagnosis['ranked_findings'] + diagnosis['constant_settings']:
        name = finding['setting']
        selected_group = next(g for g in finding['values']
                              if g['value'] == chosen['settings'][name])
        evidence.append({
            'setting': name, 'recommended_value': chosen['settings'][name],
            'diagnosis_rank': finding['rank'], 'eta_squared': finding['score'],
            'selected_value_group': selected_group,
            'all_value_groups': finding['values'],
            'interpretation': ('Constant setting; no between-value comparison.'
                               if finding['score'] is None else
                               'Marginal group association only; not an individual contribution.'),
        })
    result = {
        'objective': OBJECTIVE, 'direction': 'minimize', 'units': None,
        'evidence_type': 'observed historical measurements only',
        'model_predictions': None,
        'recommendation': chosen,
        'selection': {
            'rule': TIE_RULE, 'tied_best_data_rows': [r['data_row'] for r in summary['best_runs']],
            'tied_best_count': len(summary['best_runs']),
            'same_configuration_data_rows': same_config,
            'same_configuration_observed_values': [targets[i - 1] for i in same_config],
        },
        'historical_distribution': {
            'statistics': summary['objective']['statistics'],
            'strictly_better_count': sum(y < value for y in targets),
            'equal_count': sum(y == value for y in targets),
            'strictly_worse_count': sum(y > value for y in targets),
            'strictly_worse_percent': 100 * sum(y > value for y in targets) / n,
            'competition_rank': 1 + sum(y < value for y in targets),
            'empirical_cdf_at_recommendation': sum(y <= value for y in targets) / n,
            'cdf_definition': 'Fraction of historical rows with Performance- <= recommended observation.',
            'unique_configurations': summary['unique_configurations'],
            'duplicate_measurement_rows': summary['duplicate_measurement_rows'],
            'repeated_configuration_rows': summary['repeated_configuration_rows'],
        },
        'historical_reference': {
            'rule': REFERENCE_RULE, OBJECTIVE: median,
            'difference_recommended_minus_reference': delta,
            'observed_reduction_reference_minus_recommended': -delta,
            'observed_reduction_percent_of_reference': (-delta / median * 100
                                                       if median > 0 else None),
            'percent_rule': '100*(reference-recommended)/reference only for positive reference; otherwise null.',
        },
        'setting_evidence': evidence,
        'diagnosis': diagnosis,
        'limitations': list(LIMITATIONS),
    }
    # Reject overflow in derived differences or statistics before returning evidence.
    json.dumps(result, allow_nan=False)
    return result


def render_text(report):
    """Render all numerical claims directly from structured computed evidence."""
    chosen = report['recommendation']
    selection = report['selection']
    dist = report['historical_distribution']
    ref = report['historical_reference']
    lines = [
        'Platform administrator: retrospective encoder recommendation',
        f"Recommend recorded settings {json.dumps(chosen['settings'], sort_keys=True)}.",
        f"Observed Performance-: {chosen[OBJECTIVE]} at CSV data row {chosen['data_row']}; minimize; units unspecified.",
        selection['rule'],
        f"Exact best-value ties: {selection['tied_best_count']}; data rows {selection['tied_best_data_rows']}.",
        f"Recorded outcomes for this same configuration: {selection['same_configuration_observed_values']} at data rows {selection['same_configuration_data_rows']}.",
        f"Historical summary: {json.dumps(dist['statistics'], sort_keys=True)}.",
        f"Competition rank {dist['competition_rank']}; strictly better/equal/strictly worse counts: {dist['strictly_better_count']}/{dist['equal_count']}/{dist['strictly_worse_count']}.",
        f"Strictly worse historical rows: {dist['strictly_worse_percent']}%; empirical CDF at chosen observation: {dist['empirical_cdf_at_recommendation']}.",
        dist['cdf_definition'],
        ref['rule'],
        f"Reference Performance-: {ref[OBJECTIVE]}; recommended minus reference: {ref['difference_recommended_minus_reference']}; observed reduction: {ref['observed_reduction_reference_minus_recommended']}.",
        f"Observed reduction as percent of reference: {ref['observed_reduction_percent_of_reference']}; {ref['percent_rule']}",
        'Setting-level context from the preserved marginal diagnosis (not contributions to the chosen outcome):',
    ]
    for entry in report['setting_evidence']:
        group = entry['selected_value_group']
        lines.append(
            f"  {entry['setting']}={entry['recommended_value']}: diagnosis rank={entry['diagnosis_rank']}; eta squared={entry['eta_squared']}; "
            f"matching-value group count={group['count']}, mean Performance-={group['objective']['mean']}, mean minus overall={group['mean_minus_overall']}. "
            + entry['interpretation'])
    lines.append('All value-group summaries and pairwise setting relationships are in justification.json under diagnosis; correlated scores are not summed.')
    lines.extend(report['limitations'])
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        paths = [args.output_dir / name for name in ('justification.json', 'justification.txt')]
        if any(p.resolve() == args.dataset.resolve() for p in paths):
            raise ValueError('Output must not overwrite the dataset')
        result = justify(load_measurements(args.dataset))
        result['provenance'] = {
            'dataset': str(args.dataset.resolve()),
            'sha256': hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            'data_row_convention': '1-based CSV data rows, excluding header',
            'reuse': ['remember.load_measurements', 'remember.summarize', 'blame.diagnose'],
        }
        serialized = json.dumps(result, indent=2, allow_nan=False) + '\n'
        rendered = render_text(result)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        paths[0].write_text(serialized, encoding='utf-8')
        paths[1].write_text(rendered, encoding='utf-8')
    except (OSError, ValueError, OverflowError) as error:
        parser.exit(2, f'error: {error}\n')
    print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
